/*
 * test_wasm.c — Tests for WASM binary builder and WASM renderer
 */

#include "test_harness.h"
#include "../src/codegen.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/tensor.h"
#include "../src/wasm_builder.h"

static int wasm_run_c_i32(PolyUOp **lin, int n_lin, const char *fn_name, int32_t *out) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_i64(PolyUOp **lin, int n_lin, const char *fn_name, int64_t *out) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_f32_buffer(PolyUOp **lin, int n_lin, const char *fn_name, float *buf) {
  char *src = poly_render_c(lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {buf};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_write_module(const char *path, const uint8_t *wasm, int wasm_size) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  size_t written = fwrite(wasm, 1, (size_t)wasm_size, f);
  int close_rc = fclose(f);
  return (written == (size_t)wasm_size && close_rc == 0) ? 0 : -1;
}

static int wasm_count_simd_opcode(const uint8_t *wasm, int wasm_size, int opcode) {
  int count = 0;
  for (int i = 0; wasm && i + 1 < wasm_size; i++)
    if (wasm[i] == WASM_SIMD_PREFIX && wasm[i + 1] == opcode) count++;
  return count;
}

static bool wasm_file_contains_relaxed_madd(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  int a = -1, b = -1, c = 0;
  bool found = false;
  while ((c = fgetc(f)) != EOF) {
    if (a == WASM_SIMD_PREFIX && b == 0x85 && c == 0x02) {
      found = true;
      break;
    }
    a = b;
    b = c;
  }
  fclose(f);
  return found;
}

static const char *poly_test_node_cmd(void) {
  static int initialized = 0;
  static char cmd[512];
  if (!initialized) {
    initialized = 1;
    const char *env = getenv("POLY_TEST_NODE");
    if (env && env[0]) {
      snprintf(cmd, sizeof(cmd), "%s", env);
    } else if (system("command -v node > /dev/null 2>&1") == 0) {
      snprintf(cmd, sizeof(cmd), "%s", "node");
    }
  }
  return cmd[0] ? cmd : NULL;
}

static bool poly_test_node_relaxed_probe(const char *node_cmd) {
  if (!node_cmd || !node_cmd[0]) return false;
  const char *probe =
      "const b=Buffer.from('AGFzbQEAAAABBAFgAAACDwEDZW52Bm1lbW9yeQIAAQMCAQAHCgEGa2VybmVsAAAKQwFBAEEA/QwAAABAAAAAQAAAAEAAAABA/QwAAEBAAABAQAAAQEAAAEBA/QwAAKBAAACgQAAAoEAAAKBA/YUC/QsEAAs=','base64');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const inst=new WebAssembly.Instance(new WebAssembly.Module(b),{env:{memory:mem}});"
      "inst.exports.kernel();"
      "const got=new Float32Array(mem.buffer)[0];"
      "if(Math.abs(got-11)>1e-6){console.error('relaxed_madd probe got '+got);process.exit(3)}";
  char cmd[4096];
  snprintf(cmd, sizeof(cmd), "%s -e \"%s\" > /dev/null 2>&1", node_cmd, probe);
  return system(cmd) == 0;
}

static const char *poly_test_node_relaxed_cmd(void) {
  static int initialized = 0;
  static char cmd[512];
  if (!initialized) {
    initialized = 1;
    const char *base = poly_test_node_cmd();
    char wasm_flag[768] = {0};
    char relaxed_flag[768] = {0};
    if (base && base[0]) {
      snprintf(wasm_flag, sizeof(wasm_flag), "%s --experimental-wasm-relaxed-simd", base);
      snprintf(relaxed_flag, sizeof(relaxed_flag), "%s --experimental-relaxed-simd", base);
    }
    const char *candidates[4] = {base, wasm_flag, relaxed_flag, NULL};
    for (int i = 0; candidates[i]; i++) {
      if (!candidates[i][0]) continue;
      if (poly_test_node_relaxed_probe(candidates[i])) {
        snprintf(cmd, sizeof(cmd), "%s", candidates[i]);
        break;
      }
    }
  }
  return cmd[0] ? cmd : NULL;
}

static const char *poly_test_node_cmd_for_wasm(const char *path) {
  if (wasm_file_contains_relaxed_madd(path)) return poly_test_node_relaxed_cmd();
  return poly_test_node_cmd();
}

static PolyUOp **wasm_linearize_generic_test(PolyCtx *ctx, PolyUOp *sink, int *n_out) {
  PolyRewriteOpts opts = {
      .optimize = true,
      .devectorize = 1,
      .caps =
          {
              .has_mulacc = true,
              .has_threefry = false,
              .has_local = false,
              .has_simd_int = false,
              .has_simd_float = true,
              .max_vec_width = 4,
          },
      .device = POLY_DEVICE_WASM,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  return poly_linearize_ex(ctx, sink, opts, n_out);
}

static int node_compile_wasm_module(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');new WebAssembly.Module(fs.readFileSync('%s'))\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_i32(const char *path, int32_t expected) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "inst.exports.kernel(0);"
      "const got=new DataView(mem.buffer).getInt32(0,true);"
      "if(got!==%d){console.error('got '+got+' expected %d');process.exit(2)}\"",
      node, path, expected, expected
  );
  return system(cmd);
}

static int node_run_wasm_f32_buffer(const char *path, float expected) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "const dv=new DataView(mem.buffer);"
      "dv.setFloat32(4,2,true);dv.setFloat32(8,3,true);dv.setFloat32(12,5,true);"
      "inst.exports.kernel(0);"
      "const got=dv.getFloat32(0,true);const expected=%.9g;"
      "if(Math.abs(got-expected)>1e-6){console.error('got '+got+' expected "
      "'+expected);process.exit(2)}\"",
      node, path, (double)expected
  );
  return system(cmd);
}

static int node_run_wasm_where_f32(const char *path, int n) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const N=%d;"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "const f=new Float32Array(mem.buffer);"
      "const offA=0,offB=N,offC=N*2;"
      "for(let i=0;i<N;i++){f[offA+i]=i-3;f[offB+i]=100+i;f[offC+i]=0;}"
      "inst.exports.kernel(offA*4,offB*4,offC*4);"
      "for(let i=0;i<N;i++){"
      " const a=i-3,b=100+i,exp=(a<2.5)?a+1:b-1;"
      " const got=f[offC+i];"
      " if(Math.abs(got-exp)>1e-6){console.error('i='+i+' got '+got+' expected '+exp);process.exit(2);}"
      "}\"",
      node, n, path
  );
  return system(cmd);
}

static int node_run_wasm_reg_group_f32(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "const f=new Float32Array(mem.buffer);"
      "const inOff=0,outOff=16;"
      "for(let i=0;i<4;i++){f[inOff+i]=i+2;f[outOff+i]=0;}"
      "inst.exports.kernel(inOff*4,outOff*4);"
      "for(let i=0;i<4;i++){const exp=i+3,got=f[outOff+i];"
      " if(Math.abs(got-exp)>1e-6){console.error('i='+i+' got '+got+' expected '+exp);process.exit(2);}"
      "}\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_wide_vector_gep_f32(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "const f=new Float32Array(mem.buffer);"
      "inst.exports.kernel(0);"
      "for(let i=0;i<16;i++){"
      " const src=i+0.25;"
      " const exp=src+1.0;"
      " const got=f[i];"
      " if(Math.abs(got-exp)>1e-6){console.error('i='+i+' got '+got+' expected '+exp);process.exit(2);}"
      "}\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_broadcast_reduce_relu(const char *path, int n) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;

  const char *js_path = "/tmp/polygrad_test_broadcast_reduce_relu.js";
  FILE *f = fopen(js_path, "w");
  if (!f) return -1;
  fprintf(
      f,
      "const fs=require('fs');\n"
      "const N=%d;\n"
      "const outF=0;\n"
      "const xF=N;\n"
      "const rowF=xF+N*N;\n"
      "const colF=rowF+N;\n"
      "const totalF=colF+N+16;\n"
      "const mem=new WebAssembly.Memory({initial:Math.ceil(totalF*4/65536)+1});\n"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};\n"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));\n"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});\n"
      "const a=new Float32Array(mem.buffer);\n"
      "for(let r=0;r<N;r++){\n"
      "  a[rowF+r]=((r*5)%%23-11)/31;\n"
      "  a[colF+r]=((r*11)%%29-14)/29;\n"
      "  for(let c=0;c<N;c++) a[xF+r*N+c]=((r*13+c*7)%%37-18)/37;\n"
      "}\n"
      "inst.exports.kernel(outF*4,xF*4,rowF*4,colF*4);\n"
      "const checks=[0,1,17,N-1];\n"
      "for(const r of checks){\n"
      "  let exp=0;\n"
      "  for(let k=0;k<N;k++){\n"
      "    let v=(a[xF+r*N+k]+a[rowF+r])*a[colF+k]-0.25;\n"
      "    if(v<0) v=0;\n"
      "    exp+=v;\n"
      "  }\n"
      "  const got=a[outF+r];\n"
      "  if(Math.abs(got-exp)>2e-2){\n"
      "    console.error('row '+r+' got '+got+' expected '+exp+' diff '+Math.abs(got-exp));\n"
      "    process.exit(2);\n"
      "  }\n"
      "}\n",
      n, path
  );
  if (fclose(f) != 0) return -1;

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s %s", node, js_path);
  return system(cmd);
}

static int node_run_wasm_matmul_bias_relu(
    const char *path,
    int tokens,
    int d,
    int hidden
) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;

  const char *js_path = "/tmp/polygrad_test_matmul_bias_relu.js";
  FILE *f = fopen(js_path, "w");
  if (!f) return -1;
  fprintf(
      f,
      "const fs=require('fs');\n"
      "const T=%d,D=%d,H=%d;\n"
      "const outF=0;\n"
      "const xF=outF+T*H;\n"
      "const wF=xF+T*D;\n"
      "const bF=wF+H*D;\n"
      "const totalF=bF+H+16;\n"
      "const mem=new WebAssembly.Memory({initial:Math.ceil(totalF*4/65536)+1});\n"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};\n"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));\n"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});\n"
      "const a=new Float32Array(mem.buffer);\n"
      "for(let i=0;i<T*D;i++) a[xF+i]=((i*17+13)%%101-50)/504;\n"
      "for(let i=0;i<H*D;i++) a[wF+i]=((i*19+7)%%103-51)/611;\n"
      "for(let i=0;i<H;i++) a[bF+i]=((i*23+5)%%31-15)/257;\n"
      "inst.exports.kernel(outF*4,xF*4,wF*4,bF*4);\n"
      "const checks=[[0,0],[0,1],[3,5],[T-1,H-1]];\n"
      "for(const [r,h] of checks){\n"
      "  let exp=a[bF+h];\n"
      "  for(let k=0;k<D;k++) exp+=a[xF+r*D+k]*a[wF+h*D+k];\n"
      "  if(exp<0) exp=0;\n"
      "  const got=a[outF+r*H+h];\n"
      "  if(Math.abs(got-exp)>1e-3){\n"
      "    console.error('cell '+r+','+h+' got '+got+' expected '+exp+' diff '+Math.abs(got-exp));\n"
      "    process.exit(2);\n"
      "  }\n"
      "}\n",
      tokens, d, hidden, path
  );
  if (fclose(f) != 0) return -1;

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s %s", node, js_path);
  return system(cmd);
}

static int node_run_wasm_matmul_abt_row1(const char *path, int n, int k) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  const char *js_path = "/tmp/polygrad_test_matmul_abt_row1.js";
  FILE *f = fopen(js_path, "w");
  if (!f) return -1;
  fprintf(
      f,
      "const fs=require('fs');\n"
      "const N=%d,K=%d;\n"
      "const outF=0;\n"
      "const aF=outF+N;\n"
      "const bF=aF+K;\n"
      "const totalF=bF+N*K+16;\n"
      "const mem=new WebAssembly.Memory({initial:Math.ceil(totalF*4/65536)+1});\n"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};\n"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));\n"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});\n"
      "const a=new Float32Array(mem.buffer);\n"
      "for(let i=0;i<K;i++) a[aF+i]=((i*13+5)%%17-8)/9;\n"
      "for(let c=0;c<N;c++) for(let i=0;i<K;i++) a[bF+c*K+i]=((c*7+i*11+3)%%19-9)/11;\n"
      "inst.exports.kernel(outF*4,aF*4,bF*4);\n"
      "for(let c=0;c<N;c++){\n"
      "  let exp=0;\n"
      "  for(let i=0;i<K;i++) exp+=a[aF+i]*a[bF+c*K+i];\n"
      "  const got=a[outF+c];\n"
      "  if(Math.abs(got-exp)>1e-4){\n"
      "    console.error('col '+c+' got '+got+' expected '+exp+' diff '+Math.abs(got-exp));\n"
      "    process.exit(2);\n"
      "  }\n"
      "}\n",
      n, k, path
  );
  if (fclose(f) != 0) return -1;

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s %s", node, js_path);
  return system(cmd);
}

static int node_run_wasm_matmul_ab(const char *path, int m, int n, int k) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  const char *js_path = "/tmp/polygrad_test_matmul_ab.js";
  FILE *f = fopen(js_path, "w");
  if (!f) return -1;
  fprintf(
      f,
      "const fs=require('fs');\n"
      "const M=%d,N=%d,K=%d;\n"
      "const outF=0;\n"
      "const aF=outF+M*N;\n"
      "const bF=aF+M*K;\n"
      "const totalF=bF+K*N+16;\n"
      "const mem=new WebAssembly.Memory({initial:Math.ceil(totalF*4/65536)+1});\n"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};\n"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));\n"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});\n"
      "const a=new Float32Array(mem.buffer);\n"
      "for(let r=0;r<M;r++) for(let i=0;i<K;i++) a[aF+r*K+i]=((r*17+i*13+5)%%23-11)/13;\n"
      "for(let i=0;i<K;i++) for(let c=0;c<N;c++) a[bF+i*N+c]=((i*7+c*11+3)%%29-14)/17;\n"
      "inst.exports.kernel(outF*4,aF*4,bF*4);\n"
      "for(let r=0;r<M;r++) for(let c=0;c<N;c++){\n"
      "  let exp=0;\n"
      "  for(let i=0;i<K;i++) exp+=a[aF+r*K+i]*a[bF+i*N+c];\n"
      "  const got=a[outF+r*N+c];\n"
      "  if(Math.abs(got-exp)>1e-4){\n"
      "    console.error('row '+r+' col '+c+' got '+got+' expected '+exp+' diff '+Math.abs(got-exp));\n"
      "    process.exit(2);\n"
      "  }\n"
      "}\n",
      m, n, k, path
  );
  if (fclose(f) != 0) return -1;

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s %s", node, js_path);
  return system(cmd);
}

static int node_run_wasm_matmul_abt(const char *path, int m, int n, int k) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  const char *js_path = "/tmp/polygrad_test_matmul_abt.js";
  FILE *f = fopen(js_path, "w");
  if (!f) return -1;
  fprintf(
      f,
      "const fs=require('fs');\n"
      "const M=%d,N=%d,K=%d;\n"
      "const outF=0;\n"
      "const aF=outF+M*N;\n"
      "const bF=aF+M*K;\n"
      "const totalF=bF+N*K+16;\n"
      "const mem=new WebAssembly.Memory({initial:Math.ceil(totalF*4/65536)+1});\n"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};\n"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));\n"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});\n"
      "const a=new Float32Array(mem.buffer);\n"
      "for(let r=0;r<M;r++) for(let i=0;i<K;i++) a[aF+r*K+i]=((r*17+i*13+5)%%23-11)/13;\n"
      "for(let c=0;c<N;c++) for(let i=0;i<K;i++) a[bF+c*K+i]=((c*7+i*11+3)%%29-14)/17;\n"
      "inst.exports.kernel(outF*4,aF*4,bF*4);\n"
      "for(let r=0;r<M;r++) for(let c=0;c<N;c++){\n"
      "  let exp=0;\n"
      "  for(let i=0;i<K;i++) exp+=a[aF+r*K+i]*a[bF+c*K+i];\n"
      "  const got=a[outF+r*N+c];\n"
      "  if(Math.abs(got-exp)>1e-4){\n"
      "    console.error('row '+r+' col '+c+' got '+got+' expected '+exp+' diff '+Math.abs(got-exp));\n"
      "    process.exit(2);\n"
      "  }\n"
      "}\n",
      m, n, k, path
  );
  if (fclose(f) != 0) return -1;

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s %s", node, js_path);
  return system(cmd);
}

static int node_run_wasm_i64(const char *path, int64_t expected) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const math={exp2f:x=>Math.pow(2,x),log2f:Math.log2,sinf:Math.sin,powf:Math.pow};"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math});"
      "inst.exports.kernel(0);"
      "const got=new DataView(mem.buffer).getBigInt64(0,true);"
      "const expected=BigInt('%lld');"
      "if(got!==expected){console.error('got '+got+' expected '+expected);process.exit(2)}\"",
      node, path, (long long)expected
  );
  return system(cmd);
}

/* WASM builder tests */

TEST(wasm, leb128_unsigned) {
  WasmBuf b;
  wb_init(&b);

  /* 0 → 0x00 */
  wb_uleb128(&b, 0);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x00);

  /* 127 → 0x7F */
  b.len = 0;
  wb_uleb128(&b, 127);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x7F);

  /* 128 → 0x80 0x01 */
  b.len = 0;
  wb_uleb128(&b, 128);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0x80);
  ASSERT_INT_EQ(b.data[1], 0x01);

  /* 624485 → 0xE5 0x8E 0x26 */
  b.len = 0;
  wb_uleb128(&b, 624485);
  ASSERT_INT_EQ(b.len, 3);
  ASSERT_INT_EQ(b.data[0], 0xE5);
  ASSERT_INT_EQ(b.data[1], 0x8E);
  ASSERT_INT_EQ(b.data[2], 0x26);

  wb_free(&b);
  PASS();
}

TEST(wasm, leb128_signed) {
  WasmBuf b;
  wb_init(&b);

  /* 0 → 0x00 */
  wb_sleb128(&b, 0);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x00);

  /* -1 → 0x7F */
  b.len = 0;
  wb_sleb128(&b, -1);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x7F);

  /* 63 → 0x3F */
  b.len = 0;
  wb_sleb128(&b, 63);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x3F);

  /* -64 → 0x40 */
  b.len = 0;
  wb_sleb128(&b, -64);
  ASSERT_INT_EQ(b.len, 1);
  ASSERT_INT_EQ(b.data[0], 0x40);

  /* 64 → 0xC0 0x00 */
  b.len = 0;
  wb_sleb128(&b, 64);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0xC0);
  ASSERT_INT_EQ(b.data[1], 0x00);

  /* -65 → 0xBF 0x7F */
  b.len = 0;
  wb_sleb128(&b, -65);
  ASSERT_INT_EQ(b.len, 2);
  ASSERT_INT_EQ(b.data[0], 0xBF);
  ASSERT_INT_EQ(b.data[1], 0x7F);

  wb_free(&b);
  PASS();
}

TEST(wasm, module_header) {
  WasmBuf b;
  wb_init(&b);
  wb_module_header(&b);

  ASSERT_INT_EQ(b.len, 8);
  /* Magic: \0asm */
  ASSERT_INT_EQ(b.data[0], 0x00);
  ASSERT_INT_EQ(b.data[1], 0x61);
  ASSERT_INT_EQ(b.data[2], 0x73);
  ASSERT_INT_EQ(b.data[3], 0x6D);
  /* Version: 1 */
  ASSERT_INT_EQ(b.data[4], 0x01);
  ASSERT_INT_EQ(b.data[5], 0x00);
  ASSERT_INT_EQ(b.data[6], 0x00);
  ASSERT_INT_EQ(b.data[7], 0x00);

  wb_free(&b);
  PASS();
}

TEST(wasm, section_encoding) {
  WasmBuf content;
  wb_init(&content);
  wb_byte(&content, 0xAA);
  wb_byte(&content, 0xBB);
  wb_byte(&content, 0xCC);

  WasmBuf out;
  wb_init(&out);
  wb_section(&out, 0x07, &content); /* export section */

  /* Expected: section_id(0x07) + length(3) + content(AA BB CC) */
  ASSERT_INT_EQ(out.len, 5);
  ASSERT_INT_EQ(out.data[0], 0x07); /* section id */
  ASSERT_INT_EQ(out.data[1], 0x03); /* length = 3 */
  ASSERT_INT_EQ(out.data[2], 0xAA);
  ASSERT_INT_EQ(out.data[3], 0xBB);
  ASSERT_INT_EQ(out.data[4], 0xCC);

  wb_free(&content);
  wb_free(&out);
  PASS();
}

TEST(wasm, name_encoding) {
  WasmBuf b;
  wb_init(&b);
  wb_name(&b, "env");

  ASSERT_INT_EQ(b.len, 4); /* 1 byte length + 3 bytes */
  ASSERT_INT_EQ(b.data[0], 3);
  ASSERT_INT_EQ(b.data[1], 'e');
  ASSERT_INT_EQ(b.data[2], 'n');
  ASSERT_INT_EQ(b.data[3], 'v');

  wb_free(&b);
  PASS();
}

TEST(wasm, f32_encoding) {
  WasmBuf b;
  wb_init(&b);
  float v = 1.0f;
  wb_f32(&b, v);

  ASSERT_INT_EQ(b.len, 4);
  /* IEEE 754: 1.0f = 0x3F800000 (little-endian: 00 00 80 3F) */
  ASSERT_INT_EQ(b.data[0], 0x00);
  ASSERT_INT_EQ(b.data[1], 0x00);
  ASSERT_INT_EQ(b.data[2], 0x80);
  ASSERT_INT_EQ(b.data[3], 0x3F);

  wb_free(&b);
  PASS();
}

/* Helper: build c[i] = a[i] OP b[i] kernel IR (same as test_codegen.c) */

typedef struct {
  PolyCtx *ctx;
  PolyUOp *sink;
  int n;
} WasmVecKernel;

static WasmVecKernel wasm_make_vec_binop(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* Helper: build one direct f32x4 vector op over vector pointer params. */
static WasmVecKernel wasm_make_direct_vec_binop(PolyOps alu_op) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType f32x4 = poly_dtype_vec(POLY_FLOAT32, 4);
  PolyDType ptr_f32x4 = poly_dtype_ptr(f32x4, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32x4, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32x4, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32x4, poly_arg_int(2));

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, f32x4, p0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, f32x4, p1, poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, alu_op, f32x4, load0, load1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, p2, alu, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());

  return (WasmVecKernel){ctx, sink, 4};
}

/* Helper: build b[i] = OP(a[i]) unary kernel */
static WasmVecKernel wasm_make_vec_unary(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT32, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* Helper: c[i] = where(a[i] < 2.5, a[i] + 1, b[i] - 1) */
static WasmVecKernel wasm_make_vec_where_f32(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());

  PolyUOp *a = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *b = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *cutoff = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.5f));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0f));
  PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, a, cutoff, poly_arg_none());
  PolyUOp *if_true = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, a, one, poly_arg_none());
  PolyUOp *if_false = poly_uop2(ctx, POLY_OP_SUB, POLY_FLOAT32, b, one, poly_arg_none());
  PolyUOp *where_src[3] = {cond, if_true, if_false};
  PolyUOp *out = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, where_src, 3, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, out, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* WASM renderer tests */

TEST(wasm, render_vecadd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  /* Must produce non-empty output */
  ASSERT_TRUE(wasm != NULL);
  ASSERT_TRUE(wasm_size > 8);

  /* Must start with WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);
  ASSERT_INT_EQ(wasm[2], 0x73);
  ASSERT_INT_EQ(wasm[3], 0x6D);

  /* Must have sections (type=1 should appear after header) */
  ASSERT_INT_EQ(wasm[8], 0x01); /* type section id */

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_vecmul) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_MUL, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Valid WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_unary) {
  WasmVecKernel k = wasm_make_vec_unary(POLY_OP_NEG, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Look for NEG opcode (0x8C) somewhere in the binary */
  bool found_neg = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F32_NEG) {
      found_neg = true;
      break;
    }
  }
  ASSERT_TRUE(found_neg);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_chain) {
  /* d = (a + b) * c — fused kernel */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  int N = 10;

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(2));
  PolyUOp *p3 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(3));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p2, range, poly_arg_none());
  PolyUOp *idx3 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, p3, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *lc = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, la, lb, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, lc, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mul, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain both ADD and MUL opcodes */
  bool found_add = false, found_mul = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F32_ADD) found_add = true;
    if (wasm[i] == WASM_OP_F32_MUL) found_mul = true;
  }
  ASSERT_TRUE(found_add);
  ASSERT_TRUE(found_mul);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_unsigned_alu_opcodes) {
  /* out[i] = (a[i] // b[i]) + (a[i] % b[i]) + (a[i] >> 1), uint32 path */
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_u32 = poly_dtype_ptr(POLY_UINT32, -1, POLY_ADDR_GLOBAL);
  int N = 8;

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_u32, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(N));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_u32, p2, range, poly_arg_none());

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT32, idx1, poly_arg_none());
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT32, poly_arg_int(1));

  PolyUOp *idiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_UINT32, la, lb, poly_arg_none());
  PolyUOp *shr = poly_uop2(ctx, POLY_OP_SHR, POLY_UINT32, la, one, poly_arg_none());
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, idiv, mod, poly_arg_none());
  PolyUOp *out = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT32, sum, shr, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, out, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  bool found_div_u = false, found_rem_u = false, found_shr_u = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_I32_DIV_U) found_div_u = true;
    if (wasm[i] == WASM_OP_I32_REM_U) found_rem_u = true;
    if (wasm[i] == WASM_OP_I32_SHR_U) found_shr_u = true;
  }
  ASSERT_TRUE(found_div_u);
  ASSERT_TRUE(found_rem_u);
  ASSERT_TRUE(found_shr_u);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_simd_flag) {
  /* Render with SIMD enabled — verify SIMD prefix byte appears */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain SIMD prefix (0xFD) for f32x4 ops */
  bool found_simd = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX) {
      found_simd = true;
      break;
    }
  }
  ASSERT_TRUE(found_simd);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_simd_where_mask) {
  /* Covers the WASM renderer's native f32x4 compare/select lowering. */
  WasmVecKernel k = wasm_make_vec_where_f32(16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  bool found_f32x4_lt = false;
  bool found_bitselect = false;
  for (int i = 0; i < wasm_size - 1; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX && wasm[i + 1] == WASM_SIMD_F32X4_LT)
      found_f32x4_lt = true;
    if (wasm[i] == WASM_SIMD_PREFIX && wasm[i + 1] == WASM_SIMD_V128_BITSELECT)
      found_bitselect = true;
  }
  ASSERT_TRUE(found_f32x4_lt);
  ASSERT_TRUE(found_bitselect);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, rewritten_where_compare_mask_stays_packed) {
  WasmVecKernel k = wasm_make_vec_where_f32(16);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_LT) >= 1);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_BITSELECT) >= 1);
  ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, direct_f32x4_alu_ops_emit_simd) {
  struct {
    PolyOps op;
    int opcode;
    const char *name;
  } cases[] = {
      {POLY_OP_ADD, WASM_SIMD_F32X4_ADD, "add"},
      {POLY_OP_SUB, WASM_SIMD_F32X4_SUB, "sub"},
      {POLY_OP_MUL, WASM_SIMD_F32X4_MUL, "mul"},
      {POLY_OP_FDIV, WASM_SIMD_F32X4_DIV, "div"},
      {POLY_OP_MAX, WASM_SIMD_F32X4_MAX, "max"},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    WasmVecKernel k = wasm_make_direct_vec_binop(cases[i].op);
    int n_lin = 0;
    PolyUOp **lin = poly_toposort_alloc(k.ctx, k.sink, &n_lin);
    ASSERT_NOT_NULL(lin);

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
    ASSERT_NOT_NULL(wasm);
    ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, cases[i].opcode) >= 1);
    ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT), 0);
    ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_REPLACE), 0);

    char path[128];
    snprintf(path, sizeof(path), "/tmp/polygrad_test_f32x4_%s.wasm", cases[i].name);
    ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
    ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
  }

  PASS();
}

TEST(wasm, matmul_specialized_modules_validate_and_use_load32_splat) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t n = 64;
  int64_t shape2[2] = {n, n};
  PolyUOp *a = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *b = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, n * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab.wasm"), 0);
  free(wasm);

  int relaxed_size = 0;
  wasm = poly_render_wasm_matmul(body, &relaxed_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(relaxed_size > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_relaxed.wasm", wasm, relaxed_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_relaxed.wasm"), 0);
  free(wasm);

  poly_schedule_free(sched);

  PolyUOp *b0 = poly_reshape(ctx, poly_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *bt = poly_permute(ctx, b0, (int64_t[]){1, 0}, 2);
  PolyUOp *out_t = poly_buffer(ctx, POLY_FLOAT32, n * n);
  PolyUOp *sink_t = poly_sink1(ctx, poly_store_val(ctx, out_t, poly_dot(ctx, a, bt)));
  PolySchedule *sched_t = poly_complete_create_schedule_with_vars(ctx, sink_t, POLY_MODE_CALL);
  ASSERT_TRUE(sched_t != NULL);
  PolyUOp *body_t = poly_schedule_call_body(sched_t, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body_t));

  wasm = poly_render_wasm_matmul(body_t, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_abt.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_abt.wasm"), 0);
  free(wasm);

  wasm = poly_render_wasm_matmul(body_t, &relaxed_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(relaxed_size > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_abt_relaxed.wasm", wasm, relaxed_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_abt_relaxed.wasm"), 0);
  free(wasm);

  poly_schedule_free(sched_t);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 16, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_k_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_k_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("/tmp/polygrad_test_matmul_ab_k_tail.wasm", m, n, k), 0);
  free(wasm);
  poly_schedule_free(sched);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_all_scalar_epilogue) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 3, n = 5, k = 2;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_scalar_epilogue.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_scalar_epilogue.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("/tmp/polygrad_test_matmul_ab_scalar_epilogue.wasm", m, n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_row1_nontransposed_uses_generic_fallback) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 7, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_FALSE(poly_wasm_can_render_matmul(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  const char *path = "/tmp/polygrad_test_matmul_ab_row1_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab(path, m, n, k), 0);

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_m_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 6, n = 16, k = 8;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_m_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_m_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("/tmp/polygrad_test_matmul_ab_m_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_n_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 18, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_n_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_n_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("/tmp/polygrad_test_matmul_ab_n_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_combined_m_n_k_tails) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 6, n = 18, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *b = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2
  );
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_ab_all_tails.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_ab_all_tails.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("/tmp/polygrad_test_matmul_ab_all_tails.wasm", m, n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 16, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *bt0 = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2
  );
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_abt_k_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_abt_k_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_abt("/tmp/polygrad_test_matmul_abt_k_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_row1_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 16, k = 5;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *bt0 = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2
  );
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_abt_row1_k_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_abt_row1_k_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_abt_row1("/tmp/polygrad_test_matmul_abt_row1_k_tail.wasm", n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_specializes_single_row_token_projection) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 16, k = 8;
  PolyUOp *a = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2
  );
  PolyUOp *bt0 = poly_reshape(
      ctx, poly_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2
  );
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out = poly_buffer(ctx, POLY_FLOAT32, m * n);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_TRUE(sched != NULL);
  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_matmul_abt_row1.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("/tmp/polygrad_test_matmul_abt_row1.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_abt_row1("/tmp/polygrad_test_matmul_abt_row1.wasm", n, k), 0);
  free(wasm);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, reg_store_group_executes_without_packed_cache) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType reg_ptr = poly_dtype_ptr(POLY_FLOAT32, 4, POLY_ADDR_REG);

  PolyUOp *inp = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(1));
  PolyUOp *reg = poly_uop0(ctx, POLY_OP_DEFINE_REG, reg_ptr, poly_arg_int(0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));

  PolyUOp *stores[4];
  PolyUOp *out_stores[4];
  PolyUOp *idxs[4];
  for (int i = 0; i < 4; i++) {
    idxs[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *in_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, inp, idxs[i], poly_arg_none());
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
    PolyUOp *value = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, one, poly_arg_none());
    PolyUOp *reg_idx = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, reg, idxs[i], poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg_idx, value, poly_arg_none());
  }
  PolyUOp *store_group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, stores, 4, poly_arg_none());
  PolyUOp *after_src[2] = {reg, store_group};
  PolyUOp *after = poly_uop(ctx, POLY_OP_AFTER, reg_ptr, after_src, 2, poly_arg_none());
  for (int i = 0; i < 4; i++) {
    PolyUOp *reg_idx = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after, idxs[i], poly_arg_none());
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, reg_idx, poly_arg_none());
    PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, idxs[i], poly_arg_none());
    out_stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, load, poly_arg_none());
  }
  PolyUOp *out_group = poly_uop(ctx, POLY_OP_GROUP, POLY_VOID, out_stores, 4, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, out_group);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  const char *path = "/tmp/polygrad_test_packed_reg_store_group.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_reg_group_f32(path), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, wide_vector_gep_scalar_consumers_use_selected_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType f32x16 = poly_dtype_vec(POLY_FLOAT32, 16);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));

  PolyUOp *lanes[16];
  for (int i = 0; i < 16; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i + 0.25));
  PolyUOp *wide = poly_uop(ctx, POLY_OP_VECTORIZE, f32x16, lanes, 16, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));

  PolyUOp *stores[16];
  for (int i = 0; i < 16; i++) {
    PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *dst = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, idx, poly_arg_none());
    PolyUOp *lane = poly_uop1(ctx, POLY_OP_GEP, POLY_FLOAT32, wide, poly_arg_int(i));
    PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lane, one, poly_arg_none());
    PolyUOp *cond = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, zero, lane, poly_arg_none());
    PolyUOp *sel_src[3] = {cond, sum, zero};
    PolyUOp *sel = poly_uop(ctx, POLY_OP_WHERE, POLY_FLOAT32, sel_src, 3, poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, dst, sel, poly_arg_none());
  }
  PolyUOp *sink = poly_uop(ctx, POLY_OP_SINK, POLY_VOID, stores, 16, poly_arg_none());

  int n_lin = 0;
  PolyUOp **lin = poly_toposort_alloc(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "/tmp/polygrad_test_wide_vector_gep.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_wide_vector_gep_f32(path), 0);

  free(wasm);
  poly_toposort_free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, packed_group_reduce_uses_simd_alu) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *x =
      poly_reshape(ctx, poly_buffer_f32(ctx, 1024 * 1024), (int64_t[]){1024, 1024}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1024, 1}, 2);
  PolyUOp *col = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1, 1024}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){1024, 1024}, 2);
  PolyUOp *col_e = poly_expand(ctx, col, (int64_t[]){1024, 1024}, 2);
  PolyUOp *expr = poly_relu(
      ctx,
      poly_alu2(
          ctx, POLY_OP_SUB,
          poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
          poly_full(ctx, (int64_t[]){1024, 1024}, 2, 0.25)
      )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1024}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  int n_lin = 0;
  PolyUOp **lin = wasm_linearize_generic_test(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int n_range = 0;
  int n_reg_storage = 0;
  int n_vec_load = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op == POLY_OP_RANGE) n_range++;
    if (lin[i]->op == POLY_OP_DEFINE_REG ||
        (lin[i]->op == POLY_OP_BUFFER && lin[i]->dtype.is_ptr &&
         lin[i]->dtype.addrspace == POLY_ADDR_REG))
      n_reg_storage++;
    if (lin[i]->op == POLY_OP_LOAD && lin[i]->dtype.count == 4) n_vec_load++;
  }
  /* Matches current tinygrad CPU-style reduce lowering: output-axis UPCAST plus
   * reduce-axis UNROLL becomes a small register tile, vector loads, scalar
   * lane arithmetic, horizontal accumulation, and a packed output store. */
  ASSERT_INT_EQ(n_range, 2);
  ASSERT_TRUE(n_reg_storage >= 1);
  ASSERT_TRUE(n_vec_load >= 5);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  int n_add = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD);
  int n_mul = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_MUL);
  int n_select = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_BITSELECT);
  int n_shuffle = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE);
  int n_extract = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT);
  int n_replace = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_REPLACE);
  ASSERT_TRUE(n_add + n_mul + n_select > 0);
  ASSERT_TRUE(n_shuffle + n_extract + n_replace > 0);

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, packed_group_reduce_relu_executes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t n = 1024;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, n * n), (int64_t[]){n, n}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n, 1}, 2);
  PolyUOp *col = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){1, n}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){n, n}, 2);
  PolyUOp *col_e = poly_expand(ctx, col, (int64_t[]){n, n}, 2);
  PolyUOp *expr = poly_relu(
      ctx,
      poly_alu2(
          ctx, POLY_OP_SUB,
          poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
          poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
      )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  int n_lin = 0;
  PolyUOp **lin = wasm_linearize_generic_test(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "/tmp/polygrad_test_broadcast_reduce_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, specialized_row_reduce_relu_executes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t n = 1024;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, n * n), (int64_t[]){n, n}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n, 1}, 2);
  PolyUOp *col = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){1, n}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){n, n}, 2);
  PolyUOp *col_e = poly_expand(ctx, col, (int64_t[]){n, n}, 2);
  PolyUOp *expr = poly_relu(
      ctx,
      poly_alu2(
          ctx, POLY_OP_SUB,
          poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
          poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
      )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_reduce(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_reduce(body, &wasm_size);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD) >= 2);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_MUL) >= 1);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD) >= 2);

  const char *path = "/tmp/polygrad_test_specialized_row_reduce_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, specialized_row_reduce_relu_tail_executes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t n = 1027;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, n * n), (int64_t[]){n, n}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n, 1}, 2);
  PolyUOp *col = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){1, n}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){n, n}, 2);
  PolyUOp *col_e = poly_expand(ctx, col, (int64_t[]){n, n}, 2);
  PolyUOp *expr = poly_relu(
      ctx,
      poly_alu2(
          ctx, POLY_OP_SUB,
          poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
          poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
      )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_TRUE(poly_wasm_can_render_reduce(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_reduce(body, &wasm_size);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD) >= 2);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD) >= 2);

  const char *path = "/tmp/polygrad_test_specialized_row_reduce_relu_tail.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, row_reduce_f64_uses_generic_renderer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t n = 17;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f64(ctx, n * n), (int64_t[]){n, n}, 2);
  PolyUOp *sum = poly_sum_reduce(ctx, x, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f64(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_FALSE(poly_wasm_can_render_reduce(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "/tmp/polygrad_test_f64_row_reduce_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, row_reduce_noncompare_where_uses_generic_renderer) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t n = 16;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, n * n), (int64_t[]){n, n}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n, 1}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){n, n}, 2);
  PolyUOp *mask = poly_alu2(ctx, POLY_OP_ADD, x, row_e);
  PolyUOp *zero = poly_full(ctx, (int64_t[]){n, n}, 2, 0.0);
  PolyUOp *selected = poly_alu3(ctx, POLY_OP_WHERE, mask, x, zero);
  PolyUOp *sum = poly_sum_reduce(ctx, selected, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  PolyUOp *body = poly_schedule_call_body(sched, 0);
  ASSERT_FALSE(poly_wasm_can_render_reduce(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "/tmp/polygrad_test_noncompare_where_reduce_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_bias_relu_executes_after_packed_reduce) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  const int64_t tokens = 32, d = 128, hidden = 256;
  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, tokens * d), (int64_t[]){tokens, d}, 2);
  PolyUOp *w0 = poly_reshape(ctx, poly_buffer_f32(ctx, hidden * d), (int64_t[]){hidden, d}, 2);
  PolyUOp *w = poly_permute(ctx, w0, (int64_t[]){1, 0}, 2);
  PolyUOp *bias = poly_reshape(ctx, poly_buffer_f32(ctx, hidden), (int64_t[]){1, hidden}, 2);
  PolyUOp *bias_e = poly_expand(ctx, bias, (int64_t[]){tokens, hidden}, 2);
  PolyUOp *h = poly_relu(ctx, poly_alu2(ctx, POLY_OP_ADD, poly_dot(ctx, x, w), bias_e));
  PolyUOp *out =
      poly_reshape(ctx, poly_buffer_f32(ctx, tokens * hidden), (int64_t[]){tokens, hidden}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, h));
  PolySchedule *sched = poly_complete_create_schedule_with_vars(ctx, sink, POLY_MODE_CALL);
  ASSERT_NOT_NULL(sched);
  ASSERT_TRUE(sched->template->n_calls > 0);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, poly_schedule_call_body(sched, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "/tmp/polygrad_test_matmul_bias_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(
      node_run_wasm_matmul_bias_relu(path, (int)tokens, (int)d, (int)hidden), 0
  );

  free(wasm);
  free(lin);
  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_rand_threefry_dag_terminates) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[1] = {100};
  PolyUOp *rand = poly_rand(ctx, shape, 1, 42);
  PolyUOp *out = poly_buffer_f32(ctx, 100);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, rand));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm_env(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  ASSERT_TRUE(n_lin > 0);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, write_and_validate) {
  /* Render a vecadd kernel and write to /tmp for manual validation */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write to tmp file for external validation */
  FILE *f = fopen("/tmp/polygrad_test_vecadd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  /* Try running wasm-validate if available */
  int rc =
      system("which wasm-validate > /dev/null 2>&1 && wasm-validate /tmp/polygrad_test_vecadd.wasm"
      );
  if (rc == 0) {
    /* wasm-validate passed — great! */
  }
  /* Don't fail if wasm-validate is not installed */

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, sparse_cross_entropy_i64_gather_index_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* Sparse cross-entropy follows tinygrad's late index dtype lowering: loaded
   * class labels can remain int64 in gather address expressions. The WASM
   * backend still targets wasm32 memory, so the renderer must wrap those
   * address indexes to i32 at the memory access boundary. */
  PolyUOp *logits_buf = poly_buffer_f32(ctx, 6);
  PolyUOp *target_buf = poly_buffer(ctx, POLY_INT32, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolySchedule *sched = poly_schedule_with_vars(ctx, targets, 1, realized);
  ASSERT_NOT_NULL(sched);
  ASSERT_INT_EQ(sched->template->n_calls, 3);

  bool saw_i64_index = false;
  bool saw_i32_wrap = false;

  for (int item = 0; item < sched->template->n_calls; item++) {
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_wasm_env(ctx, poly_schedule_call_body(sched, item), &n_lin);
    ASSERT_NOT_NULL(lin);

    bool item_has_i64_index = false;
    for (int i = 0; i < n_lin; i++) {
      PolyUOp *u = lin[i];
      if (u->op == POLY_OP_INDEX && u->n_src >= 2 && u->src[1] &&
          !poly_dtype_is_float(u->src[1]->dtype) && u->src[1]->dtype.bitsize == 64) {
        item_has_i64_index = true;
        saw_i64_index = true;
      }
    }

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
    ASSERT_NOT_NULL(wasm);

    if (item_has_i64_index) {
      for (int i = 0; i < wasm_size; i++) {
        if (wasm[i] == WASM_OP_I32_WRAP_I64) {
          saw_i32_wrap = true;
          break;
        }
      }
    }

    char path[128];
    snprintf(path, sizeof(path), "/tmp/polygrad_test_ce_sparse_item%d.wasm", item);
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    fwrite(wasm, 1, (size_t)wasm_size, f);
    fclose(f);

    const char *node = poly_test_node_cmd_for_wasm(path);
    if (node) {
      char cmd[512];
      snprintf(
          cmd, sizeof(cmd),
          "%s -e \"const fs=require('fs'); new WebAssembly.Module(fs.readFileSync('%s'))\"",
          node, path
      );
      ASSERT_INT_EQ(system(cmd), 0);
    }

    free(wasm);
    free(lin);
  }

  if (saw_i64_index) ASSERT_TRUE(saw_i32_wrap);

  poly_schedule_free(sched);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mixed_width_compare_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad's UOp spec wants comparison operands to share a base dtype, but
   * imported/index-heavy graphs can expose a late widened label compared with
   * an int bound. WASM has no implicit casts, so the renderer must coerce the
   * narrower operand before emitting i64.lt_s/i64.eq/etc. */
  PolyDType ptr_i32 = poly_dtype_ptr(POLY_INT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i32, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i32, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(9));
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, lt, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int32_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i32(lin, n_lin, "mixed_width_compare_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_mixed_width_compare.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i32("/tmp/polygrad_test_mixed_width_compare.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 1);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mixed_width_shift_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* The first Qwen renderer failure after comparison coercion was i64.shl with
   * an i32 left operand. For WASM, both operands consumed by i64.shl must be
   * i64 stack values even when Polygrad created the shift count as int32. */
  PolyDType ptr_i64 = poly_dtype_ptr(POLY_INT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_i64, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_i64, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT64, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, shl, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(lin, n_lin, "mixed_width_shift_c", &c_out), 0);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_mixed_width_shift.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_i64("/tmp/polygrad_test_mixed_width_shift.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 28);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, unsigned_i64_div_mod_matches_c_renderer) {
  PolyCtx *ctx = poly_ctx_new();

  /* Tinygrad renderers distinguish signed/unsigned integer lowering:
   * LLVM emits sdiv/udiv and srem/urem, and NIR emits idiv/udiv and irem/umod.
   * WASM has the same split. This case catches the silent high-bit bug where a
   * uint64 IDIV used i64.div_s and returned -1 instead of 2. */
  PolyDType ptr_u64 = poly_dtype_ptr(POLY_UINT64, -1, POLY_ADDR_GLOBAL);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_u64, poly_arg_int(0));
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_u64, out, zero, poly_arg_none());
  PolyUOp *lhs =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)0x8000000000000005ULL));
  PolyUOp *rhs =
      poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_arg_int((int64_t)0x4000000000000000ULL));
  PolyUOp *idiv = poly_uop2(ctx, POLY_OP_IDIV, POLY_UINT64, lhs, rhs, poly_arg_none());
  PolyUOp *mod = poly_uop2(ctx, POLY_OP_MOD, POLY_UINT64, lhs, rhs, poly_arg_none());
  PolyUOp *outv = poly_uop2(ctx, POLY_OP_ADD, POLY_UINT64, idiv, mod, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, outv, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_rewritten(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  bool found_div_u = false, found_rem_u = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_I64_DIV_U) found_div_u = true;
    if (wasm[i] == WASM_OP_I64_REM_U) found_rem_u = true;
  }
  ASSERT_TRUE(found_div_u);
  ASSERT_TRUE(found_rem_u);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(lin, n_lin, "unsigned_i64_div_mod_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_unsigned_i64_div_mod.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i64("/tmp/polygrad_test_unsigned_i64_div_mod.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 7);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mulacc_operand_order_matches_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad's Ops.MULACC is (x * y) + z. WASM is stack based, so a renderer
   * that pushes x,y,z and then emits MUL,ADD silently computes x + (y * z).
   * Keep this C-vs-WASM test on loaded values to prevent constant folding from
   * hiding the operand-order bug. */
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyUOp *buf = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *i2 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i0, poly_arg_none());
  PolyUOp *x_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i1, poly_arg_none());
  PolyUOp *y_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i2, poly_arg_none());
  PolyUOp *z_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, buf, i3, poly_arg_none());
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, x_idx, poly_arg_none());
  PolyUOp *y = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, y_idx, poly_arg_none());
  PolyUOp *z = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, z_idx, poly_arg_none());
  PolyUOp *mulacc_src[3] = {x, y, z};
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_src, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, mulacc, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 2.0f, 3.0f, 5.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(lin, n_lin, "mulacc_operand_order_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 11.0f, 1e-6);
  ASSERT_INT_EQ(
      wasm_write_module("/tmp/polygrad_test_mulacc_operand_order.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(
      node_run_wasm_f32_buffer("/tmp/polygrad_test_mulacc_operand_order.wasm", c_buf[0]), 0
  );

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, define_reg_array_constant_indexes_match_c_renderer) {
  PolyCtx *ctx = poly_ctx_new();

  /* Qwen's WASM-linearized kernels contain DEFINE_REG(ptr_size=3/4) with
   * constant INDEX(reg, i) accesses. Native C/WGSL render these as local arrays;
   * WASM must not collapse every element to the same scalar local. */
  PolyDType ptr_f32 = poly_dtype_ptr(POLY_FLOAT32, -1, POLY_ADDR_GLOBAL);
  PolyDType reg_ptr = poly_dtype_ptr(POLY_FLOAT32, 4, POLY_ADDR_REG);
  PolyUOp *out = poly_uop0(ctx, POLY_OP_PARAM, ptr_f32, poly_arg_int(0));
  PolyUOp *reg = poly_uop0(ctx, POLY_OP_DEFINE_REG, reg_ptr, poly_arg_int(0));
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));

  PolyUOp *reg0 = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, reg, i0, poly_arg_none());
  PolyUOp *store0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg0, two, poly_arg_none());
  PolyUOp *after0_src[2] = {reg, store0};
  PolyUOp *after0 = poly_uop(ctx, POLY_OP_AFTER, reg_ptr, after0_src, 2, poly_arg_none());

  PolyUOp *reg3 = poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after0, i3, poly_arg_none());
  PolyUOp *store3 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg3, five, poly_arg_none());
  PolyUOp *after1_src[2] = {after0, store3};
  PolyUOp *after1 = poly_uop(ctx, POLY_OP_AFTER, reg_ptr, after1_src, 2, poly_arg_none());

  PolyUOp *load0 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after1, i0, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *load3 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, reg_ptr, after1, i3, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load0, load3, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, ptr_f32, out, i0, poly_arg_none());
  PolyUOp *store_out = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sum, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store_out);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(lin, n_lin, "define_reg_array_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 7.0f, 1e-6);
  ASSERT_INT_EQ(wasm_write_module("/tmp/polygrad_test_define_reg_array.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_f32_buffer("/tmp/polygrad_test_define_reg_array.wasm", c_buf[0]), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_pow) {
  /* POW kernel: c[i] = a[i] ^ b[i] — lowered through tinygrad-style
   * transcendental decomposition, so a math import call is expected. */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Must start with WASM magic */
  ASSERT_INT_EQ(wasm[0], 0x00);
  ASSERT_INT_EQ(wasm[1], 0x61);
  ASSERT_INT_EQ(wasm[2], 0x73);
  ASSERT_INT_EQ(wasm[3], 0x6D);

  /* Must contain a CALL instruction (0x10) for the math import. */
  bool found_call = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_CALL) {
      found_call = true;
      break;
    }
  }
  ASSERT_TRUE(found_call);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, render_pow_simd_fallback) {
  /* POW has no WASM SIMD opcode. The renderer may still use vector loads/stores,
   * but the exponentiation itself must lower through the scalar math import. */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Should contain a CALL for powf. */
  bool found_call = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_CALL) {
      found_call = true;
      break;
    }
  }
  ASSERT_TRUE(found_call);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_pow) {
  /* End-to-end: POW kernel via Node.js — c[i] = a[i]^b[i] */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_e2e_pow.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  const char *node = poly_test_node_cmd_for_wasm("/tmp/polygrad_e2e_pow.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS();
  }

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js /tmp/polygrad_e2e_pow.wasm pow 4", node);
  int rc = system(cmd);
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_vecadd) {
  /* End-to-end: render WASM, write to file, run with Node.js */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write WASM to tmp file */
  FILE *f = fopen("/tmp/polygrad_e2e_vecadd.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  /* Skip if node is not available */
  const char *node = poly_test_node_cmd_for_wasm("/tmp/polygrad_e2e_vecadd.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  /* Run the Node.js test runner */
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js /tmp/polygrad_e2e_vecadd.wasm add 8", node);
  int rc = system(cmd);
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_vecadd_simd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  const char *path = "/tmp/polygrad_e2e_vecadd_simd.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);

  const char *node = poly_test_node_cmd_for_wasm(path);
  if (node) {
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js /tmp/polygrad_e2e_vecadd_simd.wasm add 10", node);
    ASSERT_INT_EQ(system(cmd), 0);
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_where_simd) {
  WasmVecKernel k = wasm_make_vec_where_f32(10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  const char *path = "/tmp/polygrad_e2e_where_simd.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_where_f32(path, 10), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

/* F64 WASM helpers */

static WasmVecKernel wasm_make_vec_binop_f64(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f64 = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(1));
  PolyUOp *p2 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(2));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p1, range, poly_arg_none());
  PolyUOp *idx2 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p2, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT64, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

static WasmVecKernel wasm_make_vec_unary_f64(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyDType ptr_f64 = poly_dtype_ptr(POLY_FLOAT64, -1, POLY_ADDR_GLOBAL);

  PolyUOp *p0 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(0));
  PolyUOp *p1 = poly_uop0(ctx, POLY_OP_PARAM, ptr_f64, poly_arg_int(1));

  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(n));
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));

  PolyUOp *idx0 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p0, range, poly_arg_none());
  PolyUOp *idx1 = poly_uop2(ctx, POLY_OP_INDEX, ptr_f64, p1, range, poly_arg_none());

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT64, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, end, poly_arg_none());

  return (WasmVecKernel){ctx, sink, n};
}

/* F64 WASM renderer tests */

TEST(wasm_f64, render_vecadd_f64_scalar) {
  /* Render f64 vecadd kernel in scalar mode -- verify f64 opcodes */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  /* Must contain f64.add opcode (0xA0) */
  bool found_f64_add = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_ADD) {
      found_f64_add = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_add);

  /* Must contain f64.load opcode (0x2B) */
  bool found_f64_load = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_LOAD) {
      found_f64_load = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_load);

  /* Must contain f64.store opcode (0x39) */
  bool found_f64_store = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_STORE) {
      found_f64_store = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_store);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, render_neg_f64_scalar) {
  /* Render f64 unary neg kernel -- verify f64.neg opcode */
  WasmVecKernel k = wasm_make_vec_unary_f64(POLY_OP_NEG, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);

  ASSERT_NOT_NULL(wasm);

  /* Must contain f64.neg opcode (0x9A) */
  bool found_f64_neg = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_F64_NEG) {
      found_f64_neg = true;
      break;
    }
  }
  ASSERT_TRUE(found_f64_neg);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, render_f64_stays_scalar_under_wasm_caps) {
  /* Current WASM caps intentionally keep f64 graphs scalar. */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);

  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);

  bool found_f64x2_add = false;
  for (int i = 0; i < wasm_size - 1; i++) {
    if (wasm[i] == WASM_SIMD_PREFIX && wasm[i + 1] == WASM_SIMD_F64X2_ADD) {
      found_f64x2_add = true;
      break;
    }
  }
  ASSERT_FALSE(found_f64x2_add);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, validate_f64_scalar) {
  /* Write f64 scalar kernel and validate with wasm-validate */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_test_f64_scalar.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate /tmp/polygrad_test_f64_scalar.wasm");
  if (rc == 0) { /* wasm-validate passed */
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, validate_f64_simd) {
  /* Write f64x2 SIMD kernel and validate with wasm-validate */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 16);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_test_f64_simd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate --enable-simd /tmp/polygrad_test_f64_simd.wasm");
  if (rc == 0) { /* wasm-validate passed */
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, e2e_node_vecadd_f64) {
  /* End-to-end: render f64 WASM, write to file, run with Node.js */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 8);
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("/tmp/polygrad_e2e_vecadd_f64.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  const char *node = poly_test_node_cmd_for_wasm("/tmp/polygrad_e2e_vecadd_f64.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js /tmp/polygrad_e2e_vecadd_f64.wasm add_f64 8", node);
  int rc = system(cmd);
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}
