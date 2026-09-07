/*
 * test_wasm.c — Tests for WASM binary builder and WASM renderer
 */

#include "test_harness.h"
#include "../src/bigint.h"
#include "../src/codegen/codegen.h"
#include "../src/codegen/decomp/dtype.h"
#include "../src/engine/realize.h"
#include "../src/engine/schedule.h"
#include "../src/frontend.h"
#include "../src/tensor.h"
#include "../src/uop/spec.h"
#include "../src/wasm_builder.h"

static int wasm_run_c_i32(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    const char *fn_name,
    int32_t *out
) {
  char *src = poly_render_c(ctx, lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_i64(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    const char *fn_name,
    int64_t *out
) {
  char *src = poly_render_c(ctx, lin, n_lin, fn_name);
  if (!src) return -1;
  PolyProgram *prog = poly_compile_c(src, fn_name);
  free(src);
  if (!prog) return -1;
  void *args[1] = {out};
  poly_program_call(prog, args, 1);
  poly_program_destroy(prog);
  return 0;
}

static int wasm_run_c_f32_buffer(
    PolyCtx *ctx,
    PolyUOp **lin,
    int n_lin,
    const char *fn_name,
    float *buf
) {
  char *src = poly_render_c(ctx, lin, n_lin, fn_name);
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

static int wasm_count_uops(PolyUOp **uops, int n, PolyOps op) {
  int count = 0;
  for (int i = 0; i < n; i++)
    if (uops[i]->op == op) count++;
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
      "const "
      "b=Buffer.from('AGFzbQEAAAABBAFgAAACDwEDZW52Bm1lbW9yeQIAAQMCAQAHCgEGa2VybmVsAAAKQwFBAEEA/"
      "QwAAABAAAAAQAAAAEAAAABA/QwAAEBAAABAQAAAQEAAAEBA/QwAAKBAAACgQAAAoEAAAKBA/YUC/"
      "QsEAAs=','base64');"
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

      .caps =
          {
              .has_mulacc = true,
              .has_threefry = false,
              .has_int64 = true,
              .has_local = false,
              .has_simd_int = false,
              .has_simd_float = true,
              .max_vec_width = 4,
          },
      .device = POLY_DEVICE_WASM,
      .opt_policy = POLY_OPT_HEURISTIC,
  };
  return poly_test_full_rewrite_and_linearize_ex(ctx, sink, opts, n_out);
}

static int node_compile_wasm_module(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');new WebAssembly.Module(fs.readFileSync('%s'))\"", node, path
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

static int node_run_wasm_bool_neg_identity(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[2048];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math:{}});"
      "const bytes=new Uint8Array(mem.buffer);"
      "bytes[4]=0;inst.exports.kernel(0,4);"
      "if(bytes[0]!==0){console.error('NEG(false) got '+bytes[0]);process.exit(2)}"
      "bytes[4]=1;inst.exports.kernel(0,4);"
      "if(bytes[0]!==1){console.error('NEG(true) got '+bytes[0]);process.exit(3)}\"",
      node, path
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

static int node_run_wasm_sparse_f32_params(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math:{}});"
      "const f32=new Float32Array(mem.buffer);"
      "f32.set([1,3],16);f32.set([0,2],20);"
      "inst.exports.kernel(0,64,80);"
      "const got=Array.from(f32.slice(0,4));const exp=[1,3,0,2];"
      "if(got.some((x,i)=>x!==exp[i])){console.error('got '+got+' expected "
      "'+exp);process.exit(2)}\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_bf16_vector_load(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math:{}});"
      "const input=256,output=512,expected=[2,4,6,8];"
      "new Uint16Array(mem.buffer,input,4).set([0x4000,0x4080,0x40c0,0x4100]);"
      "inst.exports.kernel(output,input);"
      "const got=Array.from(new Float32Array(mem.buffer,output,4));"
      "if(got.some((x,i)=>x!==expected[i])){"
      "console.error('got '+got+' expected '+expected);process.exit(2)}\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_bf16_vector_store(const char *path) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "const mod=new WebAssembly.Module(fs.readFileSync('%s'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem},math:{}});"
      "const input=256,output=512,expected=[0x4000,0x4080,0x40c0,0x4100];"
      "new Float32Array(mem.buffer,input,4).set([1,2,3,4]);"
      "inst.exports.kernel(input,output);"
      "const got=Array.from(new Uint16Array(mem.buffer,output,4));"
      "if(got.some((x,i)=>x!==expected[i])){"
      "console.error('got '+got+' expected '+expected);process.exit(2)}\"",
      node, path
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
      " if(Math.abs(got-exp)>1e-6){console.error('i='+i+' got '+got+' expected "
      "'+exp);process.exit(2);}"
      "}\"",
      node, n, path
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
      " if(Math.abs(got-exp)>1e-6){console.error('i='+i+' got '+got+' expected "
      "'+exp);process.exit(2);}"
      "}\"",
      node, path
  );
  return system(cmd);
}

static int node_run_wasm_broadcast_reduce_relu(const char *path, int n) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;

  const char *js_path = "temp/polygrad_test_broadcast_reduce_relu.js";
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

static int node_run_wasm_matmul_bias_relu(const char *path, int tokens, int d, int hidden) {
  const char *node = poly_test_node_cmd_for_wasm(path);
  if (!node) return 0;

  const char *js_path = "temp/polygrad_test_matmul_bias_relu.js";
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
      "    console.error('cell '+r+','+h+' got '+got+' expected '+exp+' diff "
      "'+Math.abs(got-exp));\n"
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
  const char *js_path = "temp/polygrad_test_matmul_abt_row1.js";
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
  const char *js_path = "temp/polygrad_test_matmul_ab.js";
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
      "    console.error('row '+r+' col '+c+' got '+got+' expected '+exp+' diff "
      "'+Math.abs(got-exp));\n"
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
  const char *js_path = "temp/polygrad_test_matmul_abt.js";
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
      "    console.error('row '+r+' col '+c+' got '+got+' expected '+exp+' diff "
      "'+Math.abs(got-exp));\n"
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

static int node_run_wasm_u64(const char *path, const char *expected) {
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
      "const got=new DataView(mem.buffer).getBigUint64(0,true);"
      "const expected=BigInt('%s');"
      "if(got!==expected){console.error('got '+got+' expected '+expected);process.exit(2)}\"",
      node, path, expected
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
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, n, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, n, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT32, n, 2);

  PolyUOp *bound = poly_const_int(ctx, n);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_vec_binop");

  return (WasmVecKernel){ctx, sink, n};
}

/* Current Tinygrad codegen/late/coalesce.py:151-164 represents a coalesced
 * f32x4 access as SHRINK(PARAM, offset, 4); lane width comes from UOp shape. */
static WasmVecKernel wasm_make_direct_vec_binop(PolyOps alu_op) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *params[3] = {
      poly_test_program_param(ctx, POLY_FLOAT32, 4, 0),
      poly_test_program_param(ctx, POLY_FLOAT32, 4, 1),
      poly_test_program_param(ctx, POLY_FLOAT32, 4, 2),
  };
  PolyUOp *zero = poly_const_int(ctx, 0), *four = poly_const_int(ctx, 4);
  PolyUOp *starts[] = {zero}, *sizes[] = {four};
  PolyUOp *addresses[3];
  for (int i = 0; i < 3; i++)
    addresses[i] = poly_shrink_uop(ctx, params[i], starts, sizes, 1);

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, addresses[0], poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, addresses[1], poly_arg_none());
  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT32, load0, load1, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, addresses[2], alu, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_f32x4");

  return (WasmVecKernel){ctx, sink, 4};
}

/* Helper: build b[i] = OP(a[i]) unary kernel */
static WasmVecKernel wasm_make_vec_unary(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, n, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, n, 1);

  PolyUOp *bound = poly_const_int(ctx, n);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT32, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_vec_unary");

  return (WasmVecKernel){ctx, sink, n};
}

/* Helper: c[i] = where(a[i] < 2.5, a[i] + 1, b[i] - 1) */
static WasmVecKernel wasm_make_vec_where_f32(int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, n, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, n, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT32, n, 2);

  PolyUOp *bound = poly_const_int(ctx, n);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);

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
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_vec_where");

  return (WasmVecKernel){ctx, sink, n};
}

/* WASM renderer tests */

TEST(wasm, rewrite_legalizes_non_native_f16_storage_like_python_renderer) {
  /* Pinned PythonRenderer excludes half on Python 3.11 and
   * do_dtype_decomps applies pm_float_decomp before rendering
   * (runtime/ops_python.py:203-223; codegen/__init__.py:116-140). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 2, 0);
  PolyUOp *in = poly_test_program_param(ctx, POLY_FLOAT16, 2, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *in_idx = poly_uop_index(ctx, in, &zero, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, in_idx, poly_arg_none());
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_f16_storage");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  int u16_loads = 0, f32_bitcasts = 0, residual_f16 = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    if (topo[i]->op == POLY_OP_LOAD && poly_dtype_eq(topo[i]->dtype, POLY_UINT16)) u16_loads++;
    if (topo[i]->op == POLY_OP_BITCAST && poly_dtype_eq(topo[i]->dtype, POLY_FLOAT32))
      f32_bitcasts++;
    if (scalar.priority == POLY_FLOAT16.priority && scalar.bitsize == POLY_FLOAT16.bitsize)
      residual_f16++;
  }
  ASSERT_INT_EQ(u16_loads, 1);
  ASSERT_INT_EQ(f32_bitcasts, 1);
  ASSERT_INT_EQ(residual_f16, 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_vecadd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);

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

TEST(wasm, narrow_integer_casts_execute) {
  /* PythonProgram CAST truncates to dtype width, and CStyle emits a typed
   * cast. Wasm i32 locals must not erase int8/int16 conversion semantics. */
  const PolyDType destinations[] = {POLY_INT8, POLY_UINT8, POLY_INT16, POLY_UINT16};
  const int32_t values[] = {128, -240, 65535, -65537};
  const int32_t expected[] = {-128, 16, -1, 65535};
  const int32_t alu_values[] = {127, 0, 32767, 0};
  const int32_t alu_expected[] = {-128, 255, -32768, 65535};
  bool correct = true;
  for (int literal = 0; literal < 4; literal++) {
    for (int i = 0; i < 4; i++) {
      PolyCtx *ctx = poly_ctx_new();
      PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
      PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
      PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, shape, poly_arg_param(&arg));
      PolyUOp *index = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
      PolyUOp *address = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, index, poly_arg_none());
      PolyUOp *value = poly_uop0(
          ctx, POLY_OP_CONST,
          literal == 3 ? POLY_FLOAT32
          : literal    ? POLY_WEAKINT
                       : POLY_INT32,
          literal == 3 ? poly_arg_float(values[i]) : poly_arg_int(values[i])
      );
      PolyUOp *narrow = poly_uop1(ctx, POLY_OP_CAST, destinations[i], value, poly_arg_none());
      if (literal == 2) {
        PolyUOp *a = poly_uop0(ctx, POLY_OP_CONST, destinations[i], poly_arg_int(alu_values[i]));
        PolyUOp *b = poly_uop0(ctx, POLY_OP_CONST, destinations[i], poly_arg_int(1));
        narrow = poly_uop2(
            ctx, i % 2 ? POLY_OP_SUB : POLY_OP_ADD, destinations[i], a, b, poly_arg_none()
        );
      }
      PolyUOp *wide = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, narrow, poly_arg_none());
      PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, wide, poly_arg_none());
      PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
      int n = 0, size = 0;
      PolyUOp **uops = poly_toposort(ctx, sink, &n);
      uint8_t *wasm = poly_render_wasm(ctx, uops, n, &size, false);
      const char *path = "temp/polygrad_test_narrow_integer_cast.wasm";
      int rc = wasm ? wasm_write_module(path, wasm, size) : -1;
      if (rc == 0) rc = node_run_wasm_i32(path, literal == 2 ? alu_expected[i] : expected[i]);
      correct &= rc == 0;
      free(wasm);
      poly_ctx_destroy(ctx);
    }
  }
  ASSERT_TRUE(correct);
  PASS();
}

TEST(wasm, current_casted_literal_executes_without_extra_conversion) {
  /* Current tinygrad pm_casted_consts leaves CAST(int, CONST(weakint)) for
   * the renderer. Wasm aliases the identical i32 value class. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *shape = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(1));
  PolyParamArg arg = {.slot = 0, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_INT32, shape, poly_arg_param(&arg));
  PolyUOp *weak_zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_zero, poly_arg_none());
  PolyUOp *address = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, index, poly_arg_none());
  PolyUOp *weak_seven = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(7));
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_INT32, weak_seven, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, address, value, poly_arg_none());
  PolyUOp *sink = poly_uop1(ctx, POLY_OP_SINK, POLY_VOID, store, poly_arg_none());
  int n = 0;
  PolyUOp **uops = poly_toposort(ctx, sink, &n);
  ASSERT_NOT_NULL(uops);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, uops, n, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_current_casted_literal.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_i32(path, 7), 0);
  free(wasm);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_vecmul) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_MUL, 8);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);

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

TEST(wasm, raw_bool_neg_is_typed_identity_not_logical_not) {
  /* Pinned raw NEG(bool) is arithmetic NEG followed by bool truncation
   * (uop/ops.py:1182-1197), so False/True remain False/True. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_BOOL, 1, 0);
  PolyUOp *in = poly_test_program_param(ctx, POLY_BOOL, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *in_idx = poly_uop_index(ctx, in, &zero, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BOOL, in_idx, poly_arg_none());
  PolyUOp *neg = poly_uop1(ctx, POLY_OP_NEG, POLY_BOOL, load, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, neg, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_bool_neg");

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_raw_bool_neg.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_bool_neg_identity(path), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, bf16_same_width_bitcasts_match_tinygrad) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyFloatDecompContext fctx = {.from = POLY_BFLOAT16, .to = POLY_FLOAT32};
  PolyDType scalar_types[] = {POLY_UINT16, POLY_INT16, POLY_FLOAT16};

  for (int i = 0; i < 3; i++) {
    PolyDType dst = scalar_types[i];
    PolyUOp *bf16 = poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(1.5));
    PolyUOp *from = poly_uop1(ctx, POLY_OP_BITCAST, dst, bf16, poly_arg_none());
    PolyUOp *from_rewritten =
        poly_graph_rewrite_ctx_ex(ctx, from, poly_pm_float_decomp(), &fctx, true);
    ASSERT_NOT_NULL(from_rewritten);
    ASSERT_INT_EQ(from_rewritten->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(from_rewritten->dtype, dst));
    ASSERT_INT_EQ(from_rewritten->n_src, 1);
    ASSERT_TRUE(poly_dtype_eq(from_rewritten->src[0]->dtype, POLY_UINT16));

    PolyArg src_arg = poly_dtype_is_float(dst) ? poly_arg_float(1.5) : poly_arg_int(0x3fc0);
    PolyUOp *raw = poly_uop0(ctx, POLY_OP_CONST, dst, src_arg);
    PolyUOp *to = poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, raw, poly_arg_none());
    PolyUOp *to_rewritten = poly_graph_rewrite_ctx_ex(ctx, to, poly_pm_float_decomp(), &fctx, true);
    ASSERT_NOT_NULL(to_rewritten);
    ASSERT_INT_EQ(to_rewritten->op, POLY_OP_BITCAST);
    ASSERT_TRUE(poly_dtype_eq(to_rewritten->dtype, POLY_FLOAT32));

    PolyUOp *roots[] = {from_rewritten, to_rewritten};
    for (int r = 0; r < 2; r++) {
      int n_topo = 0;
      PolyUOp **topo = poly_toposort(ctx, roots[r], &n_topo);
      ASSERT_NOT_NULL(topo);
      for (int j = 0; j < n_topo; j++) {
        PolyDType scalar = topo[j]->dtype;
        ASSERT_FALSE(
            scalar.priority == POLY_BFLOAT16.priority && scalar.bitsize == POLY_BFLOAT16.bitsize
        );
      }
    }
  }

  PolyUOp *param = poly_test_program_param(ctx, POLY_BFLOAT16, 1, 0);
  PolyUOp *offset = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *index = poly_uop_index(ctx, param, &offset, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, index, poly_arg_none());
  PolyUOp *bitcasted_load = poly_graph_rewrite_ctx_ex(
      ctx, poly_uop1(ctx, POLY_OP_BITCAST, POLY_INT16, load, poly_arg_none()),
      poly_pm_float_decomp(), &fctx, true
  );
  ASSERT_NOT_NULL(bitcasted_load);
  ASSERT_INT_EQ(bitcasted_load->op, POLY_OP_BITCAST);
  ASSERT_TRUE(poly_dtype_eq(bitcasted_load->dtype, POLY_INT16));
  ASSERT_INT_EQ(bitcasted_load->src[0]->op, POLY_OP_LOAD);
  ASSERT_TRUE(poly_dtype_eq(bitcasted_load->src[0]->dtype, POLY_UINT16));
  ASSERT_TRUE(poly_dtype_eq(bitcasted_load->src[0]->src[0]->dtype, POLY_UINT16));

  PolyUOp *raw_i16 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT16, poly_arg_int(1));
  PolyUOp *bitcasted_value =
      poly_uop1(ctx, POLY_OP_BITCAST, POLY_BFLOAT16, raw_i16, poly_arg_none());
  PolyUOp *numeric_store = poly_graph_rewrite_ctx_ex(
      ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, index, bitcasted_value, poly_arg_none()),
      poly_pm_float_decomp(), &fctx, true
  );
  ASSERT_NOT_NULL(numeric_store);
  ASSERT_INT_EQ(numeric_store->op, POLY_OP_STORE);
  ASSERT_TRUE(poly_dtype_eq(numeric_store->src[0]->dtype, POLY_UINT16));
  /* tinygrad@2026-08-22/a9069c177a9d do_dtype_decomps is bottom-up: the
   * BITCAST child converts before the raw STORE rule matches. */
  ASSERT_INT_EQ(numeric_store->src[1]->op, POLY_OP_WHERE);
  ASSERT_TRUE(poly_dtype_eq(numeric_store->src[1]->dtype, POLY_UINT16));

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, rewrite_emulates_bf16_vector_load_before_render) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *output = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *input = poly_test_program_param(ctx, POLY_BFLOAT16, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *output_idx = poly_uop_index(ctx, output, &range, 1);
  PolyUOp *input_idx = poly_uop_index(ctx, input, &range, 1);
  PolyUOp *value = poly_uop1(
      ctx, POLY_OP_CAST, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_BFLOAT16, input_idx, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, output_idx, value, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_bf16_load");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_FALSE(
        scalar.priority == POLY_BFLOAT16.priority && strcmp(scalar.name, POLY_BFLOAT16.name) == 0
    );
  }

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);
  const char *path = "temp/polygrad_test_bf16_vector_load.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_bf16_vector_load(path), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, rewrite_emulates_bf16_vector_store_before_render) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *input = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *output = poly_test_program_param(ctx, POLY_BFLOAT16, 4, 1);
  PolyUOp *bound = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_LOOP));
  PolyUOp *input_idx = poly_uop_index(ctx, input, &range, 1);
  PolyUOp *output_idx = poly_uop_index(ctx, output, &range, 1);
  PolyUOp *loaded = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, input_idx, poly_arg_none());
  PolyUOp *as_bf16 = poly_uop1(ctx, POLY_OP_CAST, POLY_BFLOAT16, loaded, poly_arg_none());
  PolyUOp *doubled = poly_uop2(
      ctx, POLY_OP_MUL, POLY_BFLOAT16, as_bf16,
      poly_uop0(ctx, POLY_OP_CONST, POLY_BFLOAT16, poly_arg_float(2.0)), poly_arg_none()
  );
  PolyUOp *value = doubled;
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, output_idx, value, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_bf16_store");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_FALSE(
        scalar.priority == POLY_BFLOAT16.priority && strcmp(scalar.name, POLY_BFLOAT16.name) == 0
    );
  }

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);
  const char *path = "temp/polygrad_test_bf16_vector_store.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_bf16_vector_store(path), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_unary) {
  WasmVecKernel k = wasm_make_vec_unary(POLY_OP_NEG, 10);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);

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
  int N = 10;

  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT32, N, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT32, N, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT32, N, 2);
  PolyUOp *p3 = poly_test_program_param(ctx, POLY_FLOAT32, N, 3);

  PolyUOp *bound = poly_const_int(ctx, N);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);
  PolyUOp *idx3 = poly_uop_index(ctx, p3, &range, 1);

  PolyUOp *la = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx0, poly_arg_none());
  PolyUOp *lb = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx1, poly_arg_none());
  PolyUOp *lc = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx2, poly_arg_none());

  PolyUOp *add = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, la, lb, poly_arg_none());
  PolyUOp *mul = poly_uop2(ctx, POLY_OP_MUL, POLY_FLOAT32, add, lc, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx3, mul, poly_arg_none());
  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_chain");

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);

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
  int N = 8;

  PolyUOp *p0 = poly_test_program_param(ctx, POLY_UINT32, N, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_UINT32, N, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_UINT32, N, 2);

  PolyUOp *bound = poly_const_int(ctx, N);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);

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
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_unsigned_alu");

  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);

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

TEST(wasm, structural_vectorized_add_uses_direct_simd) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:289-356 and
   * renderer/cstyle.py:52-54 keep vector width in shaped UOps and STACK.
   * The binary Wasm renderer must lower that representation to one v128 ADD. */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 16);
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(k.ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD) >= 1);
  ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT), 0);
  ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_REPLACE), 0);

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
      {POLY_OP_ADD, WASM_SIMD_F32X4_ADD, "add"}, {POLY_OP_SUB, WASM_SIMD_F32X4_SUB, "sub"},
      {POLY_OP_MUL, WASM_SIMD_F32X4_MUL, "mul"}, {POLY_OP_FDIV, WASM_SIMD_F32X4_DIV, "div"},
      {POLY_OP_MAX, WASM_SIMD_F32X4_MAX, "max"},
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    WasmVecKernel k = wasm_make_direct_vec_binop(cases[i].op);
    PolyCtx *ctx = k.ctx;
    int n_lin = 0;
    PolyUOp **lin = poly_toposort_alloc(k.ctx, k.sink, &n_lin);
    ASSERT_NOT_NULL(lin);

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
    ASSERT_NOT_NULL(wasm);
    ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, cases[i].opcode) >= 1);
    ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT), 0);
    ASSERT_INT_EQ(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_REPLACE), 0);

    char path[128];
    snprintf(path, sizeof(path), "temp/polygrad_test_f32x4_%s.wasm", cases[i].name);
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
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));
  PolyUOp *selected = poly_rewrite_wasm(ctx, body);
  ASSERT_NOT_NULL(selected);
  ASSERT_TRUE(poly_wasm_can_render_matmul(selected));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(selected, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_ab.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab.wasm"), 0);
  free(wasm);

  int relaxed_size = 0;
  wasm = poly_render_wasm_matmul(body, &relaxed_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(relaxed_size > 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_ab_relaxed.wasm", wasm, relaxed_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_relaxed.wasm"), 0);
  free(wasm);

  PolyUOp *b0 = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *bt = poly_permute(ctx, b0, (int64_t[]){1, 0}, 2);
  PolyUOp *out_t = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * n), shape2, 2);
  PolyUOp *sink_t = poly_sink1(ctx, poly_store_val(ctx, out_t, poly_dot(ctx, a, bt)));
  PolyUOp *linear_schedule_t = poly_test_create_linear(ctx, sink_t);
  ASSERT_TRUE(linear_schedule_t != NULL);
  PolyUOp *body_t = poly_test_linear_call_body(linear_schedule_t, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body_t));

  wasm = poly_render_wasm_matmul(body_t, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_abt.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_abt.wasm"), 0);
  free(wasm);

  wasm = poly_render_wasm_matmul(body_t, &relaxed_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(relaxed_size > 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_abt_relaxed.wasm", wasm, relaxed_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_abt_relaxed.wasm"), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, generic_kernel_crosses_full_rewrite_before_program) {
  /* Tinygrad 2026-08-22/a9069c177a9d full_rewrite_to_sink commits every
   * non-CONST weak dtype before spec_program (codegen/__init__.py:292-396). */
  WasmVecKernel kernel = wasm_make_vec_binop(POLY_OP_ADD, 64);
  PolyCtx *ctx = kernel.ctx;
  ASSERT_NOT_NULL(ctx);
  PolyUOp *scheduled = kernel.sink;
  ASSERT_FALSE(poly_wasm_can_render_matmul(scheduled));
  ASSERT_FALSE(poly_wasm_can_render_reduce(scheduled));

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, scheduled);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_TRUE(poly_type_verify_program(ctx, rewritten));
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++)
    if (topo[i]->op != POLY_OP_CONST) ASSERT_FALSE(poly_dtype_is_weak(topo[i]->dtype));

  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, f64_cos_keeps_full_precision_weak_literal) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/__init__.py:381 and
   * uop/ops.py:603 retain CAST(double, CONST(weakfloat, pi/2)). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT64, 1, 0);
  PolyUOp *in = poly_test_program_param(ctx, POLY_FLOAT64, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *in_idx = poly_uop_index(ctx, in, &zero, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, in_idx, poly_arg_none());
  PolyUOp *value = poly_cos(ctx, load);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_f64_cos");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  bool found_half_pi = false;
  for (int i = 0; i < n_topo; i++) {
    PolyUOp *u = topo[i];
    if (u->op != POLY_OP_CAST || !poly_dtype_eq(u->dtype, POLY_FLOAT64) || u->n_src != 1 ||
        u->src[0]->op != POLY_OP_CONST || !poly_dtype_eq(u->src[0]->dtype, POLY_WEAKFLOAT) ||
        u->src[0]->arg.kind != POLY_ARG_FLOAT)
      continue;
    if (fabs(u->src[0]->arg.f - 1.57079632679489661923) < 1e-15) found_half_pi = true;
  }
  ASSERT_TRUE(found_half_pi);

  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, f16_round_decomposes_every_compare_operand_to_float32) {
  /* Tinygrad 2026-08-22/a9069c177a9d codegen/decomp/dtype.py:198-214
   * emulates every reachable half producer before spec_program. */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *in = poly_test_program_param(ctx, POLY_FLOAT16, 1, 1);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *out_idx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *in_idx = poly_uop_index(ctx, in, &zero, 1);
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT16, in_idx, poly_arg_none());
  PolyUOp *rounded = poly_round_f(ctx, load);
  PolyUOp *value = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT32, rounded, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_f16_round");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_TRUE(poly_type_verify_program(ctx, rewritten));
  int n_topo = 0;
  PolyUOp **topo = poly_toposort_alloc(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    ASSERT_FALSE(
        scalar.priority == POLY_FLOAT16.priority && scalar.bitsize == POLY_FLOAT16.bitsize
    );
    if (topo[i]->op == POLY_OP_CMPLT || topo[i]->op == POLY_OP_CMPEQ ||
        topo[i]->op == POLY_OP_CMPNE) {
      ASSERT_INT_EQ(topo[i]->n_src, 2);
      ASSERT_TRUE(poly_dtype_eq(topo[i]->src[0]->dtype, topo[i]->src[1]->dtype));
    }
  }

  poly_toposort_free(topo);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, f32_matmul_specializer_rejects_integer_graphs) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  int64_t shape2[2] = {2, 2};
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 4), shape2, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 4), shape2, 2);
  PolyUOp *out = poly_reshape(ctx, poly_test_buffer(ctx, POLY_INT32, 4), shape2, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_NOT_NULL(body);

  ASSERT_FALSE(poly_wasm_can_render_matmul(body));
  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_i32_generic.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_i32_generic.wasm"), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 16, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_ab_k_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_k_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("temp/polygrad_test_matmul_ab_k_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_all_scalar_epilogue) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 3, n = 5, k = 2;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_ab_scalar_epilogue.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_scalar_epilogue.wasm"), 0);
  ASSERT_INT_EQ(
      node_run_wasm_matmul_ab("temp/polygrad_test_matmul_ab_scalar_epilogue.wasm", m, n, k), 0
  );
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_row1_nontransposed_uses_generic_fallback) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 7, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_FALSE(poly_wasm_can_render_matmul(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  const char *path = "temp/polygrad_test_matmul_ab_row1_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab(path, m, n, k), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_m_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 6, n = 16, k = 8;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_ab_m_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_m_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("temp/polygrad_test_matmul_ab_m_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_nonmultiple_n_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 18, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD32_SPLAT) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_ab_n_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_n_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("temp/polygrad_test_matmul_ab_n_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_ab_specializes_combined_m_n_k_tails) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 6, n = 18, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *b = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, k * n), (int64_t[]){k, n}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, b)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_ab_all_tails.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_ab_all_tails.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_ab("temp/polygrad_test_matmul_ab_all_tails.wasm", m, n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 8, n = 16, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *bt0 =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2);
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_abt_k_tail.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_abt_k_tail.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_abt("temp/polygrad_test_matmul_abt_k_tail.wasm", m, n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_row1_specializes_nonmultiple_k_tail) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 16, k = 5;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *bt0 =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2);
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_matmul_abt_row1_k_tail.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_abt_row1_k_tail.wasm"), 0);
  ASSERT_INT_EQ(
      node_run_wasm_matmul_abt_row1("temp/polygrad_test_matmul_abt_row1_k_tail.wasm", n, k), 0
  );
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, matmul_abt_specializes_single_row_token_projection) {
  PolyCtx *ctx = poly_ctx_new();
  int64_t m = 1, n = 16, k = 8;
  PolyUOp *a = poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * k), (int64_t[]){m, k}, 2);
  PolyUOp *bt0 =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, n * k), (int64_t[]){n, k}, 2);
  PolyUOp *bt = poly_permute(ctx, bt0, (int64_t[]){1, 0}, 2);
  PolyUOp *out =
      poly_reshape(ctx, poly_test_buffer(ctx, POLY_FLOAT32, m * n), (int64_t[]){m, n}, 2);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, poly_dot(ctx, a, bt)));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_TRUE(linear_schedule != NULL);
  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_matmul(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_matmul(body, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 0);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE) > 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_matmul_abt_row1.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module("temp/polygrad_test_matmul_abt_row1.wasm"), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_abt_row1("temp/polygrad_test_matmul_abt_row1.wasm", n, k), 0);
  free(wasm);

  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, wide_vector_index_scalar_consumers_use_selected_lanes) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *out = poly_test_uop_param(ctx, POLY_FLOAT32, -1, 0, POLY_ADDR_GLOBAL);

  PolyUOp *lanes[16];
  for (int i = 0; i < 16; i++)
    lanes[i] = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float((double)i + 0.25));
  PolyUOp *wide = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, lanes, 16, poly_arg_none());
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(0.0));
  PolyUOp *one = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(1.0));

  PolyUOp *stores[16];
  for (int i = 0; i < 16; i++) {
    PolyUOp *idx = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(i));
    PolyUOp *dst = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, idx, poly_arg_none());
    PolyUOp *lane_idx = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(i));
    PolyUOp *lane = poly_uop_index(ctx, wide, &lane_idx, 1);
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
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_wide_vector_gep.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_wide_vector_gep_f32(path), 0);

  free(wasm);
  poly_toposort_free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, packed_group_reduce_uses_shaped_loads) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  PolyUOp *x = poly_reshape(ctx, poly_buffer_f32(ctx, 1024 * 1024), (int64_t[]){1024, 1024}, 2);
  PolyUOp *row = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1024, 1}, 2);
  PolyUOp *col = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1, 1024}, 2);
  PolyUOp *row_e = poly_expand(ctx, row, (int64_t[]){1024, 1024}, 2);
  PolyUOp *col_e = poly_expand(ctx, col, (int64_t[]){1024, 1024}, 2);
  PolyUOp *expr = poly_relu(
      ctx, poly_alu2(
               ctx, POLY_OP_SUB,
               poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
               poly_full(ctx, (int64_t[]){1024, 1024}, 2, 0.25)
           )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, 1024), (int64_t[]){1024}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  int n_lin = 0;
  PolyUOp **lin =
      wasm_linearize_generic_test(ctx, poly_test_linear_call_body(linear_schedule, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int n_range = 0;
  int n_reg_storage = 0;
  int n_load = 0;
  for (int i = 0; i < n_lin; i++) {
    if (lin[i]->op == POLY_OP_RANGE) n_range++;
    if (lin[i]->op == POLY_OP_BUFFER && lin[i]->arg.kind == POLY_ARG_PARAM && lin[i]->arg.param &&
        lin[i]->arg.param->addrspace == POLY_ADDR_REG)
      n_reg_storage++;
    if (lin[i]->op == POLY_OP_LOAD) n_load++;
  }
  /* Pinned tinygrad codegen/__init__.py:36-44 removes pointer dtype metadata
   * from final BUFFERs but preserves ParamArg.addrspace. The register tile is
   * therefore BUFFER(REG). */
  ASSERT_INT_EQ(n_range, 2);
  ASSERT_TRUE(n_reg_storage >= 1);
  /* tinygrad@2026-08-22/a9069c177a9d represents vector width through shaped
   * indexes; the matching rewritten reduction has 14 scalar LOAD UOps. */
  ASSERT_INT_EQ(n_load, 14);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  int n_shuffle = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_I8X16_SHUFFLE);
  int n_extract = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_EXTRACT);
  int n_replace = wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_REPLACE);
  ASSERT_TRUE(n_shuffle + n_extract + n_replace > 0);

  free(wasm);
  free(lin);
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
      ctx, poly_alu2(
               ctx, POLY_OP_SUB,
               poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
               poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
           )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  int n_lin = 0;
  PolyUOp **lin =
      wasm_linearize_generic_test(ctx, poly_test_linear_call_body(linear_schedule, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_broadcast_reduce_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
  free(lin);
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
      ctx, poly_alu2(
               ctx, POLY_OP_SUB,
               poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
               poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
           )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_reduce(body));

  /* Wasm's approved row-reduce renderer consumes the scheduled Tinygrad-form
   * REDUCE before unmatched kernels enter full_rewrite_to_sink. */
  PolyUOp *rewritten = poly_rewrite_wasm(ctx, body);
  ASSERT_NOT_NULL(rewritten);
  ASSERT_TRUE(poly_wasm_can_render_reduce(rewritten));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_reduce(rewritten, &wasm_size);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_size > 8);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD) >= 2);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_MUL) >= 1);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD) >= 2);

  const char *path = "temp/polygrad_test_specialized_row_reduce_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
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
      ctx, poly_alu2(
               ctx, POLY_OP_SUB,
               poly_alu2(ctx, POLY_OP_MUL, poly_alu2(ctx, POLY_OP_ADD, x, row_e), col_e),
               poly_full(ctx, (int64_t[]){n, n}, 2, 0.25)
           )
  );
  PolyUOp *sum = poly_sum_reduce(ctx, expr, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_TRUE(poly_wasm_can_render_reduce(body));

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm_reduce(body, &wasm_size);
  ASSERT_NOT_NULL(wasm);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_F32X4_ADD) >= 2);
  ASSERT_TRUE(wasm_count_simd_opcode(wasm, wasm_size, WASM_SIMD_V128_LOAD) >= 2);

  const char *path = "temp/polygrad_test_specialized_row_reduce_relu_tail.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_broadcast_reduce_relu(path, (int)n), 0);

  free(wasm);
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
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_FALSE(poly_wasm_can_render_reduce(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_f64_row_reduce_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

  free(wasm);
  free(lin);
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
  PolyUOp *lt = poly_alu2(ctx, POLY_OP_CMPLT, x, row_e);
  PolyUOp *gt = poly_alu2(ctx, POLY_OP_CMPLT, row_e, x);
  PolyUOp *condition = poly_alu2(ctx, POLY_OP_AND, lt, gt);
  PolyUOp *zero = poly_full(ctx, (int64_t[]){n, n}, 2, 0.0);
  PolyUOp *selected = poly_alu3(ctx, POLY_OP_WHERE, condition, x, zero);
  PolyUOp *sum = poly_sum_reduce(ctx, selected, 1, 0);
  PolyUOp *out = poly_reshape(ctx, poly_buffer_f32(ctx, n), (int64_t[]){n}, 1);
  PolyUOp *sink = poly_sink1(ctx, poly_store_val(ctx, out, sum));
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  PolyUOp *body = poly_test_linear_call_body(linear_schedule, 0);
  ASSERT_FALSE(poly_wasm_can_render_reduce(body));

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, body, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_noncompare_where_reduce_generic.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

  free(wasm);
  free(lin);
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
  PolyUOp *linear_schedule = poly_test_create_linear(ctx, sink);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_TRUE(linear_schedule->n_src > 0);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, poly_test_linear_call_body(linear_schedule, 0), &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_matmul_bias_relu.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_matmul_bias_relu(path, (int)tokens, (int)d, (int)hidden), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, render_rand_threefry_dag_terminates) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  int64_t shape[1] = {100};
  PolyUOp *rand = poly_rand(ctx, shape, 1, 42);
  PolyUOp *out = poly_buffer_f32(ctx, 100);
  PolyUOp *store = poly_test_store_to_buffer(ctx, out, rand);
  PolyUOp *linear = poly_test_create_linear(ctx, poly_sink1(ctx, store));
  ASSERT_NOT_NULL(linear);

  int rendered = 0;
  for (int i = 0; i < linear->n_src; i++) {
    PolyUOp *body = poly_test_linear_call_body(linear, i);
    if (!body || body->op != POLY_OP_SINK) continue;
    int n_lin = 0;
    PolyUOp **lin = poly_linearize_wasm(ctx, body, &n_lin);
    ASSERT_NOT_NULL(lin);
    ASSERT_TRUE(n_lin > 0);

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
    ASSERT_NOT_NULL(wasm);
    ASSERT_TRUE(wasm_size > 8);
    free(wasm);
    free(lin);
    rendered++;
  }
  /* tinygrad@2026-08-22/a9069c177a9d lowers RNG after schedule_linear wraps
   * each compiler kernel in a LINEAR/CALL/SINK occurrence. */
  ASSERT_TRUE(rendered > 0);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, full_rewrite_retains_native_u64_threefry_buffers) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyUOp *out = poly_test_program_param(ctx, POLY_UINT64, 1, 0);
  PolyUOp *xbuf = poly_test_program_param(ctx, POLY_UINT64, 1, 1);
  PolyUOp *kbuf = poly_test_program_param(ctx, POLY_UINT64, 1, 2);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *oidx = poly_uop_index(ctx, out, &zero, 1);
  PolyUOp *xidx = poly_uop_index(ctx, xbuf, &zero, 1);
  PolyUOp *kidx = poly_uop_index(ctx, kbuf, &zero, 1);
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, xidx, poly_arg_none());
  PolyUOp *key = poly_uop1(ctx, POLY_OP_LOAD, POLY_UINT64, kidx, poly_arg_none());
  PolyUOp *value = poly_uop2(ctx, POLY_OP_THREEFRY, POLY_UINT64, x, key, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oidx, value, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_threefry_u64");

  PolyUOp *rewritten = poly_rewrite_wasm(ctx, sink);
  ASSERT_NOT_NULL(rewritten);
  int n_topo = 0;
  PolyUOp **topo = poly_toposort(ctx, rewritten, &n_topo);
  ASSERT_NOT_NULL(topo);
  ASSERT_INT_EQ(wasm_count_uops(topo, n_topo, POLY_OP_THREEFRY), 0);
  ASSERT_INT_EQ(wasm_count_uops(topo, n_topo, POLY_OP_LOAD), 2);
  ASSERT_INT_EQ(wasm_count_uops(topo, n_topo, POLY_OP_STORE), 1);
  int n_long = 0;
  for (int i = 0; i < n_topo; i++) {
    PolyDType scalar = topo[i]->dtype;
    if (poly_dtype_is_int(scalar) && scalar.bitsize == 64) n_long++;
    if (topo[i]->op == POLY_OP_PARAM) {
      ASSERT_INT_EQ(topo[i]->dtype.bitsize, 64);
      ASSERT_INT_EQ(topo[i]->n_src, 1);
      /* tinygrad@2026-08-22/a9069c177a9d full_rewrite_to_sink commits the
       * placeholder extent to the renderer index dtype. */
      ASSERT_INT_EQ(topo[i]->src[0]->op, POLY_OP_CAST);
      ASSERT_INT_EQ(topo[i]->src[0]->n_src, 1);
      ASSERT_INT_EQ(topo[i]->src[0]->src[0]->op, POLY_OP_CONST);
      ASSERT_INT_EQ(topo[i]->src[0]->src[0]->arg.i, 1);
    }
  }
  ASSERT_TRUE(n_long > 0);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, rewritten, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write to tmp file for external validation */
  FILE *f = fopen("temp/polygrad_test_vecadd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  /* Try running wasm-validate if available */
  int rc =
      system("which wasm-validate > /dev/null 2>&1 && wasm-validate temp/polygrad_test_vecadd.wasm"
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
  PolyUOp *target_buf = poly_test_buffer(ctx, POLY_INT32, 2);
  PolyUOp *logits = poly_reshape(ctx, logits_buf, (int64_t[]){2, 3}, 2);
  PolyUOp *target = poly_reshape(ctx, target_buf, (int64_t[]){2}, 1);
  PolyUOp *loss = poly_cross_entropy(ctx, logits, target, 1);

  PolyUOp *targets[] = {loss};
  PolyUOp *realized[] = {NULL};
  PolyVarBinding *vars = NULL;
  int n_vars = 0;
  PolyUOp *linear_schedule = poly_linear_with_vars(ctx, targets, 1, realized, &vars, &n_vars);
  ASSERT_NOT_NULL(linear_schedule);
  ASSERT_INT_EQ(linear_schedule->n_src, 3);

  bool saw_i64_index = false;
  bool saw_i32_wrap = false;

  for (int item = 0; item < linear_schedule->n_src; item++) {
    int n_lin = 0;
    PolyUOp **lin =
        poly_linearize_wasm(ctx, poly_test_linear_call_body(linear_schedule, item), &n_lin);
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
    uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
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
    snprintf(path, sizeof(path), "temp/polygrad_test_ce_sparse_item%d.wasm", item);
    FILE *f = fopen(path, "wb");
    ASSERT_NOT_NULL(f);
    fwrite(wasm, 1, (size_t)wasm_size, f);
    fclose(f);

    const char *node = poly_test_node_cmd_for_wasm(path);
    if (node) {
      char cmd[512];
      snprintf(
          cmd, sizeof(cmd),
          "%s -e \"const fs=require('fs'); new WebAssembly.Module(fs.readFileSync('%s'))\"", node,
          path
      );
      ASSERT_INT_EQ(system(cmd), 0);
    }

    free(wasm);
    free(lin);
  }

  if (saw_i64_index) ASSERT_TRUE(saw_i32_wrap);

  free(vars);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, mixed_width_compare_validates) {
  PolyCtx *ctx = poly_ctx_new();

  /* tinygrad's UOp spec wants comparison operands to share a base dtype, but
   * imported/index-heavy graphs can expose a late widened label compared with
   * an int bound. WASM has no implicit casts, so the renderer must coerce the
   * narrower operand before emitting i64.lt_s/i64.eq/etc. */
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT32, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT32, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT64, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(9));
  PolyUOp *lt = poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, lt, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int32_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i32(ctx, lin, n_lin, "mixed_width_compare_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_mixed_width_compare.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i32("temp/polygrad_test_mixed_width_compare.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 1);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST_BACKEND(wasm, sparse_param_slots_use_dense_abi_positions) {
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);

  /* Tinygrad 2026-08-22/a9069c177a9d keeps original CALL slots in PARAM and
   * ProgramInfo.globals, but the rendered function receives only used globals
   * in sorted slot order (uop/ops.py:1239-1259, cstyle.py:204-220). */
  PolyUOp *out = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *left = poly_test_program_param(ctx, POLY_FLOAT32, 2, 2);
  PolyUOp *right = poly_test_program_param(ctx, POLY_FLOAT32, 2, 3);
  PolyUOp *stores[4] = {0};
  for (int i = 0; i < 4; i++) {
    PolyUOp *out_i = poly_const_int(ctx, i);
    PolyUOp *in_i = poly_const_int(ctx, i & 1);
    PolyUOp *src = i < 2 ? left : right;
    PolyUOp *out_idx = poly_uop_index(ctx, out, &out_i, 1);
    PolyUOp *in_idx = poly_uop_index(ctx, src, &in_i, 1);
    PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_idx, poly_arg_none());
    stores[i] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, load, poly_arg_none());
  }
  PolyUOp *sink = poly_test_kernel_sink(ctx, stores, 4, "wasm_sparse_params");
  ASSERT_NOT_NULL(sink);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_sparse_params.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_sparse_f32_params(path), 0);

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
  PolyUOp *out = poly_test_uop_param(ctx, POLY_INT64, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_INT64, out, zero, poly_arg_none());
  PolyUOp *lhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(7));
  PolyUOp *rhs = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *shl = poly_uop2(ctx, POLY_OP_SHL, POLY_INT64, lhs, rhs, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, shl, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store);

  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(ctx, lin, n_lin, "mixed_width_shift_c", &c_out), 0);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_mixed_width_shift.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_i64("temp/polygrad_test_mixed_width_shift.wasm", c_out), 0);
  ASSERT_INT_EQ(c_out, 28);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, exact_uint64_bigint_const_executes_as_fixed_width_bits) {
  /* Pinned tinygrad retains this value as a Python-int UOp argument and
   * truncates only at a fixed-width backend boundary
   * (uop/ops.py:1176-1197, dtype.py:92-100). */
  PolyCtx *ctx = poly_ctx_new();
  ASSERT_NOT_NULL(ctx);
  PolyInt value = {0};
  ASSERT_TRUE(poly_int_from_decimal(&value, "18446744073709550593"));
  PolyUOp *constant = poly_uop0(ctx, POLY_OP_CONST, POLY_UINT64, poly_int_as_arg(&value));
  poly_int_free(&value);
  PolyUOp *out = poly_test_uop_param(ctx, POLY_UINT64, 1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, out, zero, poly_arg_none());
  PolyUOp *sink =
      poly_sink1(ctx, poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, constant, poly_arg_none()));
  int n_lin = 0;
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_test_exact_uint64_bigint.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);
  ASSERT_INT_EQ(node_run_wasm_u64(path, "18446744073709550593"), 0);

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
  PolyUOp *out = poly_test_uop_param(ctx, POLY_UINT64, -1, 0, POLY_ADDR_GLOBAL);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_UINT64, out, zero, poly_arg_none());
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
  PolyUOp **lin = poly_do_linearize(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  bool found_div_u = false, found_rem_u = false;
  for (int i = 0; i < wasm_size; i++) {
    if (wasm[i] == WASM_OP_I64_DIV_U) found_div_u = true;
    if (wasm[i] == WASM_OP_I64_REM_U) found_rem_u = true;
  }
  ASSERT_TRUE(found_div_u);
  ASSERT_TRUE(found_rem_u);

  int64_t c_out = 0;
  ASSERT_INT_EQ(wasm_run_c_i64(ctx, lin, n_lin, "unsigned_i64_div_mod_c", &c_out), 0);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_unsigned_i64_div_mod.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(node_run_wasm_i64("temp/polygrad_test_unsigned_i64_div_mod.wasm", c_out), 0);
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
  PolyUOp *buf = poly_test_program_param(ctx, POLY_FLOAT32, 4, 0);
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i1 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(1));
  PolyUOp *i2 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(2));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *out_idx = poly_uop_index(ctx, buf, &i0, 1);
  PolyUOp *x_idx = poly_uop_index(ctx, buf, &i1, 1);
  PolyUOp *y_idx = poly_uop_index(ctx, buf, &i2, 1);
  PolyUOp *z_idx = poly_uop_index(ctx, buf, &i3, 1);
  PolyUOp *x = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, x_idx, poly_arg_none());
  PolyUOp *y = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, y_idx, poly_arg_none());
  PolyUOp *z = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, z_idx, poly_arg_none());
  PolyUOp *mulacc_src[3] = {x, y, z};
  PolyUOp *mulacc = poly_uop(ctx, POLY_OP_MULACC, POLY_FLOAT32, mulacc_src, 3, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, mulacc, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &store, 1, "wasm_mulacc_order");

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 2.0f, 3.0f, 5.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(ctx, lin, n_lin, "mulacc_operand_order_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 11.0f, 1e-6);
  ASSERT_INT_EQ(
      wasm_write_module("temp/polygrad_test_mulacc_operand_order.wasm", wasm, wasm_size), 0
  );
  ASSERT_INT_EQ(
      node_run_wasm_f32_buffer("temp/polygrad_test_mulacc_operand_order.wasm", c_buf[0]), 0
  );

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

TEST(wasm, reg_buffer_constant_indexes_match_c_renderer) {
  PolyCtx *ctx = poly_ctx_new();

  /* Current BUFFER(REG) constant indexes are distinct local-array elements. */
  PolyUOp *size = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(4));
  PolyParamArg out_arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyParamArg reg_arg = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_REG};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&out_arg));
  PolyUOp *reg = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&reg_arg));
  PolyUOp *i0 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(0));
  PolyUOp *i3 = poly_uop0(ctx, POLY_OP_CONST, POLY_INT32, poly_arg_int(3));
  PolyUOp *two = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(2.0));
  PolyUOp *five = poly_uop0(ctx, POLY_OP_CONST, POLY_FLOAT32, poly_arg_float(5.0));

  PolyUOp *reg0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, reg, i0, poly_arg_none());
  PolyUOp *store0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg0, two, poly_arg_none());
  PolyUOp *after0_src[2] = {reg, store0};
  PolyUOp *after0 = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after0_src, 2, poly_arg_none());

  PolyUOp *reg3 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, after0, i3, poly_arg_none());
  PolyUOp *store3 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, reg3, five, poly_arg_none());
  PolyUOp *after1_src[2] = {after0, store3};
  PolyUOp *after1 = poly_uop(ctx, POLY_OP_AFTER, POLY_FLOAT32, after1_src, 2, poly_arg_none());

  PolyUOp *load0 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, after1, i0, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *load3 = poly_uop1(
      ctx, POLY_OP_LOAD, POLY_FLOAT32,
      poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, after1, i3, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sum = poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load0, load3, poly_arg_none());
  PolyUOp *out_idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, i0, poly_arg_none());
  PolyUOp *store_out = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, out_idx, sum, poly_arg_none());
  PolyUOp *sink = poly_sink1(ctx, store_out);

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);

  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  float c_buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  ASSERT_INT_EQ(wasm_run_c_f32_buffer(ctx, lin, n_lin, "reg_buffer_array_c", c_buf), 0);
  ASSERT_FLOAT_EQ(c_buf[0], 7.0f, 1e-6);
  ASSERT_INT_EQ(wasm_write_module("temp/polygrad_test_reg_buffer_array.wasm", wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_run_wasm_f32_buffer("temp/polygrad_test_reg_buffer_array.wasm", c_buf[0]), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}

/* The binary backend implements pinned CStyleLanguage's REG and masked LOAD
 * semantics. Check guest memory too: returning the right value is insufficient
 * if a register STORE accidentally writes into the shared linear heap. */
static int wasm_check_memory_module(
    PolyCtx *ctx,
    PolyUOp **ops,
    int n,
    bool simd,
    const char *view,
    const char *expected,
    const char *setup,
    const char *args
) {
  const char *node = poly_test_node_cmd();
  if (!node) return -1;
  PolyUOp **linear = malloc((size_t)n * sizeof(*linear));
  if (!linear) return -1;
  int n_linear = 0;
  for (int i = 0; i < n; i++) {
    bool seen = false;
    for (int j = 0; j < n_linear; j++)
      if (linear[j] == ops[i]) seen = true;
    if (!seen) linear[n_linear++] = ops[i];
  }
  int size = 0;
  uint8_t *bytes = poly_render_wasm(ctx, linear, n_linear, &size, simd);
  free(linear);
  int rc = bytes ? wasm_write_module("temp/polygrad_test_memory.wasm", bytes, size) : -1;
  free(bytes);
  if (rc) return rc;
  char cmd[4096];
  snprintf(
      cmd, sizeof(cmd),
      "%s -e \"const fs=require('fs'),assert=require('assert/strict');"
      "const mem=new WebAssembly.Memory({initial:1});"
      "new Uint8Array(mem.buffer).fill(165,0,32);%s;"
      "const mod=new WebAssembly.Module(fs.readFileSync('temp/polygrad_test_memory.wasm'));"
      "const inst=new WebAssembly.Instance(mod,{env:{memory:mem}});"
      "inst.exports.kernel(%s);const expected=%s;"
      "assert.deepEqual(Array.from(new %s(mem.buffer,64,expected.length)),expected);"
      "assert.ok(new Uint8Array(mem.buffer,0,32).every(x=>x===165),'REG corrupted heap');\"",
      node, setup, args, expected, view
  );
  return system(cmd);
}

static PolyUOp *wasm_test_literal(PolyCtx *ctx, PolyDType dtype, int value) {
  return poly_uop0(
      ctx, POLY_OP_CONST, dtype,
      poly_dtype_is_bool(dtype)    ? poly_arg_bool(value != 0)
      : poly_dtype_is_float(dtype) ? poly_arg_float(value)
                                   : poly_arg_int(value)
  );
}

TEST(wasm, memory_reg_dtype_and_lane_matrix) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  PolyDType types[] = {POLY_BOOL,   POLY_INT8,  POLY_UINT8,  POLY_INT16,   POLY_UINT16, POLY_INT32,
                       POLY_UINT32, POLY_INT64, POLY_UINT64, POLY_FLOAT32, POLY_FLOAT64};
  int indices[] = {0, 3, 4};
  int failures = 0;
  for (int d = 0; d < 11; d++)
    for (int a = 0; a < 2; a++)
      for (int i = 0; i < 3; i++) {
        PolyCtx *ctx = poly_ctx_new();
        PolyDType dt = types[d];
        PolyUOp *size = poly_const_int(ctx, 5);
        PolyParamArg oa = {.slot = 0, .dtype = POLY_FLOAT64, .addrspace = POLY_ADDR_GLOBAL};
        PolyParamArg ra = {.slot = 1, .dtype = dt, .addrspace = POLY_ADDR_REG};
        PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT64, size, poly_arg_param(&oa));
        PolyUOp *reg = poly_uop1(ctx, POLY_OP_BUFFER, dt, size, poly_arg_param(&ra));
        PolyUOp *idx0 = wasm_test_literal(ctx, POLY_INT32, indices[i]);
        PolyUOp *idx1 = wasm_test_literal(ctx, POLY_INT32, (indices[i] + 1) % 5);
        PolyUOp *zero = wasm_test_literal(ctx, POLY_INT32, 0);
        PolyUOp *one = wasm_test_literal(ctx, POLY_INT32, 1);
        PolyUOp *v0 = wasm_test_literal(ctx, dt, d == 0 ? 0 : 3);
        PolyUOp *v1 = wasm_test_literal(ctx, dt, d == 0 ? 1 : 7);
        PolyUOp *r0 = poly_uop2(ctx, POLY_OP_INDEX, dt, reg, idx0, poly_arg_none());
        PolyUOp *r1 = poly_uop2(ctx, POLY_OP_INDEX, dt, reg, idx1, poly_arg_none());
        PolyUOp *s0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, r0, v0, poly_arg_none());
        PolyUOp *s1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, r1, v1, poly_arg_none());
        PolyUOp *after = poly_uop3(ctx, POLY_OP_AFTER, dt, reg, s0, s1, poly_arg_none());
        PolyUOp *l0idx = poly_uop2(ctx, POLY_OP_INDEX, dt, a ? after : reg, idx0, poly_arg_none());
        PolyUOp *l1idx = poly_uop2(ctx, POLY_OP_INDEX, dt, a ? after : reg, idx1, poly_arg_none());
        PolyUOp *l0 = poly_uop1(ctx, POLY_OP_LOAD, dt, l0idx, poly_arg_none());
        PolyUOp *l1 = poly_uop1(ctx, POLY_OP_LOAD, dt, l1idx, poly_arg_none());
        PolyUOp *c0 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, l0, poly_arg_none());
        PolyUOp *c1 = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, l1, poly_arg_none());
        PolyUOp *o0 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, out, zero, poly_arg_none());
        PolyUOp *o1 = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, out, one, poly_arg_none());
        PolyUOp *w0 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, o0, c0, poly_arg_none());
        PolyUOp *w1 = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, o1, c1, poly_arg_none());
        PolyUOp *ops[] = {size, out,   reg,   idx0,  idx1, zero, one, v0, v1, r0, r1, s0,
                          s1,   after, l0idx, l1idx, l0,   l1,   c0,  c1, o0, o1, w0, w1};
        /* Deduplicate direct INDEX/CONST aliases, as in a real LINEAR list. */
        int n = 0;
        for (int j = 0; j < (int)(sizeof(ops) / sizeof(*ops)); j++) {
          bool seen = false;
          for (int k = 0; k < n; k++)
            if (ops[k] == ops[j]) seen = true;
          if (!seen) ops[n++] = ops[j];
        }
        int rc = wasm_check_memory_module(
            ctx, ops, n, true, "Float64Array", d == 0 ? "[0,1]" : "[3,7]", "", "64"
        );
        poly_ctx_destroy(ctx);
        if (rc) {
          fprintf(stderr, "REG dtype=%d after=%d lane=%d failed\n", d, a, indices[i]);
          failures++;
        }
      }
  ASSERT_INT_EQ(failures, 0);
  PASS();
}

static int wasm_test_reg_order(int mode) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *size = poly_const_int(ctx, 1);
  PolyParamArg oa = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyParamArg ra = {.slot = 1, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_REG};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&oa));
  PolyUOp *reg = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&ra));
  PolyUOp *zero = wasm_test_literal(ctx, POLY_INT32, 0);
  PolyUOp *three = wasm_test_literal(ctx, POLY_FLOAT32, 3);
  PolyUOp *seven = wasm_test_literal(ctx, POLY_FLOAT32, 7);
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, reg, zero, poly_arg_none());
  PolyUOp *init = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, three, poly_arg_none());
  PolyUOp *load = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx, poly_arg_none());
  PolyUOp *overwrite = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, seven, poly_arg_none());
  PolyUOp *group = poly_uop1(ctx, POLY_OP_GROUP, POLY_VOID, init, poly_arg_none());
  PolyUOp *value =
      mode == 1 ? poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, load, load, poly_arg_none()) : load;
  PolyUOp *oi = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, zero, poly_arg_none());
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi, value, poly_arg_none());
  PolyUOp *ops[16] = {size, out, reg, zero, three, seven, idx, init};
  int n = 8;
  if (mode != 2) ops[n++] = load;
  ops[n++] = overwrite;
  if (mode == 2) {
    ops[n++] = group;
    ops[n++] = load;
  }
  if (mode == 1) ops[n++] = value;
  ops[n++] = oi;
  ops[n++] = store;
  int rc = wasm_check_memory_module(
      ctx, ops, n, false, "Float32Array", mode == 1 ? "[6]" : "[7]", "", "64"
  );
  poly_ctx_destroy(ctx);
  return rc;
}

TEST(wasm, memory_reg_load_single_use_control) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  ASSERT_INT_EQ(wasm_test_reg_order(0), 0);
  PASS();
}

TEST(wasm, memory_reg_load_multi_use_snapshot) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  ASSERT_INT_EQ(wasm_test_reg_order(1), 0);
  PASS();
}

TEST(wasm, memory_group_does_not_replay_stores) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  ASSERT_INT_EQ(wasm_test_reg_order(2), 0);
  PASS();
}

TEST(wasm, memory_gated_load_dtype_matrix) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  PolyDType types[] = {POLY_BOOL,   POLY_INT8,  POLY_UINT8,  POLY_INT16,   POLY_UINT16, POLY_INT32,
                       POLY_UINT32, POLY_INT64, POLY_UINT64, POLY_FLOAT32, POLY_FLOAT64};
  int failures = 0;
  for (int d = 0; d < 11; d++)
    for (int reg_space = 0; reg_space < 2; reg_space++)
      for (int g = 0; g < 2; g++) {
        PolyCtx *ctx = poly_ctx_new();
        PolyDType dt = types[d];
        PolyUOp *size = poly_const_int(ctx, 1);
        PolyParamArg oa = {.slot = 0, .dtype = POLY_FLOAT64, .addrspace = POLY_ADDR_GLOBAL};
        PolyParamArg ia = {
            .slot = 1, .dtype = dt, .addrspace = reg_space ? POLY_ADDR_REG : POLY_ADDR_GLOBAL};
        PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT64, size, poly_arg_param(&oa));
        PolyUOp *in = poly_uop1(
            ctx, reg_space ? POLY_OP_BUFFER : POLY_OP_PARAM, dt, size, poly_arg_param(&ia)
        );
        PolyUOp *zero = wasm_test_literal(ctx, POLY_INT32, 0);
        PolyUOp *alt = wasm_test_literal(ctx, dt, 11);
        PolyUOp *gate = wasm_test_literal(ctx, POLY_BOOL, g);
        PolyUOp *ii = poly_uop2(ctx, POLY_OP_INDEX, dt, in, zero, poly_arg_none());
        PolyUOp *load = poly_uop3(ctx, POLY_OP_LOAD, dt, ii, alt, gate, poly_arg_none());
        PolyUOp *cast = poly_uop1(ctx, POLY_OP_CAST, POLY_FLOAT64, load, poly_arg_none());
        PolyUOp *oi = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT64, out, zero, poly_arg_none());
        PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi, cast, poly_arg_none());
        PolyUOp *ops[13] = {size, out, in, zero, alt, gate, ii};
        int n = 7;
        if (reg_space) {
          PolyUOp *initial = wasm_test_literal(ctx, dt, 0);
          ops[n++] = initial;
          ops[n++] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, ii, initial, poly_arg_none());
        }
        ops[n++] = load;
        ops[n++] = cast;
        ops[n++] = oi;
        ops[n++] = store;
        int rc = wasm_check_memory_module(
            ctx, ops, n, true, "Float64Array",
            g        ? "[0]"
            : d == 0 ? "[1]"
                     : "[11]",
            "", g ? "64,128" : "64,65536"
        );
        poly_ctx_destroy(ctx);
        if (rc) {
          fprintf(stderr, "gated dtype=%d reg=%d gate=%d failed\n", d, reg_space, g);
          failures++;
        }
      }
  ASSERT_INT_EQ(failures, 0);
  PASS();
}

TEST(wasm, memory_gated_loop_simd_and_tail) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  int lengths[] = {0, 1, 3, 4, 5, 8, 9};
  int failures = 0;
  for (int f64 = 0; f64 < 2; f64++)
    for (int l = 0; l < 7; l++)
      for (int simd = 0; simd < 2; simd++)
        for (int g = 0; g < 3; g++) {
          PolyCtx *ctx = poly_ctx_new();
          PolyDType dt = f64 ? POLY_FLOAT64 : POLY_FLOAT32;
          PolyUOp *size = poly_const_int(ctx, lengths[l]);
          PolyParamArg oa = {.slot = 0, .dtype = dt, .addrspace = POLY_ADDR_GLOBAL};
          PolyParamArg ia = {.slot = 1, .dtype = dt, .addrspace = POLY_ADDR_GLOBAL};
          PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, dt, size, poly_arg_param(&oa));
          PolyUOp *in = poly_uop1(ctx, POLY_OP_PARAM, dt, size, poly_arg_param(&ia));
          PolyUOp *bound = wasm_test_literal(ctx, POLY_INT32, lengths[l]);
          PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
          PolyUOp *alt = wasm_test_literal(ctx, dt, 11);
          PolyUOp *limit = wasm_test_literal(ctx, POLY_INT32, 2);
          PolyUOp *gate =
              g == 2 ? poly_uop2(ctx, POLY_OP_CMPLT, POLY_BOOL, range, limit, poly_arg_none())
                     : wasm_test_literal(ctx, POLY_BOOL, g);
          PolyUOp *ii = poly_uop2(ctx, POLY_OP_INDEX, dt, in, range, poly_arg_none());
          PolyUOp *load = poly_uop3(ctx, POLY_OP_LOAD, dt, ii, alt, gate, poly_arg_none());
          PolyUOp *oi = poly_uop2(ctx, POLY_OP_INDEX, dt, out, range, poly_arg_none());
          PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi, load, poly_arg_none());
          PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, range, poly_arg_none());
          PolyUOp *ops[] = {size, out, in,   bound, alt,   limit, range,
                            gate, ii,  load, oi,    store, end};
          char expected[128], setup[128];
          snprintf(
              expected, sizeof(expected), "Array.from({length:%d},(_,i)=>%s)", lengths[l],
              g == 0   ? "11"
              : g == 1 ? "3"
                       : "i<2?3:11"
          );
          snprintf(
              setup, sizeof(setup), "new %s(mem.buffer,256,9).fill(3)",
              f64 ? "Float64Array" : "Float32Array"
          );
          int rc = wasm_check_memory_module(
              ctx, ops, 13, simd, f64 ? "Float64Array" : "Float32Array", expected, setup,
              g == 0 ? "64,65536" : "64,256"
          );
          poly_ctx_destroy(ctx);
          if (rc) {
            fprintf(stderr, "loop f64=%d n=%d simd=%d gate=%d failed\n", f64, lengths[l], simd, g);
            failures++;
          }
        }
  ASSERT_INT_EQ(failures, 0);
  PASS();
}

TEST(wasm, memory_gated_shrink_vector) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  int failures = 0;
  for (int f64 = 0; f64 < 2; f64++)
    for (int reg_space = 0; reg_space < 2; reg_space++)
      for (int g = 0; g < 2; g++) {
        PolyCtx *ctx = poly_ctx_new();
        PolyDType dt = f64 ? POLY_FLOAT64 : POLY_FLOAT32;
        int lanes = f64 ? 2 : 4;
        PolyUOp *size = poly_const_int(ctx, lanes);
        PolyParamArg oa = {.slot = 0, .dtype = dt, .addrspace = POLY_ADDR_GLOBAL};
        PolyParamArg ia = {
            .slot = 1, .dtype = dt, .addrspace = reg_space ? POLY_ADDR_REG : POLY_ADDR_GLOBAL};
        PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, dt, size, poly_arg_param(&oa));
        PolyUOp *in = poly_uop1(
            ctx, reg_space ? POLY_OP_BUFFER : POLY_OP_PARAM, dt, size, poly_arg_param(&ia)
        );
        PolyUOp *zero = wasm_test_literal(ctx, POLY_INT32, 0);
        PolyUOp *width = wasm_test_literal(ctx, POLY_INT32, lanes);
        PolyUOp *eleven = wasm_test_literal(ctx, dt, 11);
        PolyUOp *alts[] = {eleven, eleven, eleven, eleven};
        PolyUOp *alt = poly_uop(ctx, POLY_OP_STACK, dt, alts, lanes, poly_arg_none());
        PolyUOp *gate = wasm_test_literal(ctx, POLY_BOOL, g);
        PolyUOp *ii = poly_uop3(ctx, POLY_OP_SHRINK, dt, in, zero, width, poly_arg_none());
        PolyUOp *load = poly_uop3(ctx, POLY_OP_LOAD, dt, ii, alt, gate, poly_arg_none());
        PolyUOp *oi = poly_uop3(ctx, POLY_OP_SHRINK, dt, out, zero, width, poly_arg_none());
        PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, oi, load, poly_arg_none());
        PolyUOp *ops[26] = {size, out, in, zero, width, eleven, alt, gate};
        int n = 8;
        if (reg_space) {
          PolyUOp *initial = wasm_test_literal(ctx, dt, 0);
          ops[n++] = initial;
          for (int i = 0; i < lanes; i++) {
            PolyUOp *index = wasm_test_literal(ctx, POLY_INT32, i);
            PolyUOp *addr = poly_uop2(ctx, POLY_OP_INDEX, dt, in, index, poly_arg_none());
            ops[n++] = index;
            ops[n++] = addr;
            ops[n++] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, addr, initial, poly_arg_none());
          }
        }
        ops[n++] = ii;
        ops[n++] = load;
        ops[n++] = oi;
        ops[n++] = store;
        char expected[96], setup[96];
        snprintf(
            expected, sizeof(expected), "Array(%d).fill(%d)", lanes, g ? (reg_space ? 0 : 3) : 11
        );
        snprintf(
            setup, sizeof(setup), "new %s(mem.buffer,256,4).fill(3)",
            f64 ? "Float64Array" : "Float32Array"
        );
        int rc = wasm_check_memory_module(
            ctx, ops, n, true, f64 ? "Float64Array" : "Float32Array", expected, setup,
            g ? "64,256" : "64,65536"
        );
        poly_ctx_destroy(ctx);
        if (rc) {
          fprintf(stderr, "SHRINK f64=%d reg=%d gate=%d failed\n", f64, reg_space, g);
          failures++;
        }
      }
  ASSERT_INT_EQ(failures, 0);
  PASS();
}

TEST(wasm, memory_structural_vector_keeps_shared_scalar_dependencies) {
  if (!poly_test_node_cmd()) SKIP("Node required for Wasm execution");
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *size = poly_const_int(ctx, 5);
  PolyParamArg oa = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *out = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&oa));
  PolyParamArg ia = {.slot = 1, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_GLOBAL};
  PolyUOp *in = poly_uop1(ctx, POLY_OP_PARAM, POLY_FLOAT32, size, poly_arg_param(&ia));
  PolyUOp *ops[32] = {size, out, in}, *values[4], *indices[5], *sums[4];
  int n = 3;
  for (int i = 0; i < 4; i++)
    ops[n++] = values[i] = wasm_test_literal(ctx, POLY_FLOAT32, i + 1);
  for (int i = 0; i < 5; i++)
    ops[n++] = indices[i] = wasm_test_literal(ctx, POLY_INT32, i);
  PolyUOp *in_slice =
      poly_uop3(ctx, POLY_OP_SHRINK, POLY_FLOAT32, in, indices[0], indices[4], poly_arg_none());
  ops[n++] = in_slice;
  PolyUOp *vec = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, in_slice, poly_arg_none());
  ops[n++] = vec;
  for (int i = 0; i < 4; i++) {
    PolyUOp *lane = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, vec, indices[i], poly_arg_none());
    ops[n++] = lane;
    ops[n++] = sums[i] =
        poly_uop2(ctx, POLY_OP_ADD, POLY_FLOAT32, lane, values[0], poly_arg_none());
  }
  PolyUOp *sum = poly_uop(ctx, POLY_OP_STACK, POLY_FLOAT32, sums, 4, poly_arg_none());
  ops[n++] = sum;
  PolyUOp *slice =
      poly_uop3(ctx, POLY_OP_SHRINK, POLY_FLOAT32, out, indices[0], indices[4], poly_arg_none());
  ops[n++] = slice;
  ops[n++] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, slice, sum, poly_arg_none());
  PolyUOp *last = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, out, indices[4], poly_arg_none());
  ops[n++] = last;
  ops[n++] = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, last, sums[0], poly_arg_none());
  int rc = wasm_check_memory_module(
      ctx, ops, n, true, "Float32Array", "[2,3,4,5,2]",
      "new Float32Array(mem.buffer,256,4).set([1,2,3,4])", "64,256"
  );
  poly_ctx_destroy(ctx);
  ASSERT_INT_EQ(rc, 0);
  PASS();
}

TEST(wasm, memory_dynamic_reg_index_fails_closed) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *size = poly_const_int(ctx, 4);
  PolyParamArg ra = {.slot = 0, .dtype = POLY_FLOAT32, .addrspace = POLY_ADDR_REG};
  PolyUOp *reg = poly_uop1(ctx, POLY_OP_BUFFER, POLY_FLOAT32, size, poly_arg_param(&ra));
  PolyUOp *bound = wasm_test_literal(ctx, POLY_INT32, 4);
  PolyUOp *range = poly_uop1(ctx, POLY_OP_RANGE, POLY_INT32, bound, poly_arg_int(0));
  PolyUOp *idx = poly_uop2(ctx, POLY_OP_INDEX, POLY_FLOAT32, reg, range, poly_arg_none());
  PolyUOp *value = wasm_test_literal(ctx, POLY_FLOAT32, 3);
  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx, value, poly_arg_none());
  PolyUOp *end = poly_uop2(ctx, POLY_OP_END, POLY_VOID, store, range, poly_arg_none());
  PolyUOp *ops[] = {size, reg, bound, value, range, idx, store, end};
  int bytes = 0;
  uint8_t *wasm = poly_render_wasm(ctx, ops, 8, &bytes, true);
  bool rejected = wasm == NULL;
  free(wasm);
  poly_ctx_destroy(ctx);
  ASSERT_TRUE(rejected);
  ASSERT_INT_EQ(bytes, 0);
  PASS();
}

TEST(wasm, render_pow) {
  /* POW kernel: c[i] = a[i] ^ b[i] — lowered through tinygrad-style
   * transcendental decomposition, so a math import call is expected. */
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_POW, 4);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("temp/polygrad_e2e_pow.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  const char *node = poly_test_node_cmd_for_wasm("temp/polygrad_e2e_pow.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS();
  }

  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js temp/polygrad_e2e_pow.wasm pow 4", node);
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  /* Write WASM to tmp file */
  FILE *f = fopen("temp/polygrad_e2e_vecadd.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  /* Skip if node is not available */
  const char *node = poly_test_node_cmd_for_wasm("temp/polygrad_e2e_vecadd.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  /* Run the Node.js test runner */
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js temp/polygrad_e2e_vecadd.wasm add 8", node);
  int rc = system(cmd);
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_vecadd_simd) {
  WasmVecKernel k = wasm_make_vec_binop(POLY_OP_ADD, 10);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  const char *path = "temp/polygrad_e2e_vecadd_simd.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);

  const char *node = poly_test_node_cmd_for_wasm(path);
  if (node) {
    char cmd[512];
    snprintf(
        cmd, sizeof(cmd), "%s test/run_wasm.js temp/polygrad_e2e_vecadd_simd.wasm add 10", node
    );
    ASSERT_INT_EQ(system(cmd), 0);
  }

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm, e2e_node_where_simd) {
  WasmVecKernel k = wasm_make_vec_where_f32(10);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  const char *path = "temp/polygrad_e2e_where_simd.wasm";
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
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT64, n, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT64, n, 1);
  PolyUOp *p2 = poly_test_program_param(ctx, POLY_FLOAT64, n, 2);

  PolyUOp *bound = poly_const_int(ctx, n);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);
  PolyUOp *idx2 = poly_uop_index(ctx, p2, &range, 1);

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *load1 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx1, poly_arg_none());

  PolyUOp *alu = poly_uop2(ctx, alu_op, POLY_FLOAT64, load0, load1, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx2, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_vec_binop_f64");

  return (WasmVecKernel){ctx, sink, n};
}

static WasmVecKernel wasm_make_vec_unary_f64(PolyOps alu_op, int n) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *p0 = poly_test_program_param(ctx, POLY_FLOAT64, n, 0);
  PolyUOp *p1 = poly_test_program_param(ctx, POLY_FLOAT64, n, 1);

  PolyUOp *bound = poly_const_int(ctx, n);
  PolyUOp *range =
      poly_uop1(ctx, POLY_OP_RANGE, POLY_WEAKINT, bound, poly_arg_range(0, POLY_AXIS_WEAK));

  PolyUOp *idx0 = poly_uop_index(ctx, p0, &range, 1);
  PolyUOp *idx1 = poly_uop_index(ctx, p1, &range, 1);

  PolyUOp *load0 = poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx0, poly_arg_none());
  PolyUOp *alu = poly_uop1(ctx, alu_op, POLY_FLOAT64, load0, poly_arg_none());

  PolyUOp *store = poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx1, alu, poly_arg_none());

  PolyUOp *end_src[2] = {store, range};
  PolyUOp *end = poly_uop(ctx, POLY_OP_END, POLY_VOID, end_src, 2, poly_arg_none());
  PolyUOp *sink = poly_test_kernel_sink(ctx, &end, 1, "wasm_vec_unary_f64");

  return (WasmVecKernel){ctx, sink, n};
}

/* F64 WASM renderer tests */

TEST(wasm_f64, render_vecadd_f64_scalar) {
  /* Render f64 vecadd kernel in scalar mode -- verify f64 opcodes */
  WasmVecKernel k = wasm_make_vec_binop_f64(POLY_OP_ADD, 10);
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);

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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);

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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);

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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("temp/polygrad_test_f64_scalar.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate temp/polygrad_test_f64_scalar.wasm");
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, true);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("temp/polygrad_test_f64_simd.wasm", "wb");
  if (f) {
    fwrite(wasm, 1, wasm_size, f);
    fclose(f);
  }

  int rc = system("which wasm-validate > /dev/null 2>&1 && "
                  "wasm-validate --enable-simd temp/polygrad_test_f64_simd.wasm");
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
  PolyCtx *ctx = k.ctx;
  int n_lin;
  PolyUOp **lin = poly_linearize_wasm(k.ctx, k.sink, &n_lin);

  int wasm_size;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);

  FILE *f = fopen("temp/polygrad_e2e_vecadd_f64.wasm", "wb");
  ASSERT_NOT_NULL(f);
  fwrite(wasm, 1, wasm_size, f);
  fclose(f);

  const char *node = poly_test_node_cmd_for_wasm("temp/polygrad_e2e_vecadd_f64.wasm");
  if (!node) {
    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
    PASS(); /* skip gracefully */
  }

  char cmd[512];
  snprintf(
      cmd, sizeof(cmd), "%s test/run_wasm.js temp/polygrad_e2e_vecadd_f64.wasm add_f64 8", node
  );
  int rc = system(cmd);
  ASSERT_INT_EQ(rc, 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(k.ctx);
  PASS();
}

TEST(wasm_f64, e2e_node_math_imports_f64) {
  struct {
    const char *name;
    PolyOps op;
    bool binary;
  } cases[] = {
      {"exp2", POLY_OP_EXP2, false},
      {"log2", POLY_OP_LOG2, false},
      {"sin", POLY_OP_SIN, false},
      {"pow", POLY_OP_POW, true},
  };

  for (int i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
    WasmVecKernel k = cases[i].binary ? wasm_make_vec_binop_f64(cases[i].op, 4)
                                      : wasm_make_vec_unary_f64(cases[i].op, 4);
    int n_lin = 0;
    /* Normal full lowering decomposes POW into EXP2+LOG2. Exercise the raw
     * renderer extension directly so the f64 math.pow ABI is covered too. */
    PolyUOp **lin = cases[i].binary ? poly_toposort_alloc(k.ctx, k.sink, &n_lin)
                                    : poly_linearize_wasm(k.ctx, k.sink, &n_lin);
    ASSERT_NOT_NULL(lin);

    int wasm_size = 0;
    uint8_t *wasm = poly_render_wasm(k.ctx, lin, n_lin, &wasm_size, false);
    ASSERT_NOT_NULL(wasm);

    char path[256];
    snprintf(path, sizeof(path), "temp/polygrad_e2e_%s_f64.wasm", cases[i].name);
    ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
    ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

    const char *node = poly_test_node_cmd_for_wasm(path);
    if (node) {
      char cmd[512];
      snprintf(cmd, sizeof(cmd), "%s test/run_wasm.js %s %s_f64 4", node, path, cases[i].name);
      ASSERT_INT_EQ(system(cmd), 0);
    }

    free(wasm);
    free(lin);
    poly_ctx_destroy(k.ctx);
  }
  PASS();
}

TEST(wasm_f64, mixed_f32_f64_unary_import_types_validate) {
  PolyCtx *ctx = poly_ctx_new();
  PolyUOp *out32 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 0);
  PolyUOp *in32 = poly_test_program_param(ctx, POLY_FLOAT32, 1, 1);
  PolyUOp *out64 = poly_test_program_param(ctx, POLY_FLOAT64, 1, 2);
  PolyUOp *in64 = poly_test_program_param(ctx, POLY_FLOAT64, 1, 3);
  PolyUOp *zero = poly_uop0(ctx, POLY_OP_CONST, POLY_WEAKINT, poly_arg_int(0));
  PolyUOp *idx_out32 = poly_uop_index(ctx, out32, &zero, 1);
  PolyUOp *idx_in32 = poly_uop_index(ctx, in32, &zero, 1);
  PolyUOp *idx_out64 = poly_uop_index(ctx, out64, &zero, 1);
  PolyUOp *idx_in64 = poly_uop_index(ctx, in64, &zero, 1);
  PolyUOp *sin32 = poly_uop1(
      ctx, POLY_OP_SIN, POLY_FLOAT32,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT32, idx_in32, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *sin64 = poly_uop1(
      ctx, POLY_OP_SIN, POLY_FLOAT64,
      poly_uop1(ctx, POLY_OP_LOAD, POLY_FLOAT64, idx_in64, poly_arg_none()), poly_arg_none()
  );
  PolyUOp *stores[] = {
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx_out32, sin32, poly_arg_none()),
      poly_uop2(ctx, POLY_OP_STORE, POLY_VOID, idx_out64, sin64, poly_arg_none()),
  };
  PolyUOp *sink = poly_test_kernel_sink(ctx, stores, 2, "wasm_mixed_f32_f64_sin");

  int n_lin = 0;
  PolyUOp **lin = poly_linearize_wasm(ctx, sink, &n_lin);
  ASSERT_NOT_NULL(lin);
  int wasm_size = 0;
  uint8_t *wasm = poly_render_wasm(ctx, lin, n_lin, &wasm_size, false);
  ASSERT_NOT_NULL(wasm);
  const char *path = "temp/polygrad_mixed_f32_f64_sin.wasm";
  ASSERT_INT_EQ(wasm_write_module(path, wasm, wasm_size), 0);
  ASSERT_INT_EQ(node_compile_wasm_module(path), 0);

  free(wasm);
  free(lin);
  poly_ctx_destroy(ctx);
  PASS();
}
