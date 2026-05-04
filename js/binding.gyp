{
  "targets": [
    {
      "target_name": "polygrad_napi",
      "sources": [
        "napi_api.c",
        "csrc/src/alu.c",
        "csrc/src/arena.c",
        "csrc/src/autograd.c",
        "csrc/src/bundle.c",
        "csrc/vendor/cjson/cJSON.c",
        "csrc/src/codegen.c",
        "csrc/src/ctx.c",
        "csrc/src/device.c",
        "csrc/src/dtype.c",
        "csrc/src/engine/realize.c",
        "csrc/src/utils.c",
        "csrc/src/engine/schedule.c",
        "csrc/src/frontend.c",
        "csrc/src/tensor.c",
        "csrc/src/hashmap.c",
        "csrc/src/schedule/indexing.c",
        "csrc/src/instance.c",
        "csrc/src/interp.c",
        "csrc/src/ir.c",
        "csrc/src/nn.c",
        "csrc/src/ops.c",
        "csrc/src/pat.c",
        "csrc/src/placer.c",
        "csrc/src/schedule/rangeify.c",
        "csrc/src/simplify.c",
        "csrc/src/render_c.c",
        "csrc/src/runtime_wasm.c",
        "csrc/src/runtime_webgpu.c",
        "csrc/src/runtime_cpu.c",
        "csrc/src/safetensors.c",
        "csrc/src/shape.c",
        "csrc/src/sym.c",
        "csrc/src/uop.c",
        "csrc/src/wlrn.c",
        "csrc/src/models/mlp.c",
        "csrc/src/models/nam.c",
        "csrc/src/models/tabm.c",
        "csrc/src/models/registry.c",
        "csrc/src/models/hf_loader.c",
        "csrc/src/models/gpt2.c",
        "csrc/src/models/qwen3.c",
        "csrc/src/loaders/decoded.c",
        "csrc/src/loaders/import_error.c",
        "csrc/src/loaders/bind.c",
        "csrc/src/loaders/hf_decode.c",
        "csrc/src/loaders/gguf_decode.c",
        "csrc/src/loaders/gguf_loader.c",
        "csrc/src/loaders/import_desc.c",
        "csrc/src/tokenizer.c"
      ],
      "include_dirs": ["csrc/src", "csrc/vendor/cjson"],
      "cflags": ["-std=c11", "-O2", "-Wall", "-D_POSIX_C_SOURCE=200809L"],
      "conditions": [
        ["OS=='linux'", {
          "libraries": ["-ldl", "-lm"]
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "OTHER_CFLAGS": ["-std=c11"]
          },
          "libraries": ["-ldl", "-lm"]
        }],
        ["target_arch=='x64'", {
          "sources": ["csrc/src/render_x64.c"],
          "defines": ["POLY_HAS_X64=1"]
        }],
        ["'<!(test -f /usr/include/cuda.h && echo 1 || echo 0)'=='1'", {
          "sources": ["csrc/src/render_cuda.c", "csrc/src/runtime_cuda.c"],
          "defines": ["POLY_HAS_CUDA=1"]
        }],
        ["'<!(test -f /opt/rocm/include/hip/hip_runtime.h && echo 1 || ls -d /opt/rocm-*/include/hip/hip_runtime.h 2>/dev/null | head -1 | xargs test -f 2>/dev/null && echo 1 || echo 0)'=='1'", {
          "sources": ["csrc/src/render_hip.c", "csrc/src/runtime_hip.c"],
          "defines": ["POLY_HAS_HIP=1"]
        }]
      ]
    }
  ]
}
