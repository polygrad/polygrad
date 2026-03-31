{
  "targets": [
    {
      "target_name": "polygrad_napi",
      "sources": [
        "napi_api.c",
        "csrc/alu.c",
        "csrc/arena.c",
        "csrc/autograd.c",
        "csrc/bundle.c",
        "csrc/cJSON.c",
        "csrc/codegen.c",
        "csrc/dtype.c",
        "csrc/exec_plan.c",
        "csrc/frontend.c",
        "csrc/hashmap.c",
        "csrc/indexing.c",
        "csrc/instance.c",
        "csrc/interp.c",
        "csrc/ir.c",
        "csrc/nn.c",
        "csrc/tensor.c",
        "csrc/ops.c",
        "csrc/pat.c",
        "csrc/rangeify.c",
        "csrc/recipe.c",
        "csrc/render_c.c",
        "csrc/runtime_cpu.c",
        "csrc/safetensors.c",
        "csrc/sched.c",
        "csrc/shape.c",
        "csrc/sym.c",
        "csrc/uop.c",
        "csrc/wlrn.c",
        "csrc/models/mlp.c",
        "csrc/models/nam.c",
        "csrc/models/tabm.c",
        "csrc/models/registry.c",
        "csrc/models/hf_loader.c",
        "csrc/models/gpt2.c",
        "csrc/models/llama3.c",
        "csrc/models/resnet.c",
        "csrc/models/vit.c"
      ],
      "include_dirs": ["csrc"],
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
          "sources": ["csrc/render_x64.c"],
          "defines": ["POLY_HAS_X64=1"]
        }],
        ["'<!(test -f /usr/include/cuda.h && echo 1 || echo 0)'=='1'", {
          "sources": ["csrc/render_cuda.c", "csrc/runtime_cuda.c"],
          "defines": ["POLY_HAS_CUDA=1"]
        }],
        ["'<!(test -f /opt/rocm/include/hip/hip_runtime.h && echo 1 || ls -d /opt/rocm-*/include/hip/hip_runtime.h 2>/dev/null | head -1 | xargs test -f 2>/dev/null && echo 1 || echo 0)'=='1'", {
          "sources": ["csrc/render_hip.c", "csrc/runtime_hip.c"],
          "defines": ["POLY_HAS_HIP=1"]
        }]
      ]
    }
  ]
}
