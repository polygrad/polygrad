{
  "targets": [
    {
      "target_name": "polygrad_napi",
      "sources": [
        "napi_api.c",
        "csrc/alu.c",
        "csrc/arena.c",
        "csrc/autograd.c",
        "csrc/cJSON.c",
        "csrc/codegen.c",
        "csrc/dtype.c",
        "csrc/frontend.c",
        "csrc/hashmap.c",
        "csrc/indexing.c",
        "csrc/instance.c",
        "csrc/ir.c",
        "csrc/model_mlp.c",
        "csrc/model_nam.c",
        "csrc/model_tabm.c",
        "csrc/nn.c",
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
        "csrc/modelzoo/modelzoo.c",
        "csrc/modelzoo/hf_loader.c",
        "csrc/modelzoo/models/gpt2.c",
        "csrc/modelzoo/models/llama3.c",
        "csrc/modelzoo/models/resnet.c",
        "csrc/modelzoo/models/vit.c"
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
        }]
      ]
    }
  ]
}
