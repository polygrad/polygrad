/*
 * runtime_cpu.c — Compile C source, load with dlopen, execute
 *
 * Write C to tmpfile → clang -O2 -shared -fPIC → dlopen → dlsym → call.
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/wait.h>

struct PolyProgram {
  void *handle;                       /* dlopen handle */
  void (*call_fn)(void **args);       /* fn_name_call wrapper */
  char so_path[256];                  /* kept alive until destroy */
};

static int poly_compile_id = 0;

PolyProgram *poly_compile_c(const char *source, const char *fn_name) {
  /* unique temp file names */
  char c_path[256], so_path[256];
  snprintf(c_path, sizeof(c_path), "/tmp/polygrad_%d_%d.c", (int)getpid(), poly_compile_id);
  snprintf(so_path, sizeof(so_path), "/tmp/polygrad_%d_%d.so", (int)getpid(), poly_compile_id);
  poly_compile_id++;

  /* write source */
  FILE *f = fopen(c_path, "w");
  if (!f) { fprintf(stderr, "polygrad: cannot write %s\n", c_path); return NULL; }
  fputs(source, f);
  fclose(f);
  /* DEBUG: dump kernel source */
  if (getenv("POLY_DUMP_KERNELS")) {
    fprintf(stderr, "=== KERNEL %s ===\n%s\n=== END ===\n", fn_name, source);
  }

  /* compile — use fork/exec/waitpid instead of system() for Node.js compat */
  {
    pid_t pid = fork();
    if (pid == -1) {
      fprintf(stderr, "polygrad: fork failed\n");
      remove(c_path);
      return NULL;
    }
    if (pid == 0) {
      /* child */
      execlp("clang", "clang", "-O2", "-shared", "-fPIC",
             "-o", so_path, c_path, "-lm", (char *)NULL);
      _exit(127); /* exec failed */
    }
    /* parent */
    int status;
    while (waitpid(pid, &status, 0) == -1) {
      /* retry on EINTR */
    }
    int ret = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    remove(c_path);
    if (ret != 0) {
      fprintf(stderr, "polygrad: clang failed (exit %d)\n", ret);
      return NULL;
    }
  }

  /* load */
  void *handle = dlopen(so_path, RTLD_LAZY);
  if (!handle) {
    fprintf(stderr, "polygrad: dlopen: %s\n", dlerror());
    remove(so_path);
    return NULL;
  }

  /* resolve the _call wrapper */
  char call_name[256];
  snprintf(call_name, sizeof(call_name), "%s_call", fn_name);
  void *sym = dlsym(handle, call_name);
  void (*call_fn)(void **);
  memcpy(&call_fn, &sym, sizeof(sym));
  if (!call_fn) {
    fprintf(stderr, "polygrad: dlsym(%s): %s\n", call_name, dlerror());
    dlclose(handle);
    remove(so_path);
    return NULL;
  }

  PolyProgram *prog = malloc(sizeof(PolyProgram));
  prog->handle = handle;
  prog->call_fn = call_fn;
  strncpy(prog->so_path, so_path, sizeof(prog->so_path) - 1);
  prog->so_path[sizeof(prog->so_path) - 1] = '\0';

  return prog;
}

void poly_program_call(PolyProgram *prog, void **args, int n_args) {
  (void)n_args; /* wrapper knows the count */
  prog->call_fn(args);
}

void poly_program_destroy(PolyProgram *prog) {
  if (!prog) return;
  dlclose(prog->handle);
  remove(prog->so_path);
  free(prog);
}
