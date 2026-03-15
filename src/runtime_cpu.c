/*
 * runtime_cpu.c — Compile C source, load with dlopen, execute
 *
 * Pipeline: C source → content hash → disk cache check → (compile if miss) → dlopen → call
 *
 * Disk cache: ~/.cache/polygrad/<hash>.so persists compiled kernels across restarts.
 * Source-keyed: identical C source = identical .so, regardless of tensor graph.
 * Control:
 *   POLY_CACHE=0    disable disk cache (always recompile)
 *   POLY_OPT=0      use -O0 (fast compile, ~5x faster than -O2)
 *   POLY_OPT=1      use -O1
 *   POLY_OPT=2      use -O2 (default, matches tinygrad)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>

struct PolyProgram {
  void *handle;                       /* dlopen handle */
  void (*call_fn)(void **args);       /* fn_name_call wrapper */
  char so_path[512];                  /* kept alive until destroy */
  int cached;                         /* 1 if loaded from disk cache (don't remove on destroy) */
};

static int poly_compile_id = 0;

/* ── Content hash (FNV-1a 64-bit) ─────────────────────────────────────── */

static uint64_t source_hash(const char *s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (; *s; s++) {
    h ^= (uint64_t)(unsigned char)*s;
    h *= 0x100000001b3ULL;
  }
  return h;
}

/* ── Disk cache directory ─────────────────────────────────────────────── */

static int ensure_cache_dir(char *dir, int cap) {
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  if (xdg && xdg[0])
    snprintf(dir, cap, "%s/polygrad", xdg);
  else if (home && home[0])
    snprintf(dir, cap, "%s/.cache/polygrad", home);
  else
    return -1;

  /* mkdir -p (two levels: ~/.cache, ~/.cache/polygrad) */
  char parent[512];
  snprintf(parent, sizeof(parent), "%s", dir);
  char *last_slash = strrchr(parent, '/');
  if (last_slash) {
    *last_slash = '\0';
    mkdir(parent, 0755); /* ignore error if exists */
  }
  if (mkdir(dir, 0755) == -1 && errno != EEXIST)
    return -1;
  return 0;
}

/* ── Optimization level ───────────────────────────────────────────────── */

static const char *opt_flag(void) {
  const char *v = getenv("POLY_OPT");
  if (!v) return "-O2";
  if (v[0] == '0') return "-O0";
  if (v[0] == '1') return "-O1";
  return "-O2";
}

/* ── Load a .so and resolve the _call wrapper ─────────────────────────── */

static PolyProgram *load_so(const char *so_path, const char *fn_name, int cached) {
  void *handle = dlopen(so_path, RTLD_LAZY);
  if (!handle) {
    if (!cached)
      fprintf(stderr, "polygrad: dlopen: %s\n", dlerror());
    return NULL;
  }

  char call_name[256];
  snprintf(call_name, sizeof(call_name), "%s_call", fn_name);
  void *sym = dlsym(handle, call_name);
  void (*call_fn)(void **);
  memcpy(&call_fn, &sym, sizeof(sym));
  if (!call_fn) {
    if (!cached)
      fprintf(stderr, "polygrad: dlsym(%s): %s\n", call_name, dlerror());
    dlclose(handle);
    return NULL;
  }

  PolyProgram *prog = malloc(sizeof(PolyProgram));
  prog->handle = handle;
  prog->call_fn = call_fn;
  strncpy(prog->so_path, so_path, sizeof(prog->so_path) - 1);
  prog->so_path[sizeof(prog->so_path) - 1] = '\0';
  prog->cached = cached;
  return prog;
}

/* ── Compile C source to .so ──────────────────────────────────────────── */

static int compile_to_so(const char *source, const char *c_path, const char *so_path) {
  FILE *f = fopen(c_path, "w");
  if (!f) { fprintf(stderr, "polygrad: cannot write %s\n", c_path); return -1; }
  fputs(source, f);
  fclose(f);

  pid_t pid = fork();
  if (pid == -1) {
    fprintf(stderr, "polygrad: fork failed\n");
    remove(c_path);
    return -1;
  }
  if (pid == 0) {
    execlp("clang", "clang", opt_flag(), "-shared", "-fPIC",
           "-fno-math-errno", "-o", so_path, c_path, "-lm", (char *)NULL);
    _exit(127);
  }
  int status;
  while (waitpid(pid, &status, 0) == -1) { /* retry on EINTR */ }
  int ret = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  remove(c_path);
  return ret;
}

PolyProgram *poly_compile_c(const char *source, const char *fn_name) {
  if (getenv("POLY_DUMP_KERNELS")) {
    fprintf(stderr, "=== KERNEL %s ===\n%s\n=== END ===\n", fn_name, source);
  }

  /* Disk cache: check if we already compiled this exact source */
  int use_cache = !getenv("POLY_CACHE") || getenv("POLY_CACHE")[0] != '0';
  char cache_dir[512] = {0};
  char cache_path[512] = {0};
  uint64_t h = 0;

  if (use_cache) {
    h = source_hash(source);
    /* Include opt level in hash so -O0 and -O2 caches are separate */
    const char *of = opt_flag();
    for (; *of; of++) {
      h ^= (uint64_t)(unsigned char)*of;
      h *= 0x100000001b3ULL;
    }

    if (ensure_cache_dir(cache_dir, sizeof(cache_dir)) == 0) {
      snprintf(cache_path, sizeof(cache_path), "%s/%016llx.so",
               cache_dir, (unsigned long long)h);

      /* Try loading cached .so */
      if (access(cache_path, F_OK) == 0) {
        PolyProgram *prog = load_so(cache_path, fn_name, 1);
        if (prog) return prog;
        /* Cache entry corrupt — remove and recompile */
        remove(cache_path);
      }
    } else {
      cache_path[0] = '\0'; /* can't create cache dir, fall through */
    }
  }

  /* Cache miss: compile to temp .so */
  char c_path[256], so_path[256];
  snprintf(c_path, sizeof(c_path), "/tmp/polygrad_%d_%d.c", (int)getpid(), poly_compile_id);
  snprintf(so_path, sizeof(so_path), "/tmp/polygrad_%d_%d.so", (int)getpid(), poly_compile_id);
  poly_compile_id++;

  int ret = compile_to_so(source, c_path, so_path);
  if (ret != 0) {
    /* DEBUG: dump failed kernel source */
    fprintf(stderr, "polygrad: clang failed (exit %d)\n", ret);
    fprintf(stderr, "%s\n", source);
    return NULL;
  }

  /* If disk cache enabled, copy .so to cache for future runs */
  if (use_cache && cache_path[0]) {
    /* Atomic: write to .tmp, then rename (prevents corrupt partial reads) */
    char tmp_path[520];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", cache_path);

    /* Read compiled .so */
    FILE *src_f = fopen(so_path, "rb");
    if (src_f) {
      FILE *dst_f = fopen(tmp_path, "wb");
      if (dst_f) {
        char buf[8192];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), src_f)) > 0)
          fwrite(buf, 1, n, dst_f);
        fclose(dst_f);
        rename(tmp_path, cache_path);
      }
      fclose(src_f);
    }
  }

  /* Load the compiled .so */
  PolyProgram *prog = load_so(so_path, fn_name, 0);
  if (!prog) {
    remove(so_path);
  }
  return prog;
}

void poly_program_call(PolyProgram *prog, void **args, int n_args) {
  (void)n_args;
  prog->call_fn(args);
}

void poly_program_destroy(PolyProgram *prog) {
  if (!prog) return;
  dlclose(prog->handle);
  if (!prog->cached)
    remove(prog->so_path);
  free(prog);
}
