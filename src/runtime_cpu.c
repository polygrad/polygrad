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
 *   POLY_CPU_ARCH=0 disable host CPU targeting
 *   POLY_CPU_ARCH=x compile with -march=x (default: native)
 */

#define _POSIX_C_SOURCE 200809L

#include "codegen/codegen.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>

struct PolyProgram {
  void *handle; /* dlopen handle */
  void (*call_fn)(void **args); /* fn_name_call wrapper */
  void (*call_core_fn)(void **args, int core_id); /* fn_name_call_core wrapper */
  char so_path[512]; /* kept alive until destroy */
  int cached; /* 1 if loaded from disk cache (don't remove on destroy) */
};

static int poly_compile_id = 0;

typedef struct CachedSoHandle {
  char path[512];
  void *handle;
  struct CachedSoHandle *next;
} CachedSoHandle;

static CachedSoHandle *g_cached_so_handles;

static void *cached_so_handle_get(const char *so_path) {
  for (CachedSoHandle *e = g_cached_so_handles; e; e = e->next)
    if (strcmp(e->path, so_path) == 0) return e->handle;
  return NULL;
}

static void cached_so_handle_put(const char *so_path, void *handle) {
  CachedSoHandle *e = malloc(sizeof(*e));
  if (!e) return;
  strncpy(e->path, so_path, sizeof(e->path) - 1);
  e->path[sizeof(e->path) - 1] = '\0';
  e->handle = handle;
  e->next = g_cached_so_handles;
  g_cached_so_handles = e;
}

/* Content hash (FNV-1a 64-bit) */

static uint64_t source_hash(const char *s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (; *s; s++) {
    h ^= (uint64_t)(unsigned char)*s;
    h *= 0x100000001b3ULL;
  }
  return h;
}

/* Disk cache directory */

static int ensure_cache_dir(char *dir, int cap) {
  const char *xdg = getenv("XDG_CACHE_HOME");
  const char *home = getenv("HOME");
  int n = -1;
  if (xdg && xdg[0])
    n = snprintf(dir, cap, "%s/polygrad", xdg);
  else if (home && home[0])
    n = snprintf(dir, cap, "%s/.cache/polygrad", home);
  else
    return -1;
  if (n < 0 || n >= cap) {
    dir[0] = '\0';
    return -1;
  }

  /* mkdir -p (two levels: ~/.cache, ~/.cache/polygrad) */
  char parent[512];
  int parent_len = snprintf(parent, sizeof(parent), "%s", dir);
  if (parent_len < 0 || parent_len >= (int)sizeof(parent)) return -1;
  char *last_slash = strrchr(parent, '/');
  if (last_slash) {
    *last_slash = '\0';
    mkdir(parent, 0755); /* ignore error if exists */
  }
  if (mkdir(dir, 0755) == -1 && errno != EEXIST) return -1;
  return 0;
}

/* Optimization level */

static const char *opt_flag(void) {
  const char *v = getenv("POLY_OPT");
  if (!v) return "-O2";
  if (v[0] == '0') return "-O0";
  if (v[0] == '1') return "-O1";
  return "-O2";
}

/* CPU target flag.
 * tinygrad's CPU compiler always passes the selected host arch to clang.
 * Polygrad's native JIT also needs that: without F16C/AVX feature selection,
 * clang may lower __fp16 casts into unresolved compiler-rt libcalls. */
static const char *cpu_arch_flag(char *buf, size_t cap) {
  const char *v = getenv("POLY_CPU_ARCH");
  if (!v || !v[0]) v = "native";
  if (strcmp(v, "0") == 0 || strcmp(v, "none") == 0 || strcmp(v, "baseline") == 0) return NULL;
  if (v[0] == '-') return v;
  snprintf(buf, cap, "-march=%s", v);
  return buf;
}

static bool jit_asan_enabled(void) {
#if defined(__SANITIZE_ADDRESS__)
  return true;
#else
  return false;
#endif
}

static const char *compile_tmp_dir(void) {
  const char *dir = getenv("POLY_TMPDIR");
  if (dir && dir[0]) return dir;
  dir = getenv("TMPDIR");
  if (dir && dir[0]) return dir;
  return "/tmp";
}

static int compile_tmp_path(char *out, size_t cap, const char *suffix, int id) {
  const char *dir = compile_tmp_dir();
  int n = snprintf(out, cap, "%s/polygrad_%d_%d.%s", dir, (int)getpid(), id, suffix);
  if (n < 0 || n >= (int)cap) {
    if (cap) out[0] = '\0';
    return -1;
  }
  return 0;
}

/* Load a .so and resolve the _call wrapper */

static PolyProgram *load_so(const char *so_path, const char *fn_name, int cached) {
  void *handle = cached ? cached_so_handle_get(so_path) : NULL;
  if (!handle) {
    /* Resolve JIT module relocations during dlopen. Lazy binding can defer a
     * bad cached shared-object dependency until the first kernel call, which
     * turns a corrupt/stale cache entry into a crash instead of a load failure
     * that the cache path can delete and rebuild.
     *
     * tinygrad's ClangJIT path uses a freestanding object plus jit_loader, so a
     * cached CPU program is executable bytes kept resident by the runtime. The
     * Polygrad C backend still uses dlopen for ABI simplicity; keeping disk
     * cache hits resident gives it the same lifetime shape and avoids repeated
     * dlopen/dlclose cycles for one cached kernel. */
    int flags = RTLD_NOW | RTLD_LOCAL;
#ifdef RTLD_NODELETE
    if (cached) flags |= RTLD_NODELETE;
#endif
    handle = dlopen(so_path, flags);
    if (!handle) {
      if (!cached) fprintf(stderr, "polygrad: dlopen: %s\n", dlerror());
      return NULL;
    }
    if (cached) cached_so_handle_put(so_path, handle);
  }

  char call_name[256];
  snprintf(call_name, sizeof(call_name), "%s_call", fn_name);
  void *sym = dlsym(handle, call_name);
  void (*call_fn)(void **);
  memcpy(&call_fn, &sym, sizeof(sym));
  if (!call_fn) {
    if (!cached) fprintf(stderr, "polygrad: dlsym(%s): %s\n", call_name, dlerror());
    dlclose(handle);
    return NULL;
  }

  char call_core_name[256];
  snprintf(call_core_name, sizeof(call_core_name), "%s_call_core", fn_name);
  void *core_sym = dlsym(handle, call_core_name);
  void (*call_core_fn)(void **, int) = NULL;
  memcpy(&call_core_fn, &core_sym, sizeof(core_sym));

  PolyProgram *prog = malloc(sizeof(PolyProgram));
  prog->handle = handle;
  prog->call_fn = call_fn;
  prog->call_core_fn = call_core_fn;
  strncpy(prog->so_path, so_path, sizeof(prog->so_path) - 1);
  prog->so_path[sizeof(prog->so_path) - 1] = '\0';
  prog->cached = cached;
  return prog;
}

/* Compile C source to .so */

static int compile_to_so_with_flag(
    const char *source,
    const char *c_path,
    const char *so_path,
    const char *cpu_flag
) {
  if (poly_compile_timed_out()) return -1;
  bool timed = poly_compile_deadline_ms > 0;
  FILE *f = fopen(c_path, "w");
  if (!f) {
    fprintf(stderr, "polygrad: cannot write %s\n", c_path);
    return -1;
  }
  fputs(source, f);
  fclose(f);

  pid_t pid = fork();
  if (pid == -1) {
    fprintf(stderr, "polygrad: fork failed\n");
    remove(c_path);
    return -1;
  }
  if (pid == 0) {
    /* Own the compiler and any subprocesses it starts, not the caller's job. */
    if (timed && setpgid(0, 0) != 0) _exit(127);
    /* Tinygrad's Compiler receives source on stdin. A temporary input filename
     * otherwise leaks into ELF bytes and defeats BEAM's compiled-lib dedup. */
    if (!freopen(c_path, "r", stdin)) _exit(127);
    const char *cc = getenv("CC");
    if (!cc) cc = "clang";
    char *args[16];
    int n = 0;
    args[n++] = (char *)cc;
    args[n++] = (char *)opt_flag();
    if (cpu_flag) args[n++] = (char *)cpu_flag;
    args[n++] = "-shared";
    args[n++] = "-fPIC";
    args[n++] = "-fno-math-errno";
    args[n++] = "-o";
    args[n++] = (char *)so_path;
    args[n++] = "-x";
    args[n++] = "c";
    args[n++] = "-";
    args[n++] = "-lm";
    args[n] = NULL;
    execvp(cc, args);
    /* clang not found — try gcc as fallback */
    if (!getenv("CC")) {
      n = 0;
      args[n++] = "gcc";
      args[n++] = (char *)opt_flag();
      if (cpu_flag) args[n++] = (char *)cpu_flag;
      args[n++] = "-shared";
      args[n++] = "-fPIC";
      args[n++] = "-fno-math-errno";
      args[n++] = "-o";
      args[n++] = (char *)so_path;
      args[n++] = "-x";
      args[n++] = "c";
      args[n++] = "-";
      args[n++] = "-lm";
      args[n] = NULL;
      execvp("gcc", args);
    }
    _exit(127);
  }
  if (timed) (void)setpgid(pid, pid);
  int status = 0;
  bool failed = false;
  for (;;) {
    pid_t done = waitpid(pid, &status, timed ? WNOHANG : 0);
    if (done == pid) break;
    if (done < 0) {
      if (errno == EINTR) continue;
      failed = true;
      break;
    }
    if (poly_compile_timed_out()) {
      (void)kill(-pid, SIGKILL);
      (void)kill(pid, SIGKILL);
      while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
      }
      failed = true;
      break;
    }
    struct timespec pause = {.tv_nsec = 1000000};
    (void)nanosleep(&pause, NULL);
  }
  int ret = !failed && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  remove(c_path);
  return ret;
}

PolyProgram *poly_compile_c(const char *source, const char *fn_name) {
  if (poly_dump_kernels_enabled()) {
    fprintf(stderr, "=== KERNEL %s ===\n%s\n=== END ===\n", fn_name, source);
  }

  /* Disk cache: check if we already compiled this exact source */
  /* ASAN-instrumented test binaries are unstable when repeatedly loading
   * uninstrumented cached JIT DSOs. tinygrad's CPU JIT maps freestanding object
   * bytes, not dlopen'ed shared libraries, so this is a Polygrad C-backend
   * loader constraint. Keep release disk caching enabled, but make sanitizer
   * runs compile temporary kernels instead of reusing cached .so files. */
  const char *cache_env = getenv("POLY_CACHE");
  int use_cache = (!cache_env || cache_env[0] != '0') && !jit_asan_enabled();
  char cache_dir[512] = {0};
  char cache_path[512] = {0};
  uint64_t h = 0;
  char cpu_flag_buf[128] = {0};
  const char *cpu_flag = cpu_arch_flag(cpu_flag_buf, sizeof(cpu_flag_buf));

  if (use_cache) {
    h = source_hash(source);
    /* Include opt level and compiler name in hash so caches are separate */
    const char *of = opt_flag();
    for (; *of; of++) {
      h ^= (uint64_t)(unsigned char)*of;
      h *= 0x100000001b3ULL;
    }
    const char *cc_env = getenv("CC");
    if (cc_env) {
      for (const char *p = cc_env; *p; p++) {
        h ^= (uint64_t)(unsigned char)*p;
        h *= 0x100000001b3ULL;
      }
    }
    if (cpu_flag) {
      for (const char *p = cpu_flag; *p; p++) {
        h ^= (uint64_t)(unsigned char)*p;
        h *= 0x100000001b3ULL;
      }
    }

    if (ensure_cache_dir(cache_dir, sizeof(cache_dir)) == 0) {
      int path_len = snprintf(
          cache_path, sizeof(cache_path), "%s/%016llx.so", cache_dir, (unsigned long long)h
      );
      if (path_len < 0 || path_len >= (int)sizeof(cache_path)) {
        cache_path[0] = '\0'; /* path too long: fall through without disk cache */
      }

      /* Try loading cached .so */
      if (cache_path[0] && access(cache_path, F_OK) == 0) {
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
  char c_path[512], so_path[512];
  int compile_id = poly_compile_id++;
  if (compile_tmp_path(c_path, sizeof(c_path), "c", compile_id) != 0 ||
      compile_tmp_path(so_path, sizeof(so_path), "so", compile_id) != 0) {
    fprintf(stderr, "polygrad: temporary compile path is too long\n");
    return NULL;
  }

  int ret = compile_to_so_with_flag(source, c_path, so_path, cpu_flag);
  if (ret != 0) {
    /* DEBUG: dump failed kernel source */
    fprintf(stderr, "polygrad: C compiler failed (exit %d)\n", ret);
    fprintf(stderr, "%s\n", source);
    return NULL;
  }

  /* If disk cache enabled, copy .so to cache for future runs */
  if (use_cache && cache_path[0]) {
    /* Atomic: write to .tmp, then rename (prevents corrupt partial reads) */
    char tmp_path[520];
    int tmp_len = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", cache_path);

    /* Read compiled .so */
    if (tmp_len >= 0 && tmp_len < (int)sizeof(tmp_path)) {
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

typedef struct {
  int core_id;
  bool live;
  pthread_t thread;
} PolyThreadCall;

typedef struct {
  pthread_mutex_t mu;
  pthread_cond_t start_cv;
  pthread_cond_t done_cv;
  PolyThreadCall workers[64];
  PolyProgram *prog;
  void **args;
  int requested_threads;
  int active_workers;
  uint64_t generation;
  bool running;
  bool stop;
} PolyCPUThreadPool;

static PolyCPUThreadPool g_cpu_thread_pool = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .start_cv = PTHREAD_COND_INITIALIZER,
    .done_cv = PTHREAD_COND_INITIALIZER,
};
static pthread_once_t g_cpu_thread_pool_once = PTHREAD_ONCE_INIT;

static void poly_cpu_thread_pool_shutdown(void);

static void poly_cpu_thread_pool_once(void) {
  atexit(poly_cpu_thread_pool_shutdown);
}

static void *poly_thread_call_main(void *opaque) {
  PolyThreadCall *tc = (PolyThreadCall *)opaque;
  PolyCPUThreadPool *p = &g_cpu_thread_pool;
  uint64_t seen_generation = 0;

  pthread_mutex_lock(&p->mu);
  for (;;) {
    while (!p->stop && (!p->running || p->generation == seen_generation ||
                        tc->core_id >= p->requested_threads)) {
      pthread_cond_wait(&p->start_cv, &p->mu);
    }
    if (p->stop) break;

    PolyProgram *prog = p->prog;
    void **args = p->args;
    int core_id = tc->core_id;
    seen_generation = p->generation;
    pthread_mutex_unlock(&p->mu);

    prog->call_core_fn(args, core_id);

    pthread_mutex_lock(&p->mu);
    p->active_workers--;
    if (p->active_workers == 0) pthread_cond_signal(&p->done_cv);
  }
  pthread_mutex_unlock(&p->mu);
  return NULL;
}

static int poly_cpu_thread_cap(int threads) {
  int max_threads = (int)(sizeof(g_cpu_thread_pool.workers) / sizeof(g_cpu_thread_pool.workers[0]));
  if (threads < 1) return 1;
  if (threads > max_threads) return max_threads;
  return threads;
}

static void poly_cpu_thread_pool_ensure_locked(PolyCPUThreadPool *p, int threads) {
  for (int t = 1; t < threads; t++) {
    PolyThreadCall *tc = &p->workers[t];
    if (tc->live) continue;
    tc->core_id = t;
    if (pthread_create(&tc->thread, NULL, poly_thread_call_main, tc) == 0) tc->live = true;
  }
}

static void poly_cpu_thread_pool_shutdown(void) {
  PolyCPUThreadPool *p = &g_cpu_thread_pool;
  pthread_mutex_lock(&p->mu);
  p->stop = true;
  pthread_cond_broadcast(&p->start_cv);
  pthread_mutex_unlock(&p->mu);

  int max_threads = (int)(sizeof(p->workers) / sizeof(p->workers[0]));
  for (int t = 1; t < max_threads; t++) {
    if (!p->workers[t].live) continue;
    pthread_join(p->workers[t].thread, NULL);
    p->workers[t].live = false;
  }
}

void poly_program_call_threaded(PolyProgram *prog, void **args, int n_args, int threads) {
  (void)n_args;
  if (!prog) return;
  if (threads <= 1 || !prog->call_core_fn) {
    prog->call_fn(args);
    return;
  }

  pthread_once(&g_cpu_thread_pool_once, poly_cpu_thread_pool_once);
  threads = poly_cpu_thread_cap(threads);

  PolyCPUThreadPool *p = &g_cpu_thread_pool;
  pthread_mutex_lock(&p->mu);
  while (p->running)
    pthread_cond_wait(&p->done_cv, &p->mu);

  poly_cpu_thread_pool_ensure_locked(p, threads);

  p->prog = prog;
  p->args = args;
  p->requested_threads = threads;
  p->active_workers = 0;
  for (int t = 1; t < threads; t++)
    if (p->workers[t].live) p->active_workers++;
  p->running = true;
  p->generation++;
  pthread_cond_broadcast(&p->start_cv);
  pthread_mutex_unlock(&p->mu);

  prog->call_core_fn(args, 0);

  pthread_mutex_lock(&p->mu);
  for (int t = 1; t < threads; t++) {
    if (p->workers[t].live) continue;
    pthread_mutex_unlock(&p->mu);
    prog->call_core_fn(args, t);
    pthread_mutex_lock(&p->mu);
  }
  while (p->active_workers > 0)
    pthread_cond_wait(&p->done_cv, &p->mu);
  p->running = false;
  p->prog = NULL;
  p->args = NULL;
  pthread_cond_broadcast(&p->done_cv);
  pthread_mutex_unlock(&p->mu);
}

size_t poly_program_estimated_size(const PolyProgram *prog) {
  if (!prog) return 0;
  size_t nbytes = sizeof(*prog);
  struct stat st;
  if (prog->so_path[0] && stat(prog->so_path, &st) == 0 && st.st_size > 0)
    nbytes += (size_t)st.st_size;
  return nbytes;
}

uint8_t *poly_program_read_binary(const PolyProgram *prog, int *size) {
  if (!size) return NULL;
  *size = 0;
  if (!prog) return NULL;
  FILE *f = fopen(prog->so_path, "rb");
  if (!f) return NULL;
  uint8_t *bytes = NULL;
  if (fseek(f, 0, SEEK_END) != 0) goto done;
  long length = ftell(f);
  if (length <= 0 || length > INT_MAX || fseek(f, 0, SEEK_SET) != 0) goto done;
  bytes = malloc((size_t)length);
  if (!bytes) goto done;
  if (fread(bytes, (size_t)length, 1, f) != 1) {
    free(bytes);
    bytes = NULL;
    goto done;
  }
  *size = (int)length;
done:
  fclose(f);
  return bytes;
}

void poly_program_destroy(PolyProgram *prog) {
  if (!prog) return;
  if (!prog->cached) dlclose(prog->handle);
  if (!prog->cached) remove(prog->so_path);
  free(prog);
}
