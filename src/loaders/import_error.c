/*
 * import_error.c -- Thread-local error state for model import
 */

#include "import_error.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static _Thread_local PolyImportError tl_error_code = POLY_IMPORT_OK;
static _Thread_local char tl_error_msg[512] = {0};

PolyImportError poly_import_last_error_code(void) {
    return tl_error_code;
}

const char *poly_import_last_error_message(void) {
    return tl_error_msg;
}

void poly_import_error_clear(void) {
    tl_error_code = POLY_IMPORT_OK;
    tl_error_msg[0] = '\0';
}

void poly_import_error_set(PolyImportError code, const char *fmt, ...) {
    tl_error_code = code;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tl_error_msg, sizeof(tl_error_msg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "poly_import: %s\n", tl_error_msg);
}
