/*
 * uop_cache_internal.h — Private accessor into PolyUOpCache
 *
 * PolyUOpCache is opaque in polygrad.h. The cache lives in src/uop.c
 * alongside the range helpers that populate cache->ranges. The minmax
 * helpers in src/sym.c need to reach cache->minmax without exposing
 * the struct layout to the world. This header is the sanctioned bridge.
 *
 * Only src/uop.c (definition) and src/sym.c (consumer) include this.
 */
#ifndef POLY_UOP_CACHE_INTERNAL_H
#define POLY_UOP_CACHE_INTERNAL_H

#include "polygrad.h"

PolyMap *poly_uop_cache_minmax_map(PolyUOpCache *c);

#endif
