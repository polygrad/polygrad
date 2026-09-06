/* Configuration-driven Sequential and Graph model families. */
#ifndef POLY_MODELS_COMPOSE_H
#define POLY_MODELS_COMPOSE_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Build in a borrowed context, which must outlive the returned Model and allow
 * logical construction. JSON is borrowed during the call, never runtime state.
 * Optional format/type tags are checked when supplied; the factory selects the
 * family. Both return ordinary built Models, or NULL with a diagnostic. */
PolyModel *poly_sequential_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err);
PolyModel *poly_graph_from_json(PolyCtx *ctx, const char *json, int len, PolyModelError *err);

#ifdef __cplusplus
}
#endif
#endif
