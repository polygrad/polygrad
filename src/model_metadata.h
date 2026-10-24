#ifndef POLY_MODEL_METADATA_H
#define POLY_MODEL_METADATA_H

#include <stddef.h>

struct cJSON;
/* Internal artifact policy. Length excludes the terminator; the caller owns
 * the returned tree. Reject values cJSON would normalize lossily/ambiguously. */
struct cJSON *model_parse_metadata_json(const char *json, size_t length);

#endif
