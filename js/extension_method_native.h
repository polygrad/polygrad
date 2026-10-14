/* Internal method wire adapter. Generated C performs typed calls; JS owns
 * runtime/object admission. Native pointers cross N-API only as externals. */
#include <string.h>
#include <stdio.h>
static napi_value extension_invoke(napi_env env, napi_callback_info info) {
  napi_value args[5], v, result = NULL;
  size_t argc = 5;
  int32_t method = -1;
  PolyCtx *ctx = NULL;
  uint32_t n = 0, ns = 0, nv = 0;
  void *handles[64] = {0}, *out = NULL;
  char *strings[64] = {0}, *text = NULL;
  double values[64] = {0}, value = 0;
  int rc = 0;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 5 ||
      napi_get_value_external(env, args[0], (void **)&ctx) != napi_ok || !ctx ||
      napi_get_value_int32(env, args[1], &method) != napi_ok ||
      napi_get_array_length(env, args[2], &n) != napi_ok || n > 64 ||
      napi_get_array_length(env, args[3], &ns) != napi_ok || ns != n ||
      napi_get_array_length(env, args[4], &nv) != napi_ok || nv != n)
    goto failed;
  for (uint32_t i = 0; i < n; i++) {
    napi_valuetype type;
    if (napi_get_element(env, args[2], i, &v) != napi_ok || napi_typeof(env, v, &type) != napi_ok)
      goto failed;
    if (type != napi_null && napi_get_value_external(env, v, &handles[i]) != napi_ok) goto failed;
    if (napi_get_element(env, args[3], i, &v) != napi_ok || napi_typeof(env, v, &type) != napi_ok)
      goto failed;
    if (type != napi_null) {
      size_t len;
      if (napi_get_value_string_utf8(env, v, NULL, 0, &len) != napi_ok ||
          !(strings[i] = malloc(len + 1)) ||
          napi_get_value_string_utf8(env, v, strings[i], len + 1, NULL) != napi_ok ||
          strlen(strings[i]) != len)
        goto failed;
    }
    if (napi_get_element(env, args[4], i, &v) != napi_ok ||
        napi_get_value_double(env, v, &values[i]) != napi_ok)
      goto failed;
  }
  rc = poly_extension_invoke(
      method, ctx, handles, (const char *const *)strings, values, &out, &text, &value
  );
  if (rc) goto failed;
  if (napi_create_object(env, &result) != napi_ok) goto failed;
  if (out) {
    if (napi_create_external(env, out, NULL, NULL, &v) != napi_ok ||
        napi_set_named_property(env, result, "handle", v) != napi_ok)
      goto failed;
  }
  if (text) {
    if (napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &v) != napi_ok ||
        napi_set_named_property(env, result, "text", v) != napi_ok)
      goto failed;
  }
  if (napi_create_double(env, value, &v) != napi_ok ||
      napi_set_named_property(env, result, "value", v) != napi_ok)
    goto failed;
  out = NULL;
  goto cleanup;
failed:
  if (out) poly_extension_release(method, out);
  {
    char message[80];
    snprintf(message, sizeof(message), "extension method %d failed (%d)", method, rc);
    napi_throw_error(env, NULL, message);
  }
  result = NULL;
cleanup:
  for (uint32_t i = 0; i < 64; i++)
    free(strings[i]);
  free(text);
  return result;
}

static napi_value extension_release(napi_env env, napi_callback_info info) {
  napi_value args[2], result;
  size_t argc = 2;
  int32_t method;
  void *handle = NULL;
  if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc != 2 ||
      napi_get_value_int32(env, args[0], &method) != napi_ok ||
      napi_get_value_external(env, args[1], &handle) != napi_ok) {
    napi_throw_type_error(env, NULL, "invalid extension object");
    return NULL;
  }
  poly_extension_release(method, handle);
  napi_get_undefined(env, &result);
  return result;
}
