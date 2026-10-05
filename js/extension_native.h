/* Generated consumer adapters include this once. No Polygrad implementation is
 * linked here: addresses come from the frontend's loaded core. */
#include <node_api.h>
#include <stdlib.h>

static napi_value extension_bind(napi_env env, napi_callback_info info) {
  napi_value arg, out;
  size_t n = 1;
  const PolyProcResolver *resolver = NULL;
  napi_get_cb_info(env, info, &n, &arg, NULL, NULL);
  if (n != 1 || napi_get_value_external(env, arg, (void **)&resolver) != napi_ok || !resolver) {
    napi_throw_type_error(env, NULL, "invalid Polygrad resolver");
    return NULL;
  }
  napi_get_boolean(env, poly_extension_bind(*resolver) != 0, &out);
  return out;
}
static napi_value extension_manifest(napi_env env, napi_callback_info info) {
  (void)info;
  napi_value out;
  napi_create_string_utf8(env, poly_extension_manifest(), NAPI_AUTO_LENGTH, &out);
  return out;
}
static napi_value extension_abi(napi_env env, napi_callback_info info) {
  (void)info;
  napi_value out;
  napi_create_int32(env, poly_extension_abi(), &out);
  return out;
}
static napi_value extension_build(napi_env env, napi_callback_info info) {
  size_t n = 3;
  napi_value args[3], v, result;
  PolyCtx *ctx = NULL;
  uint32_t ni = 0, ns = 0;
  PolyTensor *inputs[POLY_EXTENSION_NINPUTS + 1] = {0}, *outputs[POLY_EXTENSION_NOUTPUTS + 1] = {0};
  double scalars[POLY_EXTENSION_NSCALARS + 1] = {0};
  napi_get_cb_info(env, info, &n, args, NULL, NULL);
  if (n != 3 || napi_get_value_external(env, args[0], (void **)&ctx) != napi_ok || !ctx ||
      napi_get_array_length(env, args[1], &ni) != napi_ok || ni != POLY_EXTENSION_NINPUTS ||
      napi_get_array_length(env, args[2], &ns) != napi_ok || ns != POLY_EXTENSION_NSCALARS)
    goto invalid;
  for (uint32_t i = 0; i < ni; i++) {
    if (napi_get_element(env, args[1], i, &v) != napi_ok ||
        napi_get_value_external(env, v, (void **)&inputs[i]) != napi_ok || !inputs[i])
      goto invalid;
  }
  for (uint32_t i = 0; i < ns; i++) {
    if (napi_get_element(env, args[2], i, &v) != napi_ok ||
        napi_get_value_double(env, v, &scalars[i]) != napi_ok)
      goto invalid;
  }
  if (!poly_extension_build(ctx, inputs, scalars, outputs)) goto failed;
  if (napi_create_array_with_length(env, POLY_EXTENSION_NOUTPUTS, &result) != napi_ok) goto failed;
  for (int i = 0; i < POLY_EXTENSION_NOUTPUTS; i++) {
    if (!outputs[i] || napi_create_external(env, outputs[i], NULL, NULL, &v) != napi_ok ||
        napi_set_element(env, result, i, v) != napi_ok)
      goto failed;
  }
  return result;
failed:
  for (int i = 0; i < POLY_EXTENSION_NOUTPUTS; i++)
    poly_tensor_release(outputs[i]);
  napi_throw_error(env, NULL, "extension construction failed");
  return NULL;
invalid:
  napi_throw_type_error(env, NULL, "invalid extension arguments");
  return NULL;
}
static napi_value extension_init(napi_env env, napi_value exports) {
  napi_property_descriptor methods[] = {
      {"bind", 0, extension_bind, 0, 0, 0, napi_default, 0},
      {"manifest", 0, extension_manifest, 0, 0, 0, napi_default, 0},
      {"abi", 0, extension_abi, 0, 0, 0, napi_default, 0},
      {"build", 0, extension_build, 0, 0, 0, napi_default, 0}};
  napi_define_properties(env, exports, 4, methods);
  return exports;
}
NAPI_MODULE(NODE_GYP_MODULE_NAME, extension_init)
