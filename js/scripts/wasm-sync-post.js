// Emscripten only assigns module.exports on its Node environment branch for
// non-MODULARIZE builds. Browser bundles wrap this file as CommonJS while the
// generated runtime detects ENVIRONMENT_IS_WEB, so export the ready Module
// after synchronous startup in both environments.
if (typeof module !== 'undefined' && module.exports !== Module) {
  module.exports = Module
}
