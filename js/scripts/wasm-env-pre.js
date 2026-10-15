// Emscripten materializes libc environ on its first use. Set ENV in preRun,
// before C initialization, rather than changing it after creating a context.
Module['preRun'] = Module['preRun'] || [];
Module['preRun'].push(function () {
  var inherited = typeof process !== 'undefined' && process.versions && process.versions.node ? process.env : {};
  var supplied = Module['polygradEnv'] || globalThis[Symbol.for('polygrad.wasm.env')];
  ENV = Object.assign(Object.create(null), ENV, supplied ? supplied.values : inherited);
  // libc needs the inherited environment; conflict checking does not. Keep
  // only caller-supplied keys and discard the transient factory argument.
  Module['__polygradExplicitEnv'] = supplied ? supplied.explicit : Object.create(null);
  delete Module['polygradEnv'];
});
