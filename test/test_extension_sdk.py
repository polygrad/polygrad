#!/usr/bin/env python3
"""Build and load authors using installed packages only, outside the checkout.

The caller supplies a copied author source, installed Python or npm package,
and an empty working directory. Compilers are explicit test prerequisites.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def run(args, **kwargs):
    print('RUN', shlex.join(map(str, args)), flush=True)
    return subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['python', 'native', 'wasm'])
    parser.add_argument('--node', default='node')
    parser.add_argument('--node-include', default=os.environ.get('NODE_INCLUDE', '/usr/include/node'))
    args = parser.parse_args()
    if args.kind == 'python':
        import polygrad
        sdk = Path(polygrad.__file__).parent / '_sdk'
        assert sdk.resolve().is_relative_to(Path(sys.prefix).resolve()), sdk
        generator = [sys.executable, '-I', '-m', 'polygrad.sdk']
    else:
        sdk = Path.cwd() / 'node_modules/polygrad/sdk'
        generator = [sys.executable, str(sdk / 'extension_bindings.py')]
    include = sdk / 'include'
    assert include.is_dir(), f'installed SDK missing: {include}'
    run([*generator, '--source', 'author.c', '--entry', 'macro_build',
         '--inputs', '1', '--outputs', '1', '--output', 'generated'])
    source = Path('generated/extension.c').read_text()
    manifest = json.loads(Path('generated/manifest.json').read_text())
    assert manifest['construction_abi'] == 1
    # Freeze the old handshake in a source fixture, not a checked-in binary.
    # The legacy lane uses the previous global-ABI resolver and no new field.
    legacy_manifest = dict(manifest)
    del legacy_manifest['construction_abi']
    legacy = source.replace(json.dumps(json.dumps(manifest)), json.dumps(json.dumps(legacy_manifest)))
    assert legacy != source
    legacy = legacy.replace('poly_extension_abi_version', 'poly_abi_version')
    legacy = legacy.replace('POLYGRAD_EXTENSION_ABI_VERSION', 'POLYGRAD_ABI_VERSION')
    variants = {
        'current': source,
        'compatible': source.replace('return POLYGRAD_ABI_VERSION;', 'return -1;'),
        'legacy': legacy,
        'bad-legacy': legacy.replace('return POLYGRAD_ABI_VERSION;', 'return -1;'),
    }
    for name, value in [('bad-contract', 2), ('bad-contract-type', True)]:
        spec = dict(manifest, construction_abi=value)
        variants[name] = source.replace(json.dumps(json.dumps(manifest)), json.dumps(json.dumps(spec)))
    spec = dict(manifest, imports=[*manifest['imports'], 'poly_missing'])
    variants['missing-import'] = source.replace(json.dumps(json.dumps(manifest)), json.dumps(json.dumps(spec)))
    # A declared list must include the module's actual imports on Wasm.
    spec = dict(manifest, imports=[n for n in manifest['imports'] if n != 'poly_tensor_exp'])
    if args.kind == 'wasm':
        variants['undeclared-import'] = source.replace(json.dumps(json.dumps(manifest)), json.dumps(json.dumps(spec)))
    for name, text in variants.items():
        path = Path('generated') / (name + '.c')
        path.write_text(text)
        if args.kind == 'wasm':
            command = [*shlex.split(os.environ.get('EMCC', 'emcc')), '-O2', '-I' + str(include),
                       str(path), 'author.c', '-sSTANDALONE_WASM=1', '-sALLOW_MEMORY_GROWTH=1',
                       '--no-entry', '-Wl,--allow-undefined', '-Wl,--export=poly_extension_build',
                       '-Wl,--export=poly_extension_manifest', '-Wl,--export=poly_extension_abi',
                       '-Wl,--export=malloc', '-Wl,--export=free', '-o', name + '.wasm']
        else:
            suffix = '.so' if args.kind == 'python' else '.node'
            command = [*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O2', '-shared', '-fPIC',
                       '-I' + str(include), str(path), 'author.c', '-lm', '-o', name + suffix]
            if args.kind == 'native':
                command += ['-DPOLY_EXTENSION_NAPI', '-I' + args.node_include]
        run(command)
    if args.kind == 'python':
        import numpy as np
        with polygrad.create(device='CPU') as rt:
            x = rt.Tensor([0., 1.])
            try:
                for name in variants:
                    if name.startswith('bad-') or name == 'missing-import':
                        try:
                            rt.load_extension(str(Path(name + '.so').resolve()))
                        except (ValueError, RuntimeError) as error:
                            assert 'ABI' in str(error) or 'manifest' in str(error), error
                        else:
                            raise AssertionError(f'{name}: incompatible author loaded')
                    else:
                        author = rt.load_extension(str(Path(name + '.so').resolve()))
                        y, = author.build([x])
                        author.dispose()
                        try:
                            np.testing.assert_allclose(y.numpy(), [1., np.e], rtol=1e-6)
                        finally:
                            y.dispose()
            finally:
                x.dispose()
    else:
        run([args.node, '-e', '''
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path')
const pg = require(path.resolve('node_modules/polygrad'))
;(async () => {
  const kind = process.argv[1], names = JSON.parse(process.argv[2])
  const rt = await pg.create({core:kind==='wasm'?'wasm':'native',device:'cpu'})
  const x = new rt.Tensor([0,1])
  try {
    for (const name of names) {
      const source = kind==='wasm'?fs.readFileSync(name+'.wasm'):require(path.resolve(name+'.node'))
      if (name.startsWith('bad-') || name.endsWith('import')) {
        await assert.rejects(() => rt.loadExtension(source), /ABI|manifest|import/)
      } else {
        const author = await rt.loadExtension(source), [y] = author.build([x])
        author.dispose()
        try { const a = await y.toArrayAsync(); assert.equal(a[0],1); assert.ok(Math.abs(a[1]-Math.E)<1e-6) }
        finally { y.dispose() }
      }
    }
  } finally { x.dispose(); await rt.dispose() }
})().catch(e=>{console.error(e);process.exitCode=1})
''', args.kind, json.dumps(list(variants))])
    print(f'PASS: installed {args.kind} SDK, {len(variants)} compatibility cases', flush=True)
    shutil.copyfile(sdk / 'examples/author.c', 'extension_methods_fixture.c')
    shutil.copyfile(sdk / 'examples/methods.json', 'extension_methods_fixture.json')
    run([*generator, '--source', 'extension_methods_fixture.c', '--entry', 'example_build',
         '--inputs', '1', '--outputs', '1', '--methods', 'extension_methods_fixture.json',
         '--output', 'methods-generated'])
    sources = ['methods-generated/extension.c', 'methods-generated/methods.c']
    if args.kind == 'wasm':
        command = [*shlex.split(os.environ.get('EMCC', 'emcc')), '-O2', '-I' + str(include),
                   *sources, '-sSTANDALONE_WASM=1', '-sALLOW_MEMORY_GROWTH=1', '--no-entry', '-Wl,--allow-undefined']
        command += ['-Wl,--export=' + name for name in ('poly_extension_build', 'poly_extension_manifest',
            'poly_extension_abi', 'poly_extension_invoke', 'poly_extension_release', 'malloc', 'free')]
        command += ['-o', 'methods.wasm']
    else:
        command = [*shlex.split(os.environ.get('CC', 'cc')), '-std=c11', '-O2', '-shared', '-fPIC',
                   '-I' + str(include), *sources, '-lm', '-o', 'methods.so' if args.kind == 'python' else 'methods.node']
        if args.kind == 'native': command += ['-DPOLY_EXTENSION_NAPI', '-I' + args.node_include]
    run(command)
    if args.kind == 'python':
        with polygrad.create(device='CPU', logical='always') as rt:
            author = rt.load_extension(str(Path('methods.so').resolve()))
            x = rt.Tensor([1.])
            obj = author.methods.example_new(x, 'μ')
            x.dispose(); rt.collect()
            assert author.methods.example_name(obj) == 'μ'
            y = author.methods.example_finish(obj)
            parameter = author.methods.example_input(obj)
            model = rt.Model.from_tensors(params={'x': parameter}, outputs={'y': y})
            parameter.dispose()
            assert author.methods.example_model_buffers(model) == model.buf_count
            model.set_metadata('example', {'schema': 1})
            y.dispose(); author.dispose()
            np.testing.assert_allclose(model.call('forward', {})['y'], [np.e], rtol=1e-6)
            restored = rt.Model.load(model.save())
            assert restored.get_metadata('example') == {'schema': 1}
            restored.dispose(); model.dispose()
    else:
        run([args.node, '-e', '''
const pg=require(require('path').resolve('node_modules/polygrad')),fs=require('fs'),path=require('path')
;(async()=>{
 const core=process.argv[1],rt=await pg.create({core,device:'cpu',logical:'always'})
 try { await require('./check_extension_methods')(rt,core==='native'?require(path.resolve('methods.node')):fs.readFileSync('methods.wasm'),pg.create) }
 finally { await rt.dispose() }
})().catch(e=>{console.error(e);process.exitCode=1})
''', args.kind])
    print(f'PASS: installed {args.kind} stateful methods, ownership and metadata', flush=True)


if __name__ == '__main__':
    main()
