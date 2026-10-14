"""Consumer method ownership rules; C declarations remain the signature source.

Generated methods.c includes the author source exactly once. Compile it instead
of the original source, alongside extension.c. The wire adapter is internal:
frontends expose owned objects/Tensors, never pointer-valued numbers.
"""
import json


def describe(rules, decls, signature):
    methods, roots, objects = [], [], {}
    if not isinstance(rules, dict) or not rules or len(rules) > 256:
        raise ValueError('methods must be a nonempty object with at most 256 entries')
    for name, rule in rules.items():
        if name not in decls or not isinstance(rule, dict) or set(rule) - {'args', 'return'}:
            raise ValueError(f'{name}: invalid method declaration/rules')
        ret, params = signature(decls[name])
        annotations = rule.get('args', {})
        if set(annotations) - {p['name'] for p in params}:
            raise ValueError(f'{name}: unknown method argument')
        args = []
        for p in params:
            t, n = p['type']['qualType'], p['name']
            r = annotations.get(n)
            if t == 'PolyCtx *' and r is None:
                kind = {'kind': 'context'}
            elif t in ('int', 'double') and r is None:
                kind = {'kind': t}
            elif t == 'PolyTensor *' and r == {'tensor': 'borrowed'}:
                kind = {'kind': 'tensor'}
            elif t == 'const char *' and r == {'string': 'borrowed'}:
                kind = {'kind': 'string'}
            elif isinstance(r, dict) and set(r) == {'object'} and t.endswith(' *') and t not in ('PolyTensor *', 'PolyCtx *'):
                if not isinstance(r['object'], str) or not r['object']:
                    raise ValueError(f'{name}: invalid object type')
                kind = {'kind': 'object', 'object': r['object']}
            else:
                raise ValueError(f'{name}: unsupported or unannotated argument {n}: {t}')
            args.append(dict(kind, name=n, ctype=t))
        if len(args) > 64:
            raise ValueError(f'{name}: too many arguments')
        r = rule.get('return')
        if r in ('value', 'status') and ret in ('int', 'double') and (r != 'status' or ret == 'int'):
            result = {'kind': r, 'ctype': ret}
        elif r == 'void' and ret == 'void':
            result = {'kind': 'void'}
        elif r == {'tensor': 'owned'} and ret == 'PolyTensor *':
            result = {'kind': 'tensor'}
        elif isinstance(r, dict) and r.get('string') in ('borrowed', 'owned') and ret in ('char *', 'const char *'):
            if set(r) != ({'string', 'release'} if r['string'] == 'owned' else {'string'}):
                raise ValueError(f'{name}: string ownership requires an explicit release')
            result = {'kind': 'string', 'ownership': r['string']}
            if 'release' in r: result['release'] = r['release']
        elif isinstance(r, dict) and set(r) == {'object', 'release'} and ret.endswith(' *') and ret not in ('PolyCtx *', 'PolyTensor *'):
            if not isinstance(r['object'], str) or not r['object']:
                raise ValueError(f'{name}: invalid object type')
            result = {'kind': 'object', **r}
            contract = (ret, r['release'])
            if r['object'] in objects and objects[r['object']] != contract:
                raise ValueError(f'{name}: conflicting object contract')
            objects[r['object']] = contract
        else:
            raise ValueError(f'{name}: unsupported or unannotated return {ret}')
        if 'release' in result:
            release = result['release']
            if release not in decls:
                raise ValueError(f'{name}: missing release declaration')
            rr, rp = signature(decls[release])
            if rr != 'void' or len(rp) != 1 or rp[0]['type']['qualType'] != ret:
                raise ValueError(f'{name}: release must take the returned pointer and return void')
            roots.append(release)
        methods.append({'name': name, 'args': args, 'result': result})
        roots.append(name)
    for m in methods:
        for a in m['args']:
            if a['kind'] == 'object' and (a['object'] not in objects or objects[a['object']][0] != a['ctype']):
                raise ValueError(f"{m['name']}: object argument has no matching constructor")
    return methods, roots


PROTOTYPES = '''
EXT int poly_extension_invoke(int method, PolyCtx *ctx, void *const *handles,
    const char *const *strings, const double *values, void **out, char **text, double *value);
EXT void poly_extension_release(int method, void *handle);
'''


def generate(source, methods, export_definitions):
    c = ['/* Generated: compile this instead of the author source. */',
         '#include <stdlib.h>', '#include <string.h>', '#include <math.h>',
         f'#include {json.dumps(str(source))}', *export_definitions,
         PROTOTYPES,
         'EXT void poly_extension_free(void *ptr) { free(ptr); }',
         'EXT void poly_extension_release(int method, void *handle) {',
         'if (!handle) return; switch (method) {']
    for i, m in enumerate(methods):
        r = m['result']
        if r['kind'] in ('object', 'tensor'):
            release = r.get('release', 'poly_tensor_release')
            c.append(f'case {i}: {release}(handle); break;')
    c += ['default: break; }}',
          'EXT int poly_extension_invoke(int method, PolyCtx *ctx, void *const *handles,',
          'const char *const *strings, const double *values, void **out, char **text, double *value) {',
          '*out = NULL; *text = NULL; *value = 0; switch (method) {']
    for i, m in enumerate(methods):
        args = []
        c.append(f'case {i}: {{')
        for j, a in enumerate(m['args']):
            k = a['kind']
            if k == 'context': args.append('ctx')
            elif k in ('object', 'tensor'):
                c.append(f'if (!handles[{j}]) return -1;')
                args.append(f'({a["ctype"]})handles[{j}]')
            elif k == 'string':
                c.append(f'if (!strings[{j}]) return -1;')
                args.append(f'strings[{j}]')
            else:
                c.append(f'if (!isfinite(values[{j}])) return -1;')
                if k == 'int':
                    c.append(f'if (values[{j}] < -2147483648.0 || values[{j}] > 2147483647.0 || values[{j}] != (int)values[{j}]) return -1;')
                args.append(f'({k})values[{j}]')
        call = f'{m["name"]}({", ".join(args)})'
        r = m['result']
        k = r['kind']
        if k in ('object', 'tensor'):
            c.append(f'*out = {call}; return *out ? 0 : -1;')
        elif k == 'string':
            c += [f'const char *s = {call};', 'if (!s) return -1;',
                  'size_t n = strlen(s) + 1; *text = malloc(n);',
                  'if (*text) memcpy(*text, s, n);']
            if 'release' in r: c.append(f'{r["release"]}((char *)s);')
            c.append('return *text ? 0 : -1;')
        elif k == 'status': c.append(f'return {call};')
        elif k == 'value': c.append(f'*value = {call}; return 0;')
        else: c.append(f'{call}; return 0;')
        c.append('}')
    c += ['default: return -1; }}']
    return '\n'.join(c) + '\n'
