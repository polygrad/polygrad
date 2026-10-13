"""Differential special-function values and two derivatives against SciPy.

Run with an explicitly rebuilt POLY_LIB and scipy installed; no skip fallback.
The seed and rounded inputs are fixed, including both sides of branch boundaries.
"""
import json
import math
import os

import numpy as np
from scipy import special
from polygrad import Runtime, Tensor, _ffi
from polygrad.uop.ops import UOp


def reference(name, x, order):
    if name == 'lgamma':
        return special.gammaln(x) if order == 0 else special.polygamma(order - 1, x)
    if name == 'digamma':
        return special.polygamma(order, x)
    y = getattr(special, name)(x)
    if order == 0:
        return y
    if name in ('erf', 'erfc'):
        first = (1 if name == 'erf' else -1) * 2 / math.sqrt(math.pi) * np.exp(-x*x)
        return first if order == 1 else -2*x*first
    if name == 'erfinv':
        first = math.sqrt(math.pi)/2 * np.exp(y*y)
        return first if order == 1 else 2*y*first*first
    first = math.sqrt(2*math.pi) * np.exp(y*y/2)
    return first if order == 1 else y*first*first


def main():
    rng = np.random.default_rng(20261013)
    grids = {
        'erf': np.r_[np.linspace(-9, 9, 129), [-1.000001, -1, -.999999, 0, .999999, 1, 1.000001]],
        'erfc': np.r_[np.linspace(-9, 9, 129), [0, .999999, 1, 1.000001, 7.999999, 8, 8.000001, 20, 26]],
        'erfinv': np.r_[rng.uniform(-.9999, .9999, 128), [-.99, -.5, -1e-20, 0, 1e-20, .5, .99]],
        'ndtri': np.r_[rng.uniform(.001, .999, 128), [1e-20, 1e-10, 1e-8, .13533528, .5, .86466472, .9999]],
        'lgamma': np.r_[rng.uniform(-20, 100, 128), [-6.5, -.5, -.0001, .0001, .499999, .5, .500001, 1, 2, 3.2, 6]],
        'digamma': np.r_[rng.uniform(-20, 100, 128), [-6.5, -.5, -.0001, .0001, .5, 1, 3.2, 5.9999, 6, 9.9999, 10]],
    }
    records = []
    with Runtime(device=os.environ.get('SPECIAL_DEVICE', 'CPU')) as rt:
        for dtype in ('float32', 'float64'):
            for name, values in grids.items():
                values = values.astype(dtype)
                # Tail second derivatives can exceed float32's range. Values
                # at these probabilities remain covered by the C/frontends.
                if dtype == 'float32' and name == 'ndtri':
                    values = values[values >= 1e-10]
                x = rt.Tensor(values).realize()
                y = Tensor(UOp(x._ctx, getattr(_ffi._lib, 'poly_uop_' + name)(x._ctx, x.uop.raw)), device=x.device)
                derivatives = [y]
                for _ in range(2):
                    derivatives.append(derivatives[-1].sum().gradient(x)[0])
                for order, result in enumerate(derivatives):
                    actual = result.numpy().astype('float64')
                    expected = reference(name, values.astype('float64'), order)
                    rtol, atol = ((8e-5, 5e-6) if dtype == 'float32' else (2e-11, 3e-12))
                    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                               err_msg=f'{name}/{dtype}/derivative={order}')
                    records.append(dict(function=name, dtype=dtype, derivative=order,
                                        samples=len(values), max_abs=float(np.max(np.abs(actual-expected))),
                                        max_scaled=float(np.max(np.abs(actual-expected)/(atol+rtol*np.abs(expected))))))
    print(json.dumps(dict(library=_ffi._lib._name, device=os.environ.get('SPECIAL_DEVICE', 'CPU'),
                          evaluations=sum(r['samples'] for r in records), results=records), indent=2))


if __name__ == '__main__':
    main()
