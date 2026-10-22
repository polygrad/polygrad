"""Arctangent values and differentiable range reductions against libm/NumPy."""
import numpy as np
import pytest
from polygrad import create

@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_atan_values_derivatives_and_reload(dtype):
    scalar=np.dtype(dtype).type
    # Neighbors must be computed in the tested dtype, before concatenation.
    boundaries=np.array([-1, -(np.sqrt(2)-1), -0., 0., np.sqrt(2)-1, 1],dtype=dtype)
    neighbors=np.r_[np.nextafter(boundaries,scalar(-np.inf)),boundaries,
                    np.nextafter(boundaries,scalar(np.inf))]
    values=np.r_[np.linspace(-4,4,129).astype(dtype),neighbors,
                 np.array([-1e-20,0,1e-20,-1e10,1e10,-1e20,1e20],dtype=dtype)]
    with create(device='CPU',logical='always') as rt:
        x=rt.Tensor.empty(len(values),dtype=dtype);y=x.atan();total=y.sum();g=total.gradient(x)[0]
        second_total=g.sum();gg=second_total.gradient(x)[0]
        model=rt.Model.from_tensors(inputs={'x':x},outputs={'y':y,'g':g,'gg':gg},entrypoints=[dict(name='atan',inputs=['x'],outputs=['y','g','gg'])])
        reloaded=rt.Model.load(model.save())
        try:
            z=values.astype(float);expected=np.arctan(z);grad=1/(1+z*z);grad2=-2*z*grad*grad
            tol=3e-6 if dtype=='float32' else 5e-13
            for m in (model,reloaded):
                out=m.call('atan',{'x':values})
                zeros=values==0
                np.testing.assert_array_equal(np.signbit(out['y'][zeros]),np.signbit(values[zeros]))
                np.testing.assert_array_equal(out['g'][zeros],np.ones(zeros.sum(),dtype=dtype))
                for name,want in [('y',expected),('g',grad),('gg',grad2)]:
                    np.testing.assert_allclose(out[name],want,rtol=tol,atol=2e-7 if dtype=='float32' else 2e-14)
        finally:
            reloaded.dispose();model.dispose()
            for t in (gg,second_total,g,total,y,x):t.dispose()

@pytest.mark.parametrize('dtype', ['float32','float64'])
def test_atan_extreme_values(dtype):
    maximum=np.finfo(dtype).max
    values=np.array([-maximum,-1e20,-0.,0.,1e20,maximum,-np.inf,np.inf,np.nan],dtype=dtype)
    with create(device='CPU',logical='always') as rt:
        x=rt.Tensor(values);y=x.atan()
        try:
            actual=y.numpy()
            np.testing.assert_array_equal(np.signbit(actual[2:4]),[True,False])
            np.testing.assert_allclose(actual,np.arctan(values.astype(float)),rtol=1e-7 if dtype=='float32' else 2e-16,atol=0)
        finally:y.dispose();x.dispose()
