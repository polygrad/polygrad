"""Inverse log-Normal CDF: tails, dtype-local boundaries, autodiff and reload."""
import unittest
import numpy as np
from scipy import special
from polygrad import create

class NdtriExpTests(unittest.TestCase):
    def test_values_endpoints_and_boundaries(self):
        for dtype in ('float32', 'float64'):
            dt=np.dtype(dtype);f=np.finfo(dt)
            values=[-np.inf,-f.max,-1e20,-1000,-100,-2,-np.log(2),np.log(-np.expm1(-2)),
                    -.1,-1e-10,-f.tiny,-np.nextafter(dt.type(0),dt.type(1)),0.,.1,np.nan]
            for boundary in (-2,-np.log(2),np.log(-np.expm1(-2))):
                b=dt.type(boundary)
                values += [np.nextafter(b,dt.type(-np.inf)),b,np.nextafter(b,dt.type(np.inf))]
            x=np.array(values,dtype=dtype)
            with self.subTest(dtype=dtype),create(device='CPU',logical='always') as rt:
                t=rt.Tensor(x);out=t.ndtri_exp()
                try:np.testing.assert_allclose(out.numpy(),special.ndtri_exp(x.astype(float)),rtol=3e-6 if dtype=='float32' else 4e-14,atol=3e-7 if dtype=='float32' else 2e-15)
                finally:out.dispose();t.dispose()

    def test_gradients_and_reload(self):
        for dtype in ('float32','float64'):
            x=np.array([-1000.,-100.,-2.,-np.log(2),-.1,-1e-5],dtype=dtype)
            with self.subTest(dtype=dtype),create(device='CPU',logical='always') as rt:
                t=rt.Tensor.empty(len(x),dtype=dtype);y=t.ndtri_exp();total=y.sum();g=total.gradient(t)[0];gt=g.sum();h=gt.gradient(t)[0]
                m=rt.Model.from_tensors(inputs={'x':t},outputs={'y':y,'g':g,'h':h},entrypoints=[dict(name='eval',inputs=['x'],outputs=['y','g','h'])])
                try:
                    z=special.ndtri_exp(x.astype(float));first=np.exp(x.astype(float)+.5*z*z+.5*np.log(2*np.pi));second=first*(1+z*first)
                    for model in (m,rt.Model.load(m.save())):
                        try:
                            actual=model.call('eval',{'x':x})
                            np.testing.assert_allclose(actual['y'],z,rtol=3e-6 if dtype=='float32' else 4e-14,atol=3e-7 if dtype=='float32' else 2e-15)
                            np.testing.assert_allclose(actual['g'],first,rtol=3e-5 if dtype=='float32' else 3e-11)
                            np.testing.assert_allclose(actual['h'],second,rtol=3e-3 if dtype=='float32' else 3e-7,atol=1e-10)
                        finally:
                            if model is not m:model.dispose()
                finally:
                    m.dispose()
                    for v in (t,y,total,g,gt,h):v.dispose()

if __name__=='__main__':unittest.main()
