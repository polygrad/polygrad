"""Scaled erfc in log space: independent values, derivatives and saved graphs."""
import unittest
import mpmath as mp
import numpy as np
from scipy import special
from polygrad import create

class LogErfcxTests(unittest.TestCase):
 def test_values_and_boundaries(self):
  for dtype in ('float32','float64'):
   dt=np.dtype(dtype).type
   values=[-np.inf,-1e10,-100,-8,-1,-.1,-0.,0.,.1,1,8,100,1e10,np.finfo(dtype).max,np.inf,np.nan]
   for b in (-1,1,8):values.extend([np.nextafter(dt(b),dt(-np.inf)),dt(b),np.nextafter(dt(b),dt(np.inf))])
   x=np.asarray(values,dtype=dtype);a=x.astype(float)
   with np.errstate(all='ignore'):
    expected=np.log(special.erfcx(a));negative=a<0
    expected[negative]=a[negative]*a[negative]+np.log(2)+np.log1p(-.5*special.erfc(-a[negative]))
   with self.subTest(dtype=dtype),create(device='CPU',logical='always') as rt:
    t=rt.Tensor(x);y=t.log_erfcx()
    try:np.testing.assert_allclose(y.numpy(),expected,rtol=4e-6 if dtype=='float32' else 4e-14,atol=3e-7 if dtype=='float32' else 3e-15)
    finally:y.dispose();t.dispose()

 def test_derivatives_and_reload(self):
  for dtype in ('float32','float64'):
   x=np.asarray([-100,-8,-1,-.1,0,.1,1,8,100,1e5],dtype=dtype)
   with mp.workdps(80):
    def f(t):return t*t+mp.log(mp.erfc(t))
    expected={k:np.asarray([float(mp.diff(f,mp.mpf(float(v)),i)) for v in x]) for i,k in enumerate(('y','d1','d2'))}
   with self.subTest(dtype=dtype),create(device='CPU',logical='always') as rt:
    t=rt.Tensor.empty(len(x),dtype=dtype);y=t.log_erfcx();total=y.sum();g=total.gradient(t)[0];gs=g.sum();h=gs.gradient(t)[0]
    model=rt.Model.from_tensors(inputs={'x':t},outputs={'y':y,'d1':g,'d2':h},entrypoints=[dict(name='eval',inputs=['x'],outputs=['y','d1','d2'])])
    loaded=rt.Model.load(model.save())
    try:
     for m in (model,loaded):
      got=m.call('eval',{'x':x})
      for key in expected:np.testing.assert_allclose(got[key],expected[key],rtol=7e-5 if dtype=='float32' else 4e-12,atol=3e-7 if dtype=='float32' else 3e-14,err_msg=key)
    finally:
     loaded.dispose();model.dispose()
     for v in (t,y,total,g,gs,h):v.dispose()

if __name__=='__main__':unittest.main()
