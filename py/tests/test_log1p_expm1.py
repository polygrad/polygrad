"""Cancellation, range boundaries and derivative contracts for small changes."""
import unittest
import numpy as np
from polygrad import create

class SmallChangeTests(unittest.TestCase):
 def test_values_derivatives_and_reload(self):
  for dtype in ('float32','float64'):
   dt=np.dtype(dtype).type
   values=list(np.r_[-np.geomspace(1e-12,.49,25),0,np.geomspace(1e-12,1,25)])
   values.extend([-.0001953125,.0001953125,-0.0])
   for boundary in (-.5,-1e-4,1e-4,.5,1):
    values.extend([np.nextafter(dt(boundary),dt(-np.inf)),dt(boundary),np.nextafter(dt(boundary),dt(np.inf))])
   data=np.asarray(values,dtype=dtype);ref=data.astype(np.float64)
   for name in ('log1p','expm1'):
    with self.subTest(dtype=dtype,name=name),create(device='CPU',logical='always') as rt:
     x=rt.Tensor.empty(len(data),dtype=dtype);y=getattr(x,name)();s=y.sum();d=s.gradient(x)[0];ds=d.sum();d2=ds.gradient(x)[0]
     model=rt.Model.from_tensors(inputs={'x':x},outputs={'y':y,'d1':d,'d2':d2})
     loaded=rt.Model.load(model.save())
     expected={'y':getattr(np,name)(ref),'d1':1/(1+ref) if name=='log1p' else np.exp(ref),'d2':-1/(1+ref)**2 if name=='log1p' else np.exp(ref)}
     try:
      for m in (model,loaded):
       got=m.call('forward',{'x':data})
       for key in expected:np.testing.assert_allclose(got[key],expected[key],rtol=2e-6 if dtype=='float32' else 3e-14,atol=0,err_msg=key)
       zeros=data==0;np.testing.assert_array_equal(np.signbit(got['y'][zeros]),np.signbit(data[zeros]))
     finally:
      loaded.dispose();model.dispose()
      for t in (x,y,s,d,ds,d2):t.dispose()

 def test_endpoints_and_large_arguments(self):
  for dtype in ('float32','float64'):
   for name,values in [('log1p',[-np.inf,-2,-1,-.99,-0.,0.,1,np.finfo(dtype).max,np.inf,np.nan]),('expm1',[-np.inf,-100,-0.,0.,1,20,np.inf,np.nan])]:
    data=np.asarray(values,dtype=dtype)
    with self.subTest(dtype=dtype,name=name),create(device='CPU') as rt:
     x=rt.Tensor(data);y=getattr(x,name)()
     try:
      with np.errstate(all='ignore'):expected=getattr(np,name)(data.astype(float))
      np.testing.assert_allclose(y.numpy(),expected,rtol=3e-6 if dtype=='float32' else 3e-14,atol=0)
     finally:y.dispose();x.dispose()

if __name__=='__main__':unittest.main()
