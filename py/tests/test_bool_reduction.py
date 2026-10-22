"""Broadcast Boolean reductions preserve dtype and the empty identity."""
import unittest
import numpy as np
from polygrad import create

class BooleanReductionTests(unittest.TestCase):
    def test_broadcast_all_and_any(self):
        with create(device='CPU', logical='always') as rt:
            for repeat in (1, 2, 3, 7, 32):
                for values in ([1., -1., 1., -1.], [-1.] * 4, [1.] * 4):
                    a = rt.Tensor(values).reshape(2, 2, 1).expand(2, 2, repeat) > 0
                    b = rt.Tensor([[True, False], [True, True]])
                    for op in ('all', 'any'):
                        expected = getattr(np.broadcast_to(np.array(values).reshape(2,2,1), (2,2,repeat)) > 0, op)(axis=2) & np.array([[True,False],[True,True]])
                        out = getattr(a,op)(axis=2) & b
                        self.assertEqual(out.dtype.name, 'bool')
                        np.testing.assert_array_equal(out.numpy(), expected)

if __name__ == '__main__': unittest.main()
