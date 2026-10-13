"""Actual Transformers delta functions, not a hand-written attention oracle.

Run with the HF reference interpreter; dependencies are required, never skipped.
Full Perplexity pipeline validation remains separate from this layer check.
"""
import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
)
import polygrad as pg


@pytest.mark.parametrize('length,chunk',[(1,1),(7,8),(65,64),(129,64)])
def test_hf_delta_prefill_matches_recurrent(length,chunk):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    values = [torch.randn(1,length,2,8) for _ in range(3)]
    values += [-torch.rand(1,length,2),torch.rand(1,length,2),torch.randn(1,2,8,8)]
    kwargs = dict(initial_state=values[-1],output_final_state=True,use_qk_l2norm_in_kernel=True)
    expected = torch_chunk_gated_delta_rule(*values[:5],chunk_size=chunk,**kwargs)
    recurrent = torch_recurrent_gated_delta_rule(*values[:5],**kwargs)
    for x,y in zip(expected,recurrent):
        torch.testing.assert_close(x,y,atol=3e-5,rtol=3e-4)
    with pg.create(device='CPU') as rt:
        inputs = [rt.Tensor(x.numpy()).realize() for x in values]
        actual = rt.nn.gated_delta_rule(*inputs,chunk_size=chunk)
        for x,y in zip(actual,expected):
            np.testing.assert_allclose(x.numpy(),y.numpy(),atol=3e-5,rtol=3e-4)


def test_hf_delta_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(61)
    values = [torch.randn(1,3,1,2) for _ in range(3)]
    values += [-torch.rand(1,3,1),torch.rand(1,3,1),torch.randn(1,1,2,2)]
    for x in values: x.requires_grad_(True)
    output,state = torch_recurrent_gated_delta_rule(*values[:5],initial_state=values[-1],
        output_final_state=True,use_qk_l2norm_in_kernel=True)
    (output.square().sum()+state.square().sum()).backward()
    with pg.create(device='CPU') as rt:
        inputs = [rt.Tensor(x.detach().numpy()).is_param_(True).realize() for x in values]
        output,state = rt.nn.gated_delta_rule(*inputs,chunk_size=4)
        (output.square().sum()+state.square().sum()).backward()
        for x,y in zip(inputs,values):
            np.testing.assert_allclose(x.grad.numpy(),y.grad.numpy(),atol=3e-5,rtol=5e-4)
