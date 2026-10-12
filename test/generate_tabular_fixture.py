"""Small official TabM oracle; masks come from pinned Tinygrad's concrete RNG.

Compare C's compiled symbolic random program with concrete reference draws,
then Torch autograd and optimizer updates using those same dropout masks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--tabm', type=Path, required=True)
parser.add_argument('--embeddings', type=Path, required=True)
args = parser.parse_args()
for path, expected in ((args.tabm/'tabm_reference_pinned.py', 'fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade'),
                       (args.embeddings/'rtdl_num_embeddings.py', '48dfc019d447e0cd225deb8b8145b3efa8b42cde9aa314772d6e234753e1cee1')):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'reference source lock mismatch: {path.name}')
sys.path[:0] = [str(args.tabm), str(args.embeddings), str(Path(__file__).resolve().parents[1]/'references/tinygrad_latest')]
import tabm_reference_pinned as tabm
import rtdl_num_embeddings as embeddings
from tinygrad import Tensor

torch.set_num_threads(1)
rows = []
for kind in ('none', 'linear_relu', 'piecewise_linear'):
    spec = dict(layers=[2,4,4,1], n_ensemble=2, batch_size=4, seed=321, dropout=.5, loss='mse')
    embedding = None
    if kind == 'linear_relu':
        spec['num_embeddings'] = dict(type=kind, d_embedding=3)
        embedding = embeddings.LinearReLUEmbeddings(2, 3)
    elif kind == 'piecewise_linear':
        bins = [[-1.,0.,2.], [0.,2.]]
        spec['num_embeddings'] = dict(type=kind, d_embedding=3, bins=bins)
        embedding = embeddings.PiecewiseLinearEmbeddings([torch.tensor(b) for b in bins],3,activation=False,version='B')
    ref = tabm.TabM.make(n_num_features=2, cat_cardinalities=[], d_out=1, num_embeddings=embedding,
                         n_blocks=2,d_block=4,k=2,dropout=.5)
    params = {}
    for name, p in ref.named_parameters():
        if name.startswith('backbone.blocks.'):
            _,_,layer,_,field = name.split('.')
            name = f'layers.{layer}.'+('b' if field == 'bias' else field)
        with torch.no_grad(): p.copy_((torch.arange(p.numel()).reshape(p.shape)%13-4)*.025)
        params[name] = dict(shape=list(p.shape), values=p.detach().flatten().tolist())
    row = dict(config=spec, params=params, steps=[])
    optimizer = torch.optim.AdamW(ref.parameters(),lr=.01,betas=(.8,.9),eps=1e-6,weight_decay=.03)
    Tensor.manual_seed(321)
    class Dropout(torch.nn.Module):
        def forward(self, x):
            if not self.training: return x
            mask = (Tensor.rand(*x.shape, device='CPU').numpy() >= .5).astype(np.float32)*2
            return x*torch.tensor(mask)
    for block in ref.backbone.blocks: block[2] = Dropout()
    for size in (4,1,3):
        x = torch.tensor([[-2.,-.5],[.3,1.],[3.,4.],[.7,.3]])[:size]
        y = torch.ones(size,1)*.15
        ref.eval()
        prediction = ref(x).detach()
        ref.train()
        optimizer.zero_grad()
        loss = (ref(x)-y[:,None,:]).square().mean()
        loss.backward()
        optimizer.step()
        updated = {}
        for name, p in ref.named_parameters():
            key = name
            if name.startswith('backbone.blocks.'):
                _,_,layer,_,field = name.split('.')
                key = f'layers.{layer}.'+('b' if field == 'bias' else field)
            updated[key] = p.detach().flatten().tolist()
        row['steps'].append(dict(x=x.tolist(),y=y.tolist(),members=prediction.flatten().tolist(),
                                 loss=loss.item(),updated=updated))
    rows.append(row)
print(json.dumps(dict(reference='yandex-research/tabm@28e47ae301c92ec37787dde1ce923a0793f405b4',
                     embeddings='rtdl-num-embeddings 0.0.12',cases=rows),separators=(',',':')))
