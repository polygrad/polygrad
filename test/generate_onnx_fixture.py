"""Small standard ONNX graphs, checked by ONNX Runtime and pinned Tinygrad."""
import base64
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'references/tinygrad_014'))
from tinygrad.nn.onnx import OnnxRunner


def case(name, nodes, inputs, outputs, weights, *, dims=None, external=False, opset=17, domains=None):
    graph = h.make_graph(nodes, name,
        [h.make_tensor_value_info(k, T.FLOAT, (['batch'] + list(v.shape[1:])) if dims else v.shape)
         for k, v in inputs.items()],
        [h.make_tensor_value_info(k, T.FLOAT, shape) for k, shape in outputs.items()],
        [nh.from_array(v, k) for k, v in weights.items()])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', opset)] +
                         [h.make_opsetid(k,v) for k,v in (domains or {}).items()], ir_version=8)
    onnx.checker.check_model(model)
    raw = model.SerializeToString()
    expected = ort.InferenceSession(raw, providers=['CPUExecutionProvider']).run(list(outputs), inputs)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'model.onnx'
        path.write_bytes(raw)
        actual = OnnxRunner(path)(inputs)
        for key, ref in zip(outputs, expected):
            np.testing.assert_allclose(actual[key].numpy(), ref, atol=2e-5, rtol=2e-5)
    external_data = {}
    if external:
        payload = bytearray(b'offset-prefix')
        for tensor in model.graph.initializer:
            data = tensor.raw_data
            onnx.external_data_helper.set_external_data(tensor, 'weights.bin', offset=len(payload), length=len(data))
            tensor.ClearField('raw_data')
            payload.extend(data)
        raw = model.SerializeToString()
        external_data['weights.bin'] = base64.b64encode(payload).decode()
    return dict(name=name, onnx=base64.b64encode(raw).decode(), dimensions=dims or {}, external=external_data,
                inputs={k: dict(shape=list(v.shape), values=v.reshape(-1).tolist()) for k, v in inputs.items()},
                outputs={k: dict(shape=list(v.shape), values=v.reshape(-1).tolist()) for k, v in zip(outputs, expected)})


def fixtures():
    rng = np.random.default_rng(24)
    f = lambda shape: rng.normal(size=shape).astype(np.float32) * .2
    yield case('mlp', [h.make_node('Gemm', ['x', 'w', 'b'], ['hidden']),
                        h.make_node('Relu', ['hidden'], ['relu']),
                        h.make_node('Gemm', ['relu', 'out'], ['y'], transB=1, alpha=.7)],
               {'x': f((2, 4))}, {'y': (2, 2)}, {'w': f((4, 6)), 'b': f((6,)), 'out': f((2, 6))}, dims={'batch': 2})
    yield case('conv', [h.make_node('Conv', ['x', 'w', 'b'], ['c'], pads=[1, 0, 0, 1]),
                         h.make_node('Relu', ['c'], ['r']),
                         h.make_node('GlobalAveragePool', ['r'], ['p']),
                         h.make_node('Flatten', ['p'], ['y'])],
               {'x': f((1, 2, 5, 5))}, {'y': (1, 3)}, {'w': f((3, 2, 3, 3)), 'b': f((3,))})
    yield case('encoder', [h.make_node('MatMul', ['x', 'q'], ['a']),
                            h.make_node('MatMul', ['x', 'k'], ['b']),
                            h.make_node('Transpose', ['b'], ['bt'], perm=[0, 2, 1]),
                            h.make_node('MatMul', ['a', 'bt'], ['scores']),
                            h.make_node('Mul', ['scores', 'scale'], ['scaled']),
                            h.make_node('Softmax', ['scaled'], ['prob'], axis=-1),
                            h.make_node('MatMul', ['x', 'v'], ['values']),
                            h.make_node('MatMul', ['prob', 'values'], ['attn']),
                            h.make_node('Add', ['attn', 'x'], ['res']),
                            h.make_node('LayerNormalization', ['res', 'norm', 'bias'], ['y'], epsilon=1e-5)],
               {'x': f((1, 3, 4))}, {'y': (1, 3, 4)},
               {'q': f((4, 4)), 'k': f((4, 4)), 'v': f((4, 4)), 'scale': np.array(.5, np.float32),
                'norm': np.ones(4, np.float32), 'bias': f((4,))})
    yield case('external', [h.make_node('MatMul', ['x', 'w'], ['y'])],
               {'x': f((2, 3))}, {'y': (2, 4)}, {'w': f((3, 4))}, external=True)
    # Non-raw typed INT64 storage exercises shape operands without NumPy/ONNX
    # dependencies in the import/runtime tests themselves.
    value = h.make_tensor('shape', T.INT64, [2], [3, 2])
    yield case('reshape', [h.make_node('Constant', [], ['shape'], value=value),
                           h.make_node('Reshape', ['x', 'shape'], ['y'])],
               {'x': f((2, 3))}, {'y': (3, 2)}, {})
    yield case('identity', [h.make_node('Identity', ['x'], ['y'])],
               {'x': f((2, 3))}, {'y': (2, 3)}, {})
    # Both branches capture x and deliberately reuse local names. Neither
    # the condition nor the selected output may be frozen at import time.
    branches = {name: h.make_graph([h.make_node(op, ['x', 'bias'], ['local'])], name, [],
                [h.make_tensor_value_info('local', T.FLOAT, [2, 3])],
                [nh.from_array(np.full((2, 3), value, np.float32), 'bias')])
                for name, op, value in [('then_branch', 'Add', 2), ('else_branch', 'Mul', 3)]}
    for sign in (1, -1):
        yield case('if_positive' if sign == 1 else 'if_negative',
                   [h.make_node('ReduceSum', ['x'], ['sum'], keepdims=0),
                    h.make_node('Greater', ['sum', 'zero'], ['condition']),
                    h.make_node('If', ['condition'], ['y'], **branches)],
                   {'x': np.full((2, 3), sign, np.float32)}, {'y': (2, 3)},
                   {'zero': np.array(0, np.float32)})
    yield case('elementwise', [h.make_node('Sigmoid', ['x'], ['s']),
                              h.make_node('Tanh', ['s'], ['t']),
                              h.make_node('Exp', ['t'], ['e']),
                              h.make_node('Log', ['e'], ['l']),
                              h.make_node('Sqrt', ['l'], ['r']),
                              h.make_node('Neg', ['r'], ['n']),
                              h.make_node('Sub', ['n', 'x'], ['d']),
                              h.make_node('Div', ['d', 'divisor'], ['y'])],
               {'x': f((2, 3))}, {'y': (2, 3)}, {'divisor': np.array([1., 2., 3.], np.float32)})
    unary = ['Abs', 'Sign', 'Floor', 'Ceil', 'Round', 'Reciprocal', 'Cos', 'Sin', 'Tan',
             'Asin', 'Acos', 'Atan', 'Asinh', 'Atanh', 'Sinh', 'Cosh', 'Erf', 'Softsign',
             'HardSwish', 'Mish', 'LeakyRelu', 'ThresholdedRelu', 'HardSigmoid', 'Celu', 'Selu', 'Elu', 'Softplus']
    nodes = [h.make_node(op, ['x'], [op]) for op in unary]
    nodes += [h.make_node('Acosh', ['positive'], ['Acosh'])]
    yield case('unary_catalogue', nodes,
               {'x': np.array([[-.75, -.25, .25, .75]], np.float32),
                'positive': np.array([[1.1, 1.5, 2., 3.]], np.float32)},
               {op: (1,4) for op in unary + ['Acosh']}, {}, opset=18)
    nodes, outputs = [], {}
    for op in ['Less', 'LessOrEqual', 'Equal', 'Greater', 'GreaterOrEqual', 'IsNaN', 'IsInf']:
        nodes += [h.make_node(op, ['x'] if op.startswith('Is') else ['x','z'], [op+'b']),
                  h.make_node('Cast', [op+'b'], [op], to=T.FLOAT)]
        outputs[op] = (2,3)
    nodes += [h.make_node('Where', ['Lessb', 'x', 'z'], ['selected']),
              h.make_node('Not', ['Equalb'], ['not']),
              h.make_node('And', ['Lessb','not'], ['and']),
              h.make_node('Or', ['Greaterb','Equalb'], ['or']),
              h.make_node('Xor', ['and','or'], ['xor']),
              h.make_node('Cast', ['xor'], ['logical'], to=T.FLOAT)]
    outputs.update(selected=(2,3), logical=(2,3))
    yield case('comparisons', nodes, {'x': f((2,3)), 'z': f((2,3))}, outputs, {})
    nodes, outputs = [], {}
    for kind in ['Max','Min','Mean','Prod','Sum','L1','L2','SumSquare','LogSum','LogSumExp']:
        op = 'Reduce'+kind
        nodes += [h.make_node(op, ['x','axes'] if kind == 'Sum' else ['x'], [op],
                              **({'keepdims':0} if kind == 'Sum' else {'axes':[1], 'keepdims':0}))]
        outputs[op] = (2,)
    for op in ['ArgMax','ArgMin']:
        nodes += [h.make_node(op, ['x'], [op+'i'], axis=1, keepdims=0, select_last_index=1),
                  h.make_node('Cast', [op+'i'], [op], to=T.FLOAT)]
        outputs[op] = (2,)
    yield case('reductions', nodes, {'x': np.array([[1.,2.,3.],[3.,2.,3.]],np.float32)}, outputs,
               {'axes': np.array([1],np.int64)})
    yield case('shape_program', [h.make_node('Shape',['x'],['dims']),
               h.make_node('Gather',['dims','axis'],['batch']),
               h.make_node('Unsqueeze',['batch','axis_vector'],['batch_vector']),
               h.make_node('Concat',['batch_vector','tail'],['target'],axis=0),
               h.make_node('Reshape',['x','target'],['y'])],
               {'x': f((2,3,4))}, {'y': (2,12)},
               {'axis': np.array(0,np.int64), 'axis_vector':np.array([0],np.int64), 'tail':np.array([12],np.int64)})
    yield case('movement', [h.make_node('Slice',['x','start','end','axis','step'],['sliced']),
               h.make_node('Split',['x','lengths'],['left','right'],axis=1),
               h.make_node('Pad',['x','pads'],['padded'],mode='reflect'),
               h.make_node('TopK',['x','k'],['top','index'],axis=1),
               h.make_node('Cast',['index'],['indices'],to=T.FLOAT),
               h.make_node('CumSum',['x','cum_axis'],['cumulative'],exclusive=1,reverse=1)],
               {'x': f((2,6))}, {'sliced':(2,3),'left':(2,2),'right':(2,4),'padded':(2,8),'top':(2,2),'indices':(2,2),'cumulative':(2,6)},
               {k:np.array(v,np.int64) for k,v in dict(start=[5],end=[-9223372036854775808],axis=[1],step=[-2],lengths=[2,4],pads=[0,1,0,1],k=[2],cum_axis=1).items()})
    yield case('layernorm_outputs', [h.make_node('LayerNormalization',['x','scale','bias'],['y','mean','inv'],axis=-1)],
               {'x': f((2,3,4))}, {'y':(2,3,4),'mean':(2,3,1),'inv':(2,3,1)}, {'scale':f((4,)), 'bias':f((4,))})
    yield case('arithmetic', [h.make_node('Pow',['x','exponent'],['pow']),
               h.make_node('Mod',['x','divisor'],['mod'],fmod=1),
               h.make_node('Shrink',['x'],['shrink'],bias=.2,lambd=.5),
               h.make_node('Hardmax',['x'],['hardmax'],axis=1),
               h.make_node('LpNormalization',['x'],['norm'],axis=1,p=2),
               h.make_node('Dropout',['x'],['y','mask']),
               h.make_node('Cast',['mask'],['dropout_mask'],to=T.FLOAT),
               h.make_node('Softplus',['large'],['stable_softplus']),
               h.make_node('LeakyRelu',['x'],['leaky'],alpha=.1),
               h.make_node('Elu',['x'],['elu'],alpha=.7)],
               {'x':np.array([[-2.,-.3,.3,2.],[1.,1.5,2.,3.]],np.float32),'large':np.array([-100.,100.],np.float32)},
               {**{k:(2,4) for k in ['pow','mod','shrink','hardmax','norm','y','dropout_mask','leaky','elu']},'stable_softplus':(2,)},
               {'exponent':np.array(2.,np.float32),'divisor':np.array(1.3,np.float32)})
    yield case('loss', [h.make_node('SoftmaxCrossEntropyLoss',['x','labels','weights'],['loss','logp'],ignore_index=-1),
               h.make_node('NegativeLogLikelihoodLoss',['logp','labels','weights'],['nll'],ignore_index=-1,reduction='none')],
               {'x':f((3,4))}, {'loss':(),'logp':(3,4),'nll':(3,)},
               {'labels':np.array([1,-1,3],np.int64),'weights':np.array([1.,2.,3.,4.],np.float32)})
    yield case('groupnorm_gelu', [h.make_node('GroupNormalization',['x','scale','bias'],['group'],num_groups=2),
               h.make_node('Gelu',['group'],['exact']), h.make_node('Gelu',['group'],['tanh'],approximate='tanh'),
               h.make_node('MeanVarianceNormalization',['x'],['mvn'])],
               {'x':f((1,4,2,2))}, {k:(1,4,2,2) for k in ['group','exact','tanh','mvn']},
               {'scale':f((4,)), 'bias':f((4,))},opset=21)
    for mode in ['nearest','linear','cubic']:
        yield case('resize_'+mode, [h.make_node('Resize',['x','','','sizes'],['y'],mode=mode)],
                   {'x':f((1,2,3,4))}, {'y':(1,2,5,6)}, {'sizes':np.array([1,2,5,6],np.int64)})
    yield case('resize_axes', [h.make_node('Resize',['x','','scales'],['y'],mode='nearest',
                   axes=[2,3],coordinate_transformation_mode='asymmetric',nearest_mode='ceil')],
                   {'x':f((1,1,3,4))}, {'y':(1,1,6,6)}, {'scales':np.array([2.,1.5],np.float32)},opset=18)
    yield case('indexed', [h.make_node('ScatterElements',['x','indices','updates'],['scatter'],axis=1,reduction='add'),
               h.make_node('GatherElements',['scatter','indices'],['gather'],axis=1),
               h.make_node('GatherND',['x','nd'],['nd_out']),
               h.make_node('GatherND',['x','batch_nd'],['batched'],batch_dims=1),
               h.make_node('Trilu',['x','diagonal'],['triangle']),
               h.make_node('OneHot',['indices','depth','hotvalues'],['onehot'],axis=0)],
               {'x':f((2,3)),'updates':f((2,2))}, {'scatter':(2,3),'gather':(2,2),'nd_out':(2,), 'batched':(2,2),'triangle':(2,3),'onehot':(3,2,2)},
               {'indices':np.array([[0,-1],[1,0]],np.int64),'nd':np.array([[0,1],[1,2]],np.int64),
                'batch_nd':np.array([[[0],[2]],[[1],[0]]],np.int64),'diagonal':np.array(1,np.int64),
                'depth':np.array(3,np.int64),'hotvalues':np.array([-1.,2.],np.float32)},opset=18)
    yield case('quantization', [h.make_node('QuantizeLinear',['x','scale','zero'],['q'],axis=1),
               h.make_node('DequantizeLinear',['q','scale','zero'],['dq'],axis=1),
               h.make_node('Cast',['q'],['q_float'],to=T.FLOAT),
               h.make_node('DynamicQuantizeLinear',['x'],['dynamic','ds','dz']),
               h.make_node('DequantizeLinear',['dynamic','ds','dz'],['dynamic_dq'])],
               {'x':np.array([[-2.5,0.5,2.5],[20.2,-1.7,1.1]],np.float32)},
               {'dq':(2,3),'q_float':(2,3),'dynamic_dq':(2,3)},
               {'scale':np.array([1.,.5,1.],np.float32),'zero':np.array([0,0,0],np.int8)})
    yield case('quantized_matmul', [h.make_node('MatMulInteger',['a','b','az','bz'],['integer']),
               h.make_node('Cast',['integer'],['i'],to=T.FLOAT),
               h.make_node('QLinearMatMul',['a','as','az','b','bs','bz','ys','yz'],['quantized']),
               h.make_node('Cast',['quantized'],['q'],to=T.FLOAT),
               h.make_node('Add',['q','x'],['y'])],
               {'x':f((2,2))}, {'i':(2,2),'y':(2,2)},
               {'a':np.array([[2,3,5],[1,4,7]],np.uint8),'b':np.array([[3,1],[4,5],[2,6]],np.uint8),
                'az':np.array(1,np.uint8),'bz':np.array(2,np.uint8),'yz':np.array(2,np.uint8),
                'as':np.array(.3,np.float32),'bs':np.array(.7,np.float32),'ys':np.array(.8,np.float32)})
    yield case('creation', [h.make_node('ConstantOfShape',['shape'],['filled'],value=nh.from_array(np.array([2.5],np.float32))),
               h.make_node('Add',['filled','x'],['y']),
               *[h.make_node(op,['size'],[op],periodic=0) for op in ['HannWindow','HammingWindow','BlackmanWindow']]],
               {'x':f((2,3))}, {'y':(2,3),**{op:(8,) for op in ['HannWindow','HammingWindow','BlackmanWindow']}},
               {'shape':np.array([2,3],np.int64),'size':np.array(8,np.int64)},opset=18)
    yield case('contrib', [h.make_node('BiasGelu',['x','bias'],['gelu'],domain='com.microsoft'),
               h.make_node('FastGelu',['x'],['fast'],domain='com.microsoft'),
               h.make_node('SkipLayerNormalization',['x','skip','scale','bias'],['normalized','','','added'],domain='com.microsoft'),
               h.make_node('Binarizer',['x'],['binary'],domain='ai.onnx.ml',threshold=.1),
               h.make_node('ArrayFeatureExtractor',['x','indices'],['selected'],domain='ai.onnx.ml')],
               {'x':f((2,3)),'skip':f((2,3))}, {'gelu':(2,3),'fast':(2,3),'normalized':(2,3),'added':(2,3),'binary':(2,3),'selected':(2,2)},
               {'bias':f((3,)),'scale':f((3,)),'indices':np.array([0,2],np.int64)},domains={'com.microsoft':1,'ai.onnx.ml':1})
    yield case('lrn_crop', [h.make_node('LRN',['x'],['normalized'],size=3,alpha=.2,beta=.75,bias=1.2),
               h.make_node('CenterCropPad',['x','size'],['cropped'],axes=[2,3])],
               {'x':f((2,5,4,2))}, {'normalized':(2,5,4,2),'cropped':(2,5,2,4)},
               {'size':np.array([2,4],np.int64)},opset=18)
    yield case('spatial', [h.make_node('MaxPool',['x'],['max','idx'],kernel_shape=[2,2],strides=[2,2]),
               h.make_node('Cast',['idx'],['indices'],to=T.FLOAT),
               h.make_node('AveragePool',['x'],['avg'],kernel_shape=[3,3],strides=[2,2],auto_pad='SAME_UPPER'),
               h.make_node('Conv',['x','w'],['conv'],strides=[2,2],auto_pad='SAME_LOWER'),
               h.make_node('ConvTranspose',['max','tw'],['transpose'],strides=[2,2],output_shape=[4,4]),
               h.make_node('InstanceNormalization',['x','scale','bias'],['normalized']),
               h.make_node('SpaceToDepth',['x'],['packed'],blocksize=2),
               h.make_node('DepthToSpace',['packed'],['unpacked'],blocksize=2)],
               {'x':f((1,1,4,4))}, {'max':(1,1,2,2),'indices':(1,1,2,2),'avg':(1,1,2,2),
               'conv':(1,3,2,2),'transpose':(1,3,4,4),'normalized':(1,1,4,4),'packed':(1,4,2,2),'unpacked':(1,1,4,4)},
               {'w':f((3,1,3,3)), 'tw':f((1,3,3,3)), 'scale':f((1,)), 'bias':f((1,))})


if __name__ == '__main__':
    # Regeneration is explicit; ordinary tests require neither ONNX package.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({'cases': list(fixtures()), 'onnx': onnx.__version__,
                                     'onnxruntime': ort.__version__}, separators=(',', ':')) + '\n')
