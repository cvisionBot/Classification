import onnx 
import onnxruntime 
 
import torch 
import argparse 
import numpy as np 
 
from module.classifier import Classifier 
from utils.module_select import get_model 
from utils.yaml_helper import get_train_configs 
 
 
def load_onnx_model(onnx_path): 
    return onnxruntime.InferenceSession(onnx_path) 
 
def load_pytorch_model(cfg): 
    model = get_model(cfg['model'])(in_channels=3, classes=cfg['classes']) 
    if torch.cuda.is_available: 
        model = model.to('cpu') 
    model_module = Classifier.load_from_checkpoint( 
        '/ssd2/lyj/resnet50_cls_ckpt/ResNet50_150_Epoch.ckpt', model=model 
    ) 
    model_module.eval() 
    return model_module.model 
 
def getOnnxInputs(onnx_graph): 
    input_tensors = {t.name for t in onnx_graph.initializer} 
    inputs = [] 
    for i in onnx_graph.input: 
        if i.name not in input_tensors: 
            inputs.append(i.name) 
    return inputs 
 
def getOnnxOutputs(onnx_graph): 
    outputs = [] 
    for i in onnx_graph.output: 
        outputs.append(i.name) 
    return outputs 
 
def gen_input(models, in_node): 
    np.random.seed(5) 
    in_shapes = [get_input_shape_onnx(models[0], in_node)] 
    in_shape = in_shapes[0] 
    return np.random.rand(1, *in_shape).astype(np.float32) 
 
def get_input_shape_onnx(onnx_model, in_node): 
    for node in onnxruntime.InferenceSession.get_inputs(onnx_model): 
        if node.name == in_node: 
            return node.shape[1:] 
 
def run_models(models, in_node, out_node, input_tensor): 
    net_results = [] 
    net_results.append(net_forward_onnx(models[0], in_node, out_node, input_tensor)) 
    net_results.append(net_forward_pytorch(models[1], input_tensor)) 
    return net_results 
 
def net_forward_onnx(onnx_model, in_node, out_node, input_tensor): 
    result = onnx_model.run(out_node, {in_node : input_tensor}) 
    return result 
 
def net_forward_pytorch(pytorch_model, input_tensor): 
    input_tensor = torch.Tensor(input_tensor) 
    if torch.cuda.is_available: 
        input_tensor = input_tensor.to("cpu") 
    result =[t.detach().numpy() for t in pytorch_model(input_tensor)["pred"]] 
    return result 
 
def check_results(net_results): 
    onnx_results = net_results[0] 
    pytorch_results = net_results[1] 
 
    for i, result in enumerate(onnx_results): 
        print("onnx : ", result) 
        print("pytorch : ", pytorch_results[i]) 
 
        # check if result are same by cosine distance 
        dot_result = np.dot(result.flatten(), pytorch_results[i].flatten()) 
        left_norm = np.sqrt(np.square(result).sum()) 
        right_norm = np.sqrt(np.square(pytorch_results[i]).sum()) 
        cos_sim = dot_result / (left_norm * right_norm) 
        print("cos sim between onnx and pytorch models: {}".format(cos_sim)) 
 
def main(cfg): 
    models = [load_onnx_model(cfg['model']+'.onnx'), load_pytorch_model(cfg)] 
    onnx_graph = onnx.load(cfg['model']+'.onnx').graph 
    in_node = getOnnxInputs(onnx_graph) 
    out_node = getOnnxOutputs(onnx_graph) 
    if not len(in_node) == 1: 
        raise Exception("Only one input is supported, but {} provided: {}".format( 
            len(in_node), in_node 
        )) 
     
    input_tensor = gen_input(models, in_node[0]) 
    net_results = run_models(models, in_node[0], out_node, input_tensor) 
    for i, node in enumerate(out_node): 
        print("output tensor shape of {}: {} for onnx vs {} for PyTorch".format( 
            node, net_results[0][i].shape, net_results[1][i].shape)) 
    check_results(net_results) 
 
 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--cfg', required=True, type=str, help='Train config file') 
    args = parser.parse_args() 
    cfg = get_train_configs(args.cfg) 
    main(cfg)