import torch 
import argparse 
from module.classifier import Classifier 
from utils.module_select import get_model 
from utils.yaml_helper import get_train_configs 
 
 
def main(cfg): 
    model = get_model(cfg['model'])(in_channels=3, classes=cfg['classes']) 
    model_module = Classifier.load_from_checkpoint( 
        '/ssd2/lyj/resnet50_cls_ckpt/ResNet50_150_Epoch.ckpt', model=model 
    ) 
    model_module.eval() 
 
    # Convert PyTorch To ONNX 
    dumTensor = torch.rand(1, 3, 64, 64) 
    torch.onnx.export(model_module.model, dumTensor, cfg['model']+'.onnx', 
                      export_params=True, opset_version=9, do_constant_folding=True,
                      input_names=['input'], output_names=['pred'])
 
if __name__ == '__main__': 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--cfg', required=True, type=str, help='Train config file') 
    args = parser.parse_args() 
    cfg = get_train_configs(args.cfg) 
    main(cfg)
