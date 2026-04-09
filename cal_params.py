import argparse
from net import Net
import os
import time
from thop import profile
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD Parameter and FLOPs")
parser.add_argument("--model_names", default=['ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net'], nargs='+', 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'RISTDnet', 'UIUNet', 'U-Net', 'RDIAN', 'ISTDU-Net', 'CompareSPP', 'CompareFPN', 'ComparePANet', 'CompareACM'")
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="Device used for parameter/FLOPs calculation: 'auto', 'cpu', or 'cuda:0'",
)

global opt
opt = parser.parse_args()


def resolve_device(device_arg):
    if device_arg in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested via --device={}, but the current PyTorch build has no CUDA support. "
            "Install a CUDA-enabled PyTorch build or run with --device cpu.".format(device_arg)
        )
    return device


opt.device = resolve_device(opt.device)
print("Using device: {}".format(opt.device))

if __name__ == '__main__':
    opt.f = open('./params_' + (time.ctime()).replace(' ', '_') + '.txt', 'w')
    input_img = torch.rand(1,1,256,256).to(opt.device)
    for model_name in opt.model_names:
        net = Net(model_name, mode='test').to(opt.device)    
        flops, params = profile(net, inputs=(input_img, ))
        print(model_name)
        print('Params: %2fM' % (params/1e6))
        print('FLOPs: %2fGFLOPs' % (flops/1e9))
        opt.f.write(model_name + '\n')
        opt.f.write('Params: %2fM\n' % (params/1e6))
        opt.f.write('FLOPs: %2fGFLOPs\n' % (flops/1e9))
        opt.f.write('\n')
    opt.f.close()
        
