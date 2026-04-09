import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
