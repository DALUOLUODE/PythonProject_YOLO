import torch
import torchvision

print(f"PyTorch版本: {torch.__version__}, CUDA支持: {torch.cuda.is_available()}")
print(f"Torchvision版本: {torchvision.__version__}")