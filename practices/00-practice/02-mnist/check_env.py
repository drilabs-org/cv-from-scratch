import torch
import torchvision

print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    device = "cuda (GPU Nvidia)"
elif torch.backends.mps.is_available():
    device = "mps (GPU Apple Silicon)"
else:
    device = "cpu"
print("Device:", device)
