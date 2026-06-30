import torch

print("torch version :", torch.__version__)
print("cpu :", True)
print("MPS(APPLE) :", torch.backends.mps.is_available())
print("Cuda :", torch.cuda.is_available())


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("selected device:", device)

