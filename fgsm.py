# %%
print("hello")

# %%
import torch

# %%
device = torch.device("cuda")

# %%
from torchvision import datasets
from torchvision.transforms import ToTensor

# %%
training_data = datasets.MNIST("data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST("data", train=False, download=True, transform=ToTensor())

# %%
training_data[0][1]

# %%
