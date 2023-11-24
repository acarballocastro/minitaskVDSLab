import torch
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.nn import MSELoss
from captum.attr import IntegratedGradients

from lfxai.models.images import AutoEncoderMnist, EncoderMnist, DecoderMnist
from lfxai.models.pretext import Identity
from lfxai.explanations.features import attribute_auxiliary
from lfxai.explanations.examples import SimplEx

# Select torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("Loading data...")

# Load data
data_dir = Path.cwd() / "data/mnist"
train_dataset = MNIST(data_dir, train=True, download=True)
test_dataset = MNIST(data_dir, train=False, download=True)
train_dataset.transform = transforms.Compose([transforms.ToTensor()])
test_dataset.transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=100)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

print("Loading model...")

# Get a model
encoder = EncoderMnist(encoded_space_dim=10)
decoder = DecoderMnist(encoded_space_dim=10)
model = AutoEncoderMnist(encoder, decoder, latent_dim=10, input_pert=Identity())
model.to(device)

print("Training model... te quiero francesc")

# Get label-free example importance
train_subset = Subset(train_dataset, indices=list(range(500))) # Limit the number of training examples
train_subloader = DataLoader(train_subset, batch_size=500)
attr_method = SimplEx(model, loss_f=MSELoss())
example_importance = attr_method.attribute_loader(device, train_subloader, test_loader)