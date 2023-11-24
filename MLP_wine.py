import lfxai
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lfxai.models.pretext import Identity
from lfxai.explanations.examples import InfluenceFunctions, TracIn

from ucimlrepo import fetch_ucirepo 

from pathlib import Path

res_dir = Path.cwd() / "results"
dtype = torch.float32
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 

# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 

# split in train, validation and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# normalize data using scikit-learn
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
  
# convert to tensors
X_train = torch.tensor(X_train, dtype=dtype)
X_val = torch.tensor(X_val, dtype=dtype)
X_test = torch.tensor(X_test, dtype=dtype)

y_train = torch.tensor(y_train["class"].values-1, dtype=torch.long)
y_val = torch.tensor(y_val["class"].values-1, dtype=torch.long)
y_test = torch.tensor(y_test["class"].values-1, dtype=torch.long)

# Multilayer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

print(X_train.shape)
print(y_train.shape)    
    
# create data loaders
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(   
    torch.utils.data.TensorDataset(X_val, y_val),
    batch_size=32,
    shuffle=False,
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_test, y_test),
    batch_size=32,
    shuffle=False,
)

    
# Training
mlp = MLP(13, [64, 64], 3)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate the model
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            outputs = mlp(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch [{}/{}], Accuracy: {:.2f}%'.format(epoch + 1, 100, 100 * correct / total))

# Use lfxai to explain the model

attr_method = InfluenceFunctions(mlp, criterion, res_dir)
example_importance = attr_method.attribute_loader("cpu", train_loader, val_loader)