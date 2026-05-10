from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader

from datasets import NumpyDataset
from model import MLP
from train import train_model
from evaluate import evaluate_model


# Load dataset
wine = load_wine()

X = wine.data
y = wine.target


# Normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Dataset
train_dataset = NumpyDataset(X_train, y_train)
test_dataset = NumpyDataset(X_test, y_test)


# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


# Model
model = MLP(
    input_size=13,
    hidden_size=32,
    num_classes=3,
    use_relu=True
)


# Train
train_model(model, train_loader, epochs=100, lr=0.01)


# Evaluate
evaluate_model(model, test_loader)