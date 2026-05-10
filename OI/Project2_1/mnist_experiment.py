import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import NumpyDataset
from model import MLP
from train import train_model
# Załóżmy, że dodałeś nową funkcję do evaluate.py:
from evaluate import evaluate_advanced_metrics
from visualize import voronoi_diagram

# 1. Wczytanie wybranego zestawu cech (np. Twoja Metoda 1)
# Pamiętaj, aby ścieżki zgadzały się z tymi, gdzie zapisałeś pliki .npy
X_train_raw = np.load("features/X1_train.npy")
y_train_raw = np.load("features/y_train.npy")
X_test_raw = np.load("features/X1_test.npy")
y_test_raw = np.load("features/y_test.npy")

# 2. Skalowanie danych (bardzo ważne dla MLP i Silhouette Score!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 3. Przygotowanie Datasetów i DataLoaderów
train_dataset = NumpyDataset(X_train_scaled, y_train_raw)
test_dataset = NumpyDataset(X_test_scaled, y_test_raw)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Inicjalizacja modelu (input_size = 2, bo wczytaliśmy X1, czyli 2 cechy)
model = MLP(
    input_size=X_train_scaled.shape[1],
    hidden_size=16,
    num_classes=10,
    use_relu=True
)

# 5. Trening
print("Rozpoczynam trening...")
train_model(model, train_loader)

# 7. ZAAWANSOWANE METRYKI
# Przekazujemy model, loader oraz wyskalowane cechy testowe
evaluate_advanced_metrics(model, test_loader, X_test_scaled, y_test_raw)
voronoi_diagram(X_test_scaled, y_test_raw)