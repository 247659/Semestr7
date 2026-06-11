import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from captum.attr import Saliency, IntegratedGradients, Lime

from models.model import MLP


# ==========================================
# 1. FUNKCJA DO OCENY DOKŁADNOŚCI MODELU
# ==========================================
def evaluate_model_accuracy(model, test_dataloader):
    model.eval()  # Tryb ewaluacji
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # Wyłączenie gradientów dla przyspieszenia obliczeń
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
print("Ładowanie i przygotowanie danych...")
iris = load_iris()
X_all = iris.data
y_all = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)
# -----------------------------------------------------------------

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ==========================================
# 3. DEFINICJA I ŁADOWANIE MODELU
# ==========================================
# TUTAJ PODMIEŃ KOD NA SWOJĄ ARCHITEKTURĘ I ZAŁADUJ WAGI
model = MLP(input_size=4, hidden_size=16, num_classes=3, use_relu=True)

# Tutaj załaduj wagi swojego modelu, np.:
model.load_state_dict(torch.load('trained_models/best_model_iris.pth'))

# ==========================================
# 4. ETAP 1: GLOBALNA OCENA DOKŁADNOŚCI (ACCURACY)
# ==========================================
print("\n=== ETAP 1: OCENA DOKŁADNOŚCI MODELU ===")
acc = evaluate_model_accuracy(model, test_loader)
print(f"Dokładność modelu na zbiorze testowym: {acc:.2f}%\n")

# ==========================================
# 5. ETAP 2: LOKALNE OBJAŚNIANIE (CAPTUM XAI)
# ==========================================
print("=== ETAP 2: OBJAŚNIANIE DECYZJI MODELU (XAI) ===")
model.eval()  # Upewnienie się, że model jest w trybie ewaluacji

saliency = Saliency(model)
ig = IntegratedGradients(model)
lime = Lime(model)

# Baseline: średnie wymiary cech z całego zbioru treningowego
baseline = torch.mean(X_train_tensor, dim=0, keepdim=True)

print(f"Kolejność cech: {feature_names}\n")
selected_indices = []
class_counts = {0: 0, 1: 0, 2: 0}  # Słownik do zliczania znalezionych próbek dla każdej klasy

# Szukamy w całym zbiorze testowym
for i in range(len(y_test_tensor)):
    true_label = y_test_tensor[i].item()

    # Jeśli mamy mniej niż 2 próbki z danej klasy, dodajemy jej indeks
    if class_counts[true_label] < 2:
        selected_indices.append(i)
        class_counts[true_label] += 1

    # Jeśli zebraliśmy już 6 próbek (po 2 z 3 klas), przerywamy pętlę
    if sum(class_counts.values()) == 6:
        break

# Analizujemy wybrane 6 próbek
for i in selected_indices:
    input_tensor = X_test_tensor[i].unsqueeze(0)
    true_label = y_test_tensor[i].item()

    # Śledzenie gradientów włączone tylko dla analizowanej próbki
    input_tensor.requires_grad_()

    # Przewidywanie klasy przez model
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    print(f"--- Próbka nr {i + 1} ---")
    print(f"Prawdziwa klasa: {target_names[true_label]} | Predykcja modelu: {target_names[predicted_class]}")
    original_values = X_test[i]
    print(f"Wartości wejściowe:  {original_values.round(2)}")

    # Obliczanie atrybucji Saliency
    saliency_attr = saliency.attribute(input_tensor, target=predicted_class)
    print(f"Saliency:            {saliency_attr.detach().numpy()[0].round(4)}")

    # Obliczanie atrybucji Integrated Gradients
    ig_attr = ig.attribute(input_tensor, baselines=baseline, target=predicted_class)
    print(f"Integrated Gradients:{ig_attr.detach().numpy()[0].round(4)}")

    # Obliczanie LIME
    lime_attr = lime.attribute(input_tensor, target=predicted_class, n_samples=200)
    print(f"LIME:                {lime_attr.detach().numpy()[0].round(4)}\n")
