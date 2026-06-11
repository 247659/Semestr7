import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from captum.attr import Saliency, IntegratedGradients, Lime

from models.model import MLP


# ==========================================
# 1. FUNKCJA DO OCENY DOKŁADNOŚCI MODELU
# ==========================================
def evaluate_model_accuracy(model, test_dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

    return (correct_predictions / total_predictions) * 100


# ==========================================
# 2. PRZYGOTOWANIE I SKALOWANIE DANYCH
# ==========================================
print("Ładowanie, podział i SKALOWANIE danych Wine...")
wine = load_wine()
X_all = wine.data
y_all = wine.target
feature_names = np.array(wine.feature_names)  # Zmiana na numpy array dla łatwiejszego indeksowania
target_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Skalowanie (niezbędne, tak jak przy treningu)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ==========================================
# 3. DEFINICJA I ŁADOWANIE MODELU
# ==========================================
# Parametry odczytane z Twojego 'wine_optimal_config' z pliku treningowego
model = MLP(input_size=13, hidden_size=8, num_classes=3, use_relu=True)

# Ładowanie wag dla modelu Wine
model.load_state_dict(torch.load('trained_models/best_model_wine.pth'))

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
model.eval()

saliency = Saliency(model)
ig = IntegratedGradients(model)
lime = Lime(model)

# Baseline dla IG: średnie wartości wyliczone ze skalowanego zbioru treningowego
baseline = torch.mean(X_train_tensor, dim=0, keepdim=True)


def format_all_features(attr_array, feature_names):
    """Wypisuje wszystkie cechy posortowane malejąco według istotności (wartości bezwzględnej)"""
    attr_flat = attr_array.flatten()
    # Sortowanie indeksów po wartości bezwzględnej (od największej)
    sorted_indices = np.argsort(np.abs(attr_flat))[::-1]

    results = []
    for idx in sorted_indices:
        # Znak minus/plus informuje, czy cecha ciągnęła w stronę przewidzianej klasy, czy przeciw niej
        results.append(f"{feature_names[idx]:<30} {attr_flat[idx]:>8.4f}")

    # Dodanie wcięć dla lepszej czytelności w konsoli
    return "\n    " + "\n    ".join(results)

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

    input_tensor.requires_grad_()

    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    print(f"--- Próbka nr {i + 1} ---")
    print(f"Prawdziwa klasa: {target_names[true_label]} | Predykcja modelu: {target_names[predicted_class]}")
    print(f"Wartości wejściowe:  {input_tensor.detach().numpy()[0].round(2)}")
    # Saliency
    saliency_attr = saliency.attribute(input_tensor, target=predicted_class)
    print(f"SALIENCY (Wszystkie cechy): {format_all_features(saliency_attr.detach().numpy(), feature_names)}\n")

    ig_attr = ig.attribute(input_tensor, baselines=baseline, target=predicted_class)
    print(f"INTEGRATED GRADIENTS (Wszystkie cechy): {format_all_features(ig_attr.detach().numpy(), feature_names)}\n")

    lime_attr = lime.attribute(input_tensor, target=predicted_class, n_samples=200)
    print(f"LIME (Wszystkie cechy): {format_all_features(lime_attr.detach().numpy(), feature_names)}\n")