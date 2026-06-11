import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
print("Ładowanie i SKALOWANIE ręcznie wyekstrahowanych danych MNIST...")

# UWAGA: Podmień ścieżki na pliki, które faktycznie zawierają Twoje 2 cechy!
# Wcześniej w Twoim kodzie widziałem "X_flat_train.npy", ale to prawdopodobnie
# spłaszczone 784 piksele. Tutaj musisz wczytać macierze o wymiarach (N, 2).
try:
    X_train = np.load("features/X2_train.npy")
    X_test = np.load("features/X2_test.npy")
    y_train = np.load("features/y_train.npy")
    y_test = np.load("features/y_test.npy")
except FileNotFoundError:
    print("[BŁĄD] Nie znaleziono plików .npy. Upewnij się, że podałeś prawidłowe ścieżki do plików z 2 cechami.")
    exit()

# Nadajemy nazwy cechom dla czytelności wyników
feature_names = ["Cecha 1 (pętle)", "Cecha 2 (np. wariancja szerokości)"]

# Skalowanie (niezbędne!)
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
# Tworzymy model z wejściem = 2, wyjściem = 10 klas cyfr (0-9).
# Parametr hidden_size ustawiłem na 100, tak jak w Twoim pliku konfiguracyjnym,
# ale jeśli dla 2 cech użyłeś innej wartości podczas treningu, zmień ją tutaj.
model = MLP(input_size=2, hidden_size=16, num_classes=10, use_relu=True)

try:
    model.load_state_dict(torch.load('trained_models/best_model_mnist_2_features.pth'))
except FileNotFoundError:
    print("[BŁĄD] Nie znaleziono wag modelu. Sprawdź nazwę pliku .pth.")
    exit()

# ==========================================
# 4. ETAP 1: GLOBALNA OCENA DOKŁADNOŚCI
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

# Baseline dla Integrated Gradients
baseline = torch.mean(X_train_tensor, dim=0, keepdim=True)


def format_all_features(attr_array, feature_names):
    """Wypisuje obie cechy posortowane malejąco według istotności"""
    attr_flat = attr_array.flatten()
    sorted_indices = np.argsort(np.abs(attr_flat))[::-1]

    results = []
    for idx in sorted_indices:
        results.append(f"{feature_names[idx]:<25} {attr_flat[idx]:>8.4f}")

    return "\n    " + "\n    ".join(results)


selected_indices = []
class_counts = {k: 0 for k in range(10)}  # Słownik dla cyfr 0-9

# Przeszukujemy zbiór testowy
for i in range(len(y_test_tensor)):
    true_label = y_test_tensor[i].item()

    # Jeśli dla danej cyfry nie mamy jeszcze próbki, dodajemy jej indeks
    if class_counts[true_label] < 1:
        selected_indices.append(i)
        class_counts[true_label] += 1

    # Jeśli zebraliśmy już wszystkie 10 cyfr, przerywamy pętlę
    if sum(class_counts.values()) == 10:
        break

# Sortujemy indeksy, aby wyniki wyświetlały się w miarę po kolei (opcjonalnie)
selected_indices.sort(key=lambda idx: y_test_tensor[idx].item())

# Analizujemy 10 wyselekcjonowanych próbek
for i in selected_indices:
    input_tensor = X_test_tensor[i].unsqueeze(0)
    true_label = y_test_tensor[i].item()

    input_tensor.requires_grad_()

    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    print("-" * 50)
    print(f"Próbka nr {i + 1}")
    print(f"Prawdziwa cyfra: {true_label} | Przewidziana cyfra: {predicted_class}\n")

    saliency_attr = saliency.attribute(input_tensor, target=predicted_class)
    print(f"SALIENCY: {format_all_features(saliency_attr.detach().numpy(), feature_names)}\n")

    ig_attr = ig.attribute(input_tensor, baselines=baseline, target=predicted_class)
    print(f"INTEGRATED GRADIENTS: {format_all_features(ig_attr.detach().numpy(), feature_names)}\n")

    # Dla tak prostego wejścia można zredukować liczbę n_samples
    lime_attr = lime.attribute(input_tensor, target=predicted_class, n_samples=100)
    print(f"LIME: {format_all_features(lime_attr.detach().numpy(), feature_names)}\n")