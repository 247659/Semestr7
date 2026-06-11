import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from skimage.segmentation import slic

from captum.attr import Saliency, IntegratedGradients, Lime
from captum.attr import visualization as viz

# Załóżmy, że importujesz model z pliku cnn_classifier_second.py
from models.cnn_classifier_second import CNNClassifierSecond


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
# 2. PRZYGOTOWANIE DANYCH (Zgodnie z Twoim plikiem)
# ==========================================
print("Pobieranie i przygotowanie danych MNIST za pomocą torchvision...")

transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform_no_aug
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ==========================================
# 3. DEFINICJA I ŁADOWANIE MODELU CNN
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifierSecond(num_classes=10).to(device)

try:
    # Zgodnie z formatem zapisu z pliku experiment2_mnist.py
    model.load_state_dict(torch.load('trained_models/BEST_OVERALL_CNNClassifier_MNIST.pth', map_location=device))
except FileNotFoundError:
    print("[BŁĄD] Nie znaleziono wag modelu. Sprawdź, czy uruchomiłeś najpierw skrypt treningowy.")
    exit()

# ==========================================
# 4. ETAP 1: GLOBALNA OCENA DOKŁADNOŚCI
# ==========================================
print("\n=== ETAP 1: OCENA DOKŁADNOŚCI MODELU ===")
acc = evaluate_model_accuracy(model, test_loader)
print(f"Dokładność CNN na zbiorze testowym: {acc:.2f}%\n")

# ==========================================
# 5. ETAP 2: LOKALNE OBJAŚNIANIE I WIZUALIZACJE
# ==========================================
print("=== ETAP 2: GENEROWANIE WIZUALIZACJI XAI ===")
model.eval()

saliency = Saliency(model)
ig = IntegratedGradients(model)
lime = Lime(model)


def unnormalize(tensor):
    """Cofa normalizację (0.1307, 0.3081) na potrzeby czystej wizualizacji i segmentacji"""
    return tensor * 0.3081 + 0.1307


# Baseline dla obrazów MNIST: całkowicie czarne tło.
# Po normalizacji "0" (czarny piksel) to (0 - 0.1307) / 0.3081
normalized_black_pixel = -0.1307 / 0.3081


def plot_image_attributions(input_tensor, saliency_attr, ig_attr, lime_attr, true_lbl, pred_lbl):
    """Generuje panel z nałożonymi mapami ciepła dla analizowanego obrazu."""

    # Odwrócenie normalizacji do wizualizacji [0, 1]
    unnorm_img = unnormalize(input_tensor).squeeze(0).cpu().detach().numpy()

    # Format (H, W, C)
    original_image = np.transpose(unnorm_img, (1, 2, 0))
    s_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    i_attr = np.transpose(ig_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    l_attr = np.transpose(lime_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    print(f"Rysowanie... Prawdziwa klasa: {true_lbl} | Predykcja: {pred_lbl}")

    # Tworzymy jedno główne okno (Figure) i 4 miejsca na wykresy (Axes) obok siebie
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Prawdziwa klasa: {true_lbl} | Predykcja: {pred_lbl}", fontsize=16, fontweight='bold')

    # 1. Oryginalny obraz
    # Podajemy s_attr jako atrapę, aby Captum nie wyrzuciło błędu o braku właściwości .shape
    viz.visualize_image_attr(
        s_attr, original_image,
        method="original_image",
        title="Oryginał",
        plt_fig_axis=(fig, axes[0]),
        use_pyplot=False
    )

    # 2. Saliency
    viz.visualize_image_attr(
        s_attr, original_image,
        method="blended_heat_map", sign="absolute_value",
        title="Saliency", cmap="PiYG",
        plt_fig_axis=(fig, axes[1]),
        use_pyplot=False
    )

    # 3. Integrated Gradients
    viz.visualize_image_attr(
        i_attr, original_image,
        method="blended_heat_map", sign="all",
        title="Integrated Gradients", cmap="PiYG",
        plt_fig_axis=(fig, axes[2]),
        use_pyplot=False
    )

    # 4. LIME (SLIC)
    viz.visualize_image_attr(
        l_attr, original_image,
        method="blended_heat_map", sign="all",
        title="LIME", cmap="PiYG",
        plt_fig_axis=(fig, axes[3]),
        use_pyplot=False
    )

    # Poprawia marginesy, żeby tytuły na siebie nie nachodziły
    plt.tight_layout()
    plt.show()


selected_indices = []
class_counts = {k: 0 for k in range(10)}  # Liczniki dla cyfr 0-9

for i in range(len(test_dataset)):
    _, true_label = test_dataset[i]

    if class_counts[true_label] < 1:
        selected_indices.append(i)
        class_counts[true_label] += 1

    if sum(class_counts.values()) == 10:
        break

# Sortujemy indeksy, żeby okienka wyskakiwały ładnie po kolei: najpierw 0, potem 1, itd.
selected_indices.sort(key=lambda idx: test_dataset[idx][1])

# ==========================================
# ANALIZA WYBRANYCH 10 PRÓBEK
# ==========================================
for i in selected_indices:  # Zmień tę wartość, aby przeanalizować więcej próbek
    input_tensor, true_label = test_dataset[i]
    input_tensor = input_tensor.unsqueeze(0).to(device)  # Dodanie wymiaru batch i przeniesienie na device

    input_tensor.requires_grad_()
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # --- 1. Saliency ---
    saliency_attr = saliency.attribute(input_tensor, target=predicted_class)

    # --- 2. Integrated Gradients ---
    baseline_image = torch.full_like(input_tensor, normalized_black_pixel).to(device)
    ig_attr = ig.attribute(input_tensor, baselines=baseline_image, target=predicted_class)

    # --- 3. LIME z segmentacją SLIC ---
    # Generujemy super-piksele na odnormalizowanym obrazie (lepsze i bardziej naturalne grupy pikseli)
    image_for_slic = unnormalize(input_tensor).squeeze().cpu().detach().numpy()
    segments = slic(image_for_slic, n_segments=40, compactness=0.1, start_label=0, channel_axis=None)

    feature_mask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).to(device)

    lime_attr = lime.attribute(
        input_tensor,
        target=predicted_class,
        feature_mask=feature_mask,
        n_samples=150
    )

    plot_image_attributions(input_tensor, saliency_attr, ig_attr, lime_attr, true_label, predicted_class)