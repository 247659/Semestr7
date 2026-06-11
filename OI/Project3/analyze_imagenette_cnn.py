import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision import transforms
from skimage.segmentation import slic

from captum.attr import Saliency, IntegratedGradients, Lime
from captum.attr import visualization as viz

from models.imagenette_cnn import ImagenetteCNN

# --- 1. SŁOWNIK KLAS IMAGENETTE ---
# Klasy ułożone są zgodnie z domyślnym sortowaniem PyTorch (po WordNet ID)
IMAGENETTE_CLASSES = [
    'Tench',
    'English Springer',
    'Cassette player',
    'Chain saw',
    'Church',
    'French horn',
    'Garbage truck',
    'Gas pump',
    'Golf ball',
    'Parachute'
]


# ==========================================
# 1. FUNKCJA DO OCENY DOKŁADNOŚCI MODELU
# ==========================================
def evaluate_model_accuracy(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()

    return (correct_predictions / total_predictions) * 100


# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================
print("Pobieranie i przygotowanie danych Imagenette...")

transform_no_aug = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = Imagenette(root="./data", split="val", download=True, transform=transform_no_aug)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 3. DEFINICJA I ŁADOWANIE MODELU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImagenetteCNN(num_classes=10).to(device)

try:
    model.load_state_dict(torch.load('trained_models/BEST_OVERALL_ImagenetteCNN_IMAGENETTE.pth', map_location=device))
except FileNotFoundError:
    print("[BŁĄD] Nie znaleziono wag modelu. Sprawdź nazwę pliku .pth.")
    exit()

# ==========================================
# 4. ETAP 1: GLOBALNA OCENA DOKŁADNOŚCI
# ==========================================
print("\n=== ETAP 1: OCENA DOKŁADNOŚCI MODELU ===")
acc = evaluate_model_accuracy(model, test_loader, device)
print(f"Dokładność CNN na zbiorze walidacyjnym Imagenette: {acc:.2f}%\n")

# ==========================================
# 5. ETAP 2: LOKALNE OBJAŚNIANIE I WIZUALIZACJE
# ==========================================
print("=== ETAP 2: GENEROWANIE WIZUALIZACJI XAI ===")
model.eval()

saliency = Saliency(model)
ig = IntegratedGradients(model)
lime = Lime(model)


def unnormalize(tensor):
    return tensor * 0.5 + 0.5


normalized_black_pixel = -1.0


def plot_image_attributions(input_tensor, saliency_attr, ig_attr, lime_attr, true_name, pred_name):
    """Generuje panel z nałożonymi mapami ciepła. Przyjmuje stringi z nazwami klas."""

    unnorm_img = unnormalize(input_tensor).squeeze(0).cpu().detach().numpy()
    original_image = np.transpose(unnorm_img, (1, 2, 0))

    s_attr = np.transpose(saliency_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    i_attr = np.transpose(ig_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    l_attr = np.transpose(lime_attr.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

    title_str = f"Cel: {true_name}  |  Predykcja: {pred_name}"
    print(f"Rysowanie... {title_str}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title_str, fontsize=14, fontweight='bold')

    viz.visualize_image_attr(
        s_attr, original_image, method="original_image", title="Oryginał",
        plt_fig_axis=(fig, axes[0]), use_pyplot=False
    )

    viz.visualize_image_attr(
        s_attr, original_image, method="blended_heat_map", sign="absolute_value",
        outlier_perc=2, title="Saliency", cmap="PiYG",
        plt_fig_axis=(fig, axes[1]), use_pyplot=False
    )

    viz.visualize_image_attr(
        i_attr, original_image, method="blended_heat_map", sign="all",
        outlier_perc=2, title="Integrated Gradients", cmap="PiYG",
        plt_fig_axis=(fig, axes[2]), use_pyplot=False
    )

    viz.visualize_image_attr(
        l_attr, original_image, method="blended_heat_map", sign="all",
        outlier_perc=2, title="LIME (SLIC)", cmap="PiYG",
        plt_fig_axis=(fig, axes[3]), use_pyplot=False
    )

    plt.tight_layout()
    plt.show()


indices_to_analyze = [0, 689, 782]

# Wyodrębnienie i analiza znalezionych próbek
for i in indices_to_analyze:
    input_tensor, true_label = test_dataset[i]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    input_tensor.requires_grad_()
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

    # Mapowanie indeksów na nazwy klas!
    true_name = IMAGENETTE_CLASSES[true_label]
    pred_name = IMAGENETTE_CLASSES[predicted_class]

    # Saliency
    saliency_attr = saliency.attribute(input_tensor, target=predicted_class)

    # Integrated Gradients
    baseline_image = torch.full_like(input_tensor, normalized_black_pixel).to(device)
    ig_attr = ig.attribute(input_tensor, baselines=baseline_image, target=predicted_class, n_steps=50)

    # LIME z segmentacją SLIC
    image_for_slic = unnormalize(input_tensor).squeeze().cpu().detach().numpy()
    image_for_slic = np.transpose(image_for_slic, (1, 2, 0))

    segments = slic(image_for_slic, n_segments=100, compactness=10, start_label=0)
    feature_mask = torch.tensor(segments).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)

    lime_attr = lime.attribute(
        input_tensor, target=predicted_class, feature_mask=feature_mask, n_samples=200
    )

    plot_image_attributions(input_tensor, saliency_attr, ig_attr, lime_attr, true_name, pred_name)