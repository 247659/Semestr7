import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import Imagenette
from torchvision import transforms

from models_imagenette.imagenette_2features import ImagenetteCNN2Features
from models_imagenette.imagenette_2features_second import ImagenetteCNN2FeaturesSecond
from models_imagenette.imagenette_cnn import ImagenetteCNN
from utils.metrics import evaluate_model
from utils.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# AUGMENTACJE (Dostosowane do RGB i 128x128)
# ==========================================

transform_no_aug = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_aug_1 = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_aug_2 = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

augmentations = {
    "no_aug": transform_no_aug,
    "aug_1": transform_aug_1,
    "aug_2": transform_aug_2
}

dataset_sizes = {
    "all": None,
    "100": 10,
    "200": 20,
    "1000": 100
}

patience_map = {
    "all": 5,
    "1000": 10,
    "200": 15,
    "100": 20,
}

models_to_test = {
    "ImagenetteCNN": ImagenetteCNN,
    "ImagenetteCNN2Features": ImagenetteCNN2FeaturesSecond
}

results = {}
best_overall_accuracy = {
    "ImagenetteCNN": 0.0,
    "ImagenetteCNN2Features": 0.0
}


def create_balanced_subset(dataset, samples_per_class):
    # Bezpieczne wyciąganie etykiet niezależnie od architektury zbioru PyTorch
    if hasattr(dataset, '_samples'):
        # Nowsze wersje Imagenette ukrywają to pod atrybutem _samples
        targets = np.array([sample[1] for sample in dataset._samples])
    elif hasattr(dataset, 'samples'):
        # ImageFolder z torchvision używa samples
        targets = np.array([sample[1] for sample in dataset.samples])
    elif hasattr(dataset, 'targets'):
        # Klasyczny MNIST i CIFAR używają targets
        targets = np.array(dataset.targets)
    else:
        # Rozwiązanie siłowe - skanuje zbiór, jeśli żadne atrybuty nie istnieją
        print("Skanowanie etykiet zbioru...")
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    indices = []
    unique_classes = np.unique(targets)

    for class_id in unique_classes:
        # Szukamy wszystkich indeksów obrazów należących do danej klasy
        class_indices = np.where(targets == class_id)[0]
        # Losujemy z nich zadaną próbkę
        chosen = np.random.choice(class_indices, samples_per_class, replace=False)
        indices.extend(chosen)

    return Subset(dataset, indices)


for model_name, model_class in models_to_test.items():
    results[model_name] = {}
    print("\n====================================")
    print(f"MODEL: {model_name}")
    print("====================================")

    for aug_name, transform in augmentations.items():
        results[model_name][aug_name] = {}
        print(f"\nAUGMENTACJA: {aug_name}")

        # ==========================================
        # DATASET
        # ==========================================
        # Używamy size="160px" zeby przyspieszyc ładowanie mniejszych obrazow z dysku
        train_dataset = Imagenette(root="./data", split="train", download=True, transform=transform)
        test_dataset = Imagenette(root="./data", split="val", download=True, transform=transform_no_aug)

        total_train_samples_in_full_dataset = len(train_dataset)  # ok. 9469

        for size_name, samples_per_class in dataset_sizes.items():
            print(f"\nRozmiar zbioru: {size_name}")

            if size_name == "all":
                current_dataset = train_dataset
                target_samples = total_train_samples_in_full_dataset
            else:
                current_dataset = create_balanced_subset(train_dataset, samples_per_class)
                target_samples = int(size_name)

            train_loader = DataLoader(
                current_dataset,
                batch_size=32,  # Mniejsze batch_size dla Imagenette!
                shuffle=True
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False
            )

            accuracies = []

            for run in range(10):
                print(f"\nRun {run + 1}/10")

                model = model_class(num_classes=10).to(device)
                feature_model = (model_name == "ImagenetteCNN2Features")
                temp_model_name = f"temp_{model_name}_{aug_name}_{size_name}_run{run}"

                # PRZEKAZUJEMY CALCULATED_EPOCHS
                train_model(
                    model,
                    train_loader,
                    test_loader,
                    temp_model_name,
                    patience=patience_map[size_name],
                    feature_model=feature_model
                )

                accuracy = evaluate_model(model, test_loader, device, feature_model=feature_model)
                accuracies.append(accuracy)
                print(f"Accuracy: {accuracy:.4f}")

                # Zapis najlepszego modelu globalnie
                if accuracy > best_overall_accuracy[model_name]:
                    best_overall_accuracy[model_name] = accuracy
                    best_model_filename = f"BEST_OVERALL_{model_name}_IMAGENETTE.pth"
                    torch.save(model.state_dict(), best_model_filename)
                    print(f"🌟 UWAGA: Nowy rekord! Zapisano model: {best_model_filename}")

                # Sprzątanie dysku
                temp_file_path = f'best_model_{temp_model_name}.pth'
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            # Zapis średnich
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            results[model_name][aug_name][size_name] = {"mean": mean_acc, "std": std_acc}

            print(f"\nWYNIKI -> Mean: {mean_acc:.4f} | Std: {std_acc:.4f}")

print("\n\n====================================")
print("FINAL RESULTS (IMAGENETTE)")
print("====================================")

for model_name in results:
    print(f"\nMODEL: {model_name}")
    for aug_name in results[model_name]:
        print(f"\nAugmentation: {aug_name}")
        for size_name in results[model_name][aug_name]:
            mean_acc = results[model_name][aug_name][size_name]["mean"]
            std_acc = results[model_name][aug_name][size_name]["std"]
            print(f"{size_name}: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")