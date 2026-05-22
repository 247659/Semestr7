import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms

from models_mnist.cnn_2features_first import CNN2Features
from models_mnist.cnn_classifier_second import CNNClassifierSecond
from utils.metrics import evaluate_model
from utils.train import train_model

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


transform_aug_1 = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_aug_2 = transforms.Compose([
    transforms.RandomAffine(
        degrees=15,
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
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
    "200": 15,
    "100": 20,
    "1000": 10,
}

models_to_test = {
    "CNNClassifier": CNNClassifierSecond,
    "CNN2Features": CNN2Features
}

results = {}
best_overall_accuracy = {
    "CNNClassifier": 0.0,
    "CNN2Features": 0.0
}

def create_balanced_subset(dataset, samples_per_class):
    targets = np.array(dataset.targets)
    indices = []
    for class_id in range(10):

        class_indices = np.where(
            targets == class_id
        )[0]

        chosen = np.random.choice(
            class_indices,
            samples_per_class,
            replace=False
        )

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

        train_dataset = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transform_no_aug
        )

        for size_name, samples_per_class in dataset_sizes.items():
            print(f"\nRozmiar zbioru: {size_name}")

            if size_name == "all":
                current_dataset = train_dataset

            else:
                current_dataset = create_balanced_subset(
                    train_dataset,
                    samples_per_class
                )

            train_loader = DataLoader(
                current_dataset,
                batch_size=64,
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

                model = model_class().to(device)

                feature_model = (
                    model_name == "CNN2Features"
                )

                temp_model_name = f"temp_{model_name}_{aug_name}_{size_name}_run{run}"

                train_model(
                    model,
                    train_loader,
                    test_loader,
                    temp_model_name,
                    patience=patience_map[size_name],
                    feature_model=feature_model
                )

                accuracy = evaluate_model(
                    model,
                    test_loader,
                    device,
                    feature_model=feature_model
                )

                accuracies.append(accuracy)

                print(f"Accuracy: {accuracy:.4f}")

                if accuracy > best_overall_accuracy[model_name]:
                    best_overall_accuracy[model_name] = accuracy
                    best_model_filename = f"BEST_OVERALL_{model_name}_MNIST.pth"
                    torch.save(model.state_dict(), best_model_filename)

                temp_file_path = f'best_model_{temp_model_name}.pth'
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            results[model_name][aug_name][size_name] = {
                "mean": mean_acc,
                "std": std_acc
            }

            print("\nWYNIKI:")
            print(f"Mean Accuracy: {mean_acc:.4f}")
            print(f"Std Accuracy: {std_acc:.4f}")


print("\n\n====================================")
print("FINAL RESULTS")
print("====================================")

for model_name in results:
    print(f"\nMODEL: {model_name}")
    for aug_name in results[model_name]:
        print(f"\nAugmentation: {aug_name}")
        for size_name in results[model_name][aug_name]:
            mean_acc = results[model_name][aug_name][size_name]["mean"]
            std_acc = results[model_name][aug_name][size_name]["std"]
            print(
                f"{size_name}: "
                f"{mean_acc:.4f} ± {std_acc:.4f}"
            )