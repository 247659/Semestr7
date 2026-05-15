import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


def show_confusion_matrix(
        model,
        loader,
        classes,
        device,
        feature_model=False):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            if feature_model:
                outputs, _ = model(images)
            else:
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(
                predicted.cpu().numpy()
            )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    accuracy = accuracy_score(all_labels, all_predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )
    plt.figure(figsize=(10, 10))
    disp.plot(cmap="Blues", xticks_rotation='vertical')
    plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device, feature_model=False):
    """
    Funkcja przepuszcza zbiór testowy przez model i zwraca dokładność (accuracy).
    """
    model.eval()  # BARDZO WAŻNE: przełączenie w tryb ewaluacji

    correct = 0
    total = 0

    with torch.no_grad():  # Wyłączenie śledzenia gradientów (oszczędza pamięć i czas)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            if feature_model:
                outputs, _ = model(images)
            else:
                outputs = model(images)

            # Pobranie indeksu z największym prawdopodobieństwem (przewidywana klasa)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy