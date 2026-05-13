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
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")

    plt.show()