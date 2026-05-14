import torch

from models_imagenette.imagenette_2features import ImagenetteCNN2Features
from models_imagenette.imagenette_2features_second import ImagenetteCNN2FeaturesSecond
from models_imagenette.imagenette_cnn import ImagenetteCNN
from models_imagenette.imagenette_cnn_second import ImagenetteCNNSecond

from datasets.data_loaders import (
    get_imagenette_loaders
)

from utils.train import train_model

from utils.plots import plot_accuracy, plot_decision_boundaries

from utils.metrics import (
    show_confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def run_experiment1(train_loader, test_loader, class_names, model, name, title, feature_model=False, decision=False):

    train_acc_2f, test_acc_2f = train_model(
        model,
        train_loader,
        test_loader,
        name,
        feature_model=feature_model
    )

    plot_accuracy(
        train_acc_2f,
        test_acc_2f,
        title
    )

    if decision:
        plot_decision_boundaries(
            model,
            train_loader,
            test_loader,
            device,
            "Granice decyzyjne",
            class_names=class_names
        )

    show_confusion_matrix(
        model,
        test_loader,
        classes=class_names,
        device=device,
        feature_model=feature_model
    )

if __name__ == "__main__":
    train_loader, test_loader, original_classes = get_imagenette_loaders()
    classes = [synonyms[0] for synonyms in original_classes]

    # model_cnn = ImagenetteCNN().to(device)
    # run_experiment1(
    #     train_loader,
    #     test_loader,
    #     classes,
    #     model_cnn,
    #     "ImagenetteCNNFirst",
    #     "Imagenette - CNN",
    # )

    # model_2f = ImagenetteCNN2Features().to(device)
    # run_experiment1(
    #     train_loader,
    #     test_loader,
    #     classes,
    #     model_2f,
    #     "ImagenetteCNN2Features",
    #     "Imagenette - CNN 2 Features",
    #     True,
    #     True
    # )

    model_cnn_second = ImagenetteCNNSecond().to(device)
    run_experiment1(
        train_loader,
        test_loader,
        classes,
        model_cnn_second,
        "ImagenetteCNNSecond",
            "Imagenette - CNN",
    )

    model_2f_second = ImagenetteCNN2FeaturesSecond().to(device)
    run_experiment1(
        train_loader,
        test_loader,
        classes,
        model_2f_second,
        "ImagenetteCNN2FeaturesSecond",
        "Imagenette - CNN - 2features",
        True,
        True
    )
