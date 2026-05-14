import torch

from models_mnist.cnn_classifier_first import CNNClassifier
from models_mnist.cnn_2features_first import CNN2Features
from models_mnist.cnn_classifier_second import CNNClassifierSecond
from models_mnist.cnn_2features_second import CNN2FeaturesSecond

from datasets.data_loaders import get_mnist_loaders

from utils.train import train_model
from utils.plots import (
    plot_accuracy,
    plot_decision_boundaries
)

from utils.metrics import (
    show_confusion_matrix
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def run_experiment1(train_loader, test_loader, model, name, title, feature_model=False, decision=False):

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
            "Granice decyzyjne"
        )

    show_confusion_matrix(
        model,
        test_loader,
        classes=[str(i) for i in range(10)],
        device=device,
        feature_model=feature_model
    )

if __name__ == "__main__":
    train_loader, test_loader = get_mnist_loaders()

    # model_cnn = CNNClassifier().to(device)
    # run_experiment1(
    #     train_loader,
    #     test_loader,
    #     model_cnn,
    #     "CNNFirst",
    #     "MNIST - CNN",
    # )

    # run_experiment1(
    #     train_loader,
    #     test_loader,
    #     model_2f,
    #     "CNN2FeaturesFirst",
    #     "MNIST - CNN 2 Features",
    #     feature_model=True,
    #     confusion=True
    # )

    model_cnn_second = CNNClassifierSecond().to(device)
    run_experiment1(
        train_loader,
        test_loader,
        model_cnn_second,
        "CNN2Second",
        "MNIST - CNN",
    )

    # model_2f_second = CNN2FeaturesSecond().to(device)
    # run_experiment1(
    #     train_loader,
    #     test_loader,
    #     model_2f_second,
    #     "CNN2FeaturesSecond",
    #     "MNIST - CNN 2 Features",
    #     feature_model=True,
    #     decision=True
    # )