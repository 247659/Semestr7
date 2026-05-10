import cv2
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
from torchvision import transforms


# =========================
# FLATTEN
# =========================
def flatten_images(images):
    return images.reshape(images.shape[0], -1)


# =========================
# FEATURES
# =========================

def extract_features_method_1_1(image, threshold=0.2):
    image = image.squeeze()
    binary = image > threshold

    active_sum = np.sum(binary)

    upper = np.sum(binary[:14, :])
    lower = np.sum(binary[14:, :])

    balance = upper - lower

    return np.array([active_sum, balance])


def extract_features_method_2_1(image, threshold=0.2):
    image = image.squeeze()
    binary = image > threshold
    filled = ndimage.binary_fill_holes(binary)
    holes = filled & ~binary
    labeled, num_holes = ndimage.label(holes)
    rows, cols = np.where(binary)

    if len(cols) == 0:
        var_x = 0
    else:
        var_x = np.var(cols)

    return np.array([num_holes, var_x])


def extract_features_method_2_2(image, threshold=0.2):
    image = image.squeeze()

    binary = image > threshold
    image_float = image.astype(np.float32)

    # 5. SYMETRIA PIONOWA (Lewo vs Prawo)
    left = image_float[:, :14]
    right = image_float[:, 14:]
    right_flipped = np.fliplr(right)  # Odbicie lustrzane prawej strony
    diff_v = np.abs(left - right_flipped)
    # Wzór: 1 - (błąd / całkowita_masa), 1e-8 zapobiega dzieleniu przez zero
    sym_v = 1.0 - (np.sum(diff_v) / (np.sum(left + right_flipped) + 1e-8))

    # 6. SYMETRIA POZIOMA (Góra vs Dół)
    top = image_float[:14, :]
    bottom = image_float[14:, :]
    bottom_flipped = np.flipud(bottom)  # Odbicie lustrzane dołu
    diff_h = np.abs(top - bottom_flipped)
    sym_h = 1.0 - (np.sum(diff_h) / (np.sum(top + bottom_flipped) + 1e-8))
    # ---------------------------------------------------------
    # 2. CECHY Z METODY 1_1 (2 cechy: suma pikseli, balans)
    # ---------------------------------------------------------
    features_1_1 = extract_features_method_1_1(image, threshold)

    # ---------------------------------------------------------
    # 3. CECHY Z METODY 2_1 (2 cechy: liczba pętli, wariancja X)
    # ---------------------------------------------------------
    features_2_1 = extract_features_method_2_1(image, threshold)

    # ---------------------------------------------------------
    # 4. POŁĄCZENIE WEKTORÓW
    # ---------------------------------------------------------
    combined_features = np.concatenate([[sym_v, sym_h], features_1_1, features_2_1])

    return combined_features

# =========================
# PCA (NA CAŁYM ZBIORZE)
# =========================
def extract_features_method_1_2(X, n_components=20):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca


# =========================
# PIPELINE
# =========================
def save_all_features():

    transform = transforms.ToTensor()

    mnist_train = MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = MNIST(root="./data", train=False, download=True, transform=transform)

    X1_train, X2_train, X3_train, Y_train = [], [], [], []
    X1_test, X2_test, X3_test, Y_test = [], [], [], []

    # =========================
    # TRAIN
    # =========================
    for img, label in mnist_train:
        img = img.numpy()

        X1_train.append(extract_features_method_1_1(img))
        X2_train.append(extract_features_method_2_1(img))
        X3_train.append(extract_features_method_2_2(img))
        Y_train.append(label)

    # =========================
    # TEST
    # =========================
    for img, label in mnist_test:
        img = img.numpy()

        X1_test.append(extract_features_method_1_1(img))
        X2_test.append(extract_features_method_2_1(img))
        X3_test.append(extract_features_method_2_2(img))
        Y_test.append(label)

    # =========================
    # PCA (osobno na flatten TRAIN)
    # =========================
    train_flat = flatten_images(mnist_train.data.numpy())
    test_flat = flatten_images(mnist_test.data.numpy())

    X_pca_train, pca = extract_features_method_1_2(train_flat, 20)
    X_pca_test = pca.transform(test_flat)

    # =========================
    # ZAPIS
    # =========================

    np.save("features/X1_train.npy", np.array(X1_train))
    np.save("features/X2_train.npy", np.array(X2_train))
    np.save("features/X3_train.npy", np.array(X3_train))
    np.save("features/y_train.npy", np.array(Y_train))

    np.save("features/X1_test.npy", np.array(X1_test))
    np.save("features/X2_test.npy", np.array(X2_test))
    np.save("features/X3_test.npy", np.array(X3_test))
    np.save("features/y_test.npy", np.array(Y_test))

    np.save("features/X_pca_train.npy", X_pca_train)
    np.save("features/X_pca_test.npy", X_pca_test)

    np.save("features/X_flat_train.npy", train_flat)
    np.save("features/X_flat_test.npy", test_flat)


if __name__ == "__main__":
    save_all_features()
