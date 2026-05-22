import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, path='best_model.pth'):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] Brak poprawy ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

def train_model(
        model,
        train_loader,
        test_loader,
        name,
        epochs=200,
        lr=0.001,
        feature_model=False,
        patience=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_history = []
    test_acc_history = []

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if feature_model:
                outputs, _ = model(images)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = correct_train / total_train
        train_acc_history.append(train_acc)

        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                if feature_model:
                    outputs, _ = model(images)
                else:
                    outputs = model(images)

                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = test_loss / total_test
        test_acc = correct_test / total_test
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

        early_stopping(test_loss, model, path=f'best_model_{name}.pth')

        if early_stopping.early_stop:
            print(f"\nUruchomiono Early Stopping! Trening zatrzymany po {epoch + 1} epokach.")
            break

        print("--------------------------------")

    model.load_state_dict(torch.load(f'best_model_{name}.pth'))
    print("Wczytano wagi z najlepszej epoki.")

    return train_acc_history, test_acc_history