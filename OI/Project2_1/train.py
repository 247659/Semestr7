import torch
import torch.nn as nn

import copy  # Potrzebne do zapisania najlepszych wag w pamięci


def train_model(model, train_loader, test_loader=None, epochs=100000, lr=0.01, patience=5):
    """
    Zaktualizowana funkcja treningu z mechanizmem Early Stopping.
    Ustalamy duży zapas epok (np. 500), bo i tak trening przerwie się sam.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Zmienne do Early Stopping
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):

        # --- FAZA TRENINGU ---
        model.train()
        total_train_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- FAZA WALIDACJI (Opcjonalna, ale kluczowa dla Early Stopping) ---
        if test_loader is not None:
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for X, y in test_loader:
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(test_loader)
            current_loss = avg_val_loss  # Decyzje podejmujemy na podstawie błędu walidacyjnego

            print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        else:
            current_loss = avg_train_loss
            print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f}")

        # --- MECHANIZM EARLY STOPPING ---
        if current_loss < best_loss:
            # Jest poprawa! Zapisujemy nowy najlepszy wynik i resetujemy licznik
            best_loss = current_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())  # Zapisujemy "zdjęcie" najlepszego mózgu
        else:
            # Brak poprawy. Zwiększamy licznik cierpliwości
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[!] EARLY STOPPING: Brak poprawy od {patience} epok. Zatrzymuję trening.")
                break

    # Po zakończeniu (lub przerwaniu) pętli, wczytujemy najlepsze znalezione wagi
    print(f"Przywracanie najlepszych wag (Best Loss: {best_loss:.4f})")
    model.load_state_dict(best_model_wts)

    return model