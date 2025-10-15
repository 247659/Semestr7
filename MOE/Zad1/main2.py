import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog


def solve_and_draw():
    try:
        # Pobranie współczynników funkcji celu
        c1 = float(entry_c1.get())
        c2 = float(entry_c2.get())

        # Pobranie ograniczeń
        A = []
        b = []

        for i in range(4):
            a1_text = entry_a1[i].get()
            a2_text = entry_a2[i].get()
            b_text = entry_b[i].get()

            if a1_text and a2_text and b_text:
                a1 = float(a1_text)
                a2 = float(a2_text)
                bi = float(b_text)
                A.append([a1, a2])
                b.append(bi)

        if not A:
            messagebox.showerror("Błąd", "Musisz podać przynajmniej jedno ograniczenie.")
            return

        # Maksymalizacja → minimalizujemy -Z
        c = [-c1, -c2]

        # Rozwiązanie
        result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")

        if not result.success:
            messagebox.showerror("Brak rozwiązania", "Nie znaleziono rozwiązania dopuszczalnego.")
            return

        x1_opt, x2_opt = result.x
        z_opt = -result.fun

        # Tworzenie wykresu
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(0, 20, 400)

        for i, (a1, a2, bi) in enumerate(zip([r[0] for r in A], [r[1] for r in A], b)):
            if a2 != 0:
                y = (bi - a1 * x) / a2
                ax.plot(x, y, label=f"{a1}x₁ + {a2}x₂ = {bi}")
            else:
                ax.axvline(bi / a1, label=f"x₁ = {bi / a1:.1f}")

        # Punkt optymalny
        ax.plot(x1_opt, x2_opt, 'ro', label=f"Optimum ({x1_opt:.2f}, {x2_opt:.2f})")
        ax.text(x1_opt + 0.4, x2_opt, f"Z = {z_opt:.2f}", color="red")

        # Ustawienia wykresu
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("Rozwiązanie zadania optymalizacji liniowej")
        ax.legend()
        ax.grid(True)

        # Wyświetlenie wyniku
        result_label.config(
            text=f"✅ Wynik:\n"
                 f"x₁ = {x1_opt:.2f}\n"
                 f"x₂ = {x2_opt:.2f}\n"
                 f"Maksimum Z = {z_opt:.2f}"
        )

        # Rysowanie wykresu w GUI
        for widget in frame_plot.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    except ValueError:
        messagebox.showerror("Błąd danych", "Upewnij się, że wszystkie pola zawierają liczby.")


# === GUI ===
root = tk.Tk()
root.title("Optymalizacja liniowa — interaktywny solver")
root.geometry("900x800")

# --- Funkcja celu ---
frame_objective = ttk.LabelFrame(root, text="Funkcja celu (maksymalizacja)")
frame_objective.pack(fill=tk.X, padx=10, pady=5)

ttk.Label(frame_objective, text="Z = ").pack(side=tk.LEFT)
entry_c1 = ttk.Entry(frame_objective, width=6)
entry_c1.pack(side=tk.LEFT)
ttk.Label(frame_objective, text=" * x₁ + ").pack(side=tk.LEFT)
entry_c2 = ttk.Entry(frame_objective, width=6)
entry_c2.pack(side=tk.LEFT)
ttk.Label(frame_objective, text=" * x₂ → maksymalizuj").pack(side=tk.LEFT)

# --- Ograniczenia ---
frame_constraints = ttk.LabelFrame(root, text="Ograniczenia (w formie: a₁*x₁ + a₂*x₂ ≤ b)")
frame_constraints.pack(fill=tk.X, padx=10, pady=5)

entry_a1, entry_a2, entry_b = [], [], []

for i in range(4):
    row = ttk.Frame(frame_constraints)
    row.pack(fill=tk.X, pady=2)
    e1 = ttk.Entry(row, width=6)
    e1.pack(side=tk.LEFT)
    ttk.Label(row, text=" * x₁ + ").pack(side=tk.LEFT)
    e2 = ttk.Entry(row, width=6)
    e2.pack(side=tk.LEFT)
    ttk.Label(row, text=" * x₂ ≤ ").pack(side=tk.LEFT)
    e3 = ttk.Entry(row, width=6)
    e3.pack(side=tk.LEFT)
    entry_a1.append(e1)
    entry_a2.append(e2)
    entry_b.append(e3)

# --- Przycisk i wynik ---
ttk.Button(root, text="Oblicz optimum", command=solve_and_draw).pack(pady=10)
result_label = ttk.Label(root, text="Wprowadź dane i kliknij „Oblicz optimum”.", font=("Arial", 11))
result_label.pack(pady=10)

# --- Wykres ---
frame_plot = ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True)

root.mainloop()
