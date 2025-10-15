import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog


def solve_and_draw():
    case = case_var.get()
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6, 6))

    # Wspólne ograniczenia
    # x1 + 5x2 <= 37
    # 4x1 + x2 <= 34
    # x1 + x2 >= 3  → -x1 - x2 <= -3
    # x1, x2 >= 0
    A = [[1, 5],
         [4, 1],
         [-1, -1]]
    b = [37, 34, -3]

    if case == "A":
        c = [-50, -20]  # maksymalizacja zysku 50x1 + 20x2 → minimalizujemy -Z
    else:
        # dodajemy 5x1 - x2 >= 3 → -5x1 + x2 <= -3
        A.append([-5, 1])
        b.append(-3)
        c = [50, -20]  # maksymalizacja (-50x1 + 20x2) → minimalizujemy odwrotność

    # Rozwiązanie zadania
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method="highs")

    if not result.success:
        result_label.config(text="❌ Brak rozwiązania!")
        return

    x1_opt, x2_opt = result.x
    z_opt = -result.fun if case == "A" else -result.fun  # w obu przypadkach chcemy maks

    # Rysowanie obszaru
    x = np.linspace(0, 10, 400)
    y1 = (37 - x) / 5
    y2 = 34 - 4*x
    y3 = 3 - x
    ax.plot(x, y1, label="x₁ + 5x₂ = 37")
    ax.plot(x, y2, label="4x₁ + x₂ = 34")
    ax.plot(x, y3, label="x₁ + x₂ = 3")

    if case == "B":
        y4 = 5*x - 3
        ax.plot(x, y4, label="5x₁ - x₂ = 3")

    # Punkt optymalny
    ax.plot(x1_opt, x2_opt, 'ro', label=f"Optimum ({x1_opt:.2f}, {x2_opt:.2f})")
    ax.text(x1_opt + 0.3, x2_opt, f"Z = {z_opt:.2f} zł", color="red")

    # Wygląd wykresu
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("x₁ (urządzenia u₁)")
    ax.set_ylabel("x₂ (urządzenia u₂)")
    ax.set_title(f"Przypadek {case} — obszar dopuszczalny i optimum")
    ax.legend()
    ax.grid(True)

    # Aktualizacja pola wyników
    result_label.config(
        text=f"✅ Wynik:\n"
             f"x₁ = {x1_opt:.2f}\n"
             f"x₂ = {x2_opt:.2f}\n"
             f"Zysk/strata = {z_opt:.2f} zł"
    )

    # Umieszczenie wykresu w oknie
    for widget in frame_plot.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# === GUI ===
root = tk.Tk()
root.title("Optymalizacja produkcji urządzeń")
root.geometry("850x750")

frame_controls = tk.Frame(root)
frame_controls.pack(side=tk.TOP, pady=10)

ttk.Label(frame_controls, text="Wybierz przypadek:").pack(side=tk.LEFT, padx=5)
case_var = tk.StringVar(value="A")

ttk.Radiobutton(frame_controls, text="A — zysk 50/20", variable=case_var, value="A").pack(side=tk.LEFT, padx=5)
ttk.Radiobutton(frame_controls, text="B — czwarty surowiec", variable=case_var, value="B").pack(side=tk.LEFT, padx=5)

ttk.Button(frame_controls, text="Oblicz", command=solve_and_draw).pack(side=tk.LEFT, padx=10)

result_label = ttk.Label(root, text="Kliknij „Oblicz”, aby zobaczyć wynik", font=("Arial", 11))
result_label.pack(pady=10)

frame_plot = tk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True)

root.mainloop()
