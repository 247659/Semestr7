import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog


def show_help():
    """Okno pomocy i opisu programu."""
    help_text = (
        "📘 **Instrukcja obsługi programu**\n\n"
        "Program rozwiązuje zadania *programowania liniowego* w postaci:\n"
        "Maksymalizuj: Z = c₁·x₁ + c₂·x₂\n"
        "przy zadanych ograniczeniach liniowych (≤, ≥, =)\n\n"
        "👉 Etapy pracy:\n"
        "1️⃣ Wprowadź współczynniki funkcji celu.\n"
        "2️⃣ Wprowadź współczynniki i znaki ograniczeń.\n"
        "3️⃣ Wybierz tryb rozwiązania:\n"
        "   • 'Krok po kroku' – pokazuje kolejne etapy obliczeń,\n"
        "   • 'Ostateczny wynik' – pokazuje tylko końcowy wynik.\n"
        "4️⃣ Kliknij 'Oblicz optimum'.\n\n"
        "📈 Program wyświetli wykres ograniczeń i punkt optimum.\n"
        "Wartość Z oraz współrzędne x₁, x₂ pojawią się poniżej.\n\n"
        "Program korzysta z metody Simplex (biblioteka SciPy, metoda 'HiGHS').\n"
    )

    messagebox.showinfo("Pomoc — Optymalizacja liniowa", help_text)


def solve_and_draw(step_by_step=False):
    try:
        c1 = float(entry_c1.get())
        c2 = float(entry_c2.get())

        A_ub, b_ub = [], []
        A_eq, b_eq = [], []

        for i in range(4):
            a1_text = entry_a1[i].get()
            a2_text = entry_a2[i].get()
            b_text = entry_b[i].get()
            sign = sign_var[i].get()

            if a1_text and a2_text and b_text:
                a1 = float(a1_text)
                a2 = float(a2_text)
                bi = float(b_text)

                if sign == "≤":
                    A_ub.append([a1, a2])
                    b_ub.append(bi)
                elif sign == "≥":
                    A_ub.append([-a1, -a2])
                    b_ub.append(-bi)
                elif sign == "=":
                    A_eq.append([a1, a2])
                    b_eq.append(bi)

        if not (A_ub or A_eq):
            messagebox.showerror("Błąd", "Musisz podać przynajmniej jedno ograniczenie.")
            return

        # Maksymalizacja → minimalizujemy -Z
        c = [-c1, -c2]

        # Rozwiązanie
        result = linprog(
            c,
            A_ub=A_ub if A_ub else None,
            b_ub=b_ub if b_ub else None,
            A_eq=A_eq if A_eq else None,
            b_eq=b_eq if b_eq else None,
            bounds=(0, None),
            method="highs"
        )

        if not result.success:
            messagebox.showerror("Brak rozwiązania", "Nie znaleziono rozwiązania dopuszczalnego.")
            return

        x1_opt, x2_opt = result.x
        z_opt = -result.fun

        # Tworzenie wykresu
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.linspace(0, 20, 400)

        # Ograniczenia
        for i in range(4):
            if entry_a1[i].get() and entry_a2[i].get() and entry_b[i].get():
                a1 = float(entry_a1[i].get())
                a2 = float(entry_a2[i].get())
                bi = float(entry_b[i].get())
                sign = sign_var[i].get()

                if a2 != 0:
                    y = (bi - a1 * x) / a2
                    label = f"{a1}x₁ + {a2}x₂ {sign} {bi}"
                    ax.plot(x, y, label=label)
                else:
                    ax.axvline(bi / a1, label=f"x₁ {sign} {bi / a1:.1f}")

        # Punkt optymalny
        ax.plot(x1_opt, x2_opt, 'ro', label=f"Optimum ({x1_opt:.2f}, {x2_opt:.2f})")
        ax.text(x1_opt + 0.4, x2_opt, f"Z = {z_opt:.2f}", color="red")

        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title("Rozwiązanie zadania optymalizacji liniowej")
        ax.legend()
        ax.grid(True)

        # Wyświetlenie wyniku
        steps = []
        if step_by_step:
            steps.append("1️⃣ Wczytano funkcję celu: Z = %.2f·x₁ + %.2f·x₂ → maks" % (c1, c2))
            steps.append(f"2️⃣ Zebrano {len(A_ub) + len(A_eq)} ograniczeń")
            steps.append("3️⃣ Przekształcono na postać minimalizacji -Z")
            steps.append("4️⃣ Użyto solvera Simplex (HiGHS)")
            steps.append("5️⃣ Otrzymano rozwiązanie:")
        steps.append(f"x₁ = {x1_opt:.2f}")
        steps.append(f"x₂ = {x2_opt:.2f}")
        steps.append(f"Z = {z_opt:.2f}")

        result_label.config(text="\n".join(steps))

        # Rysowanie wykresu
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
root.geometry("950x850")

# --- Menu Pomocy ---
menubar = tk.Menu(root)
help_menu = tk.Menu(menubar, tearoff=0)
help_menu.add_command(label="Instrukcja obsługi", command=show_help)
menubar.add_cascade(label="Pomoc", menu=help_menu)
root.config(menu=menubar)

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
frame_constraints = ttk.LabelFrame(root, text="Ograniczenia (a₁*x₁ + a₂*x₂ ≤, ≥ lub = b)")
frame_constraints.pack(fill=tk.X, padx=10, pady=5)

entry_a1, entry_a2, entry_b, sign_var = [], [], [], []

for i in range(4):
    row = ttk.Frame(frame_constraints)
    row.pack(fill=tk.X, pady=2)
    e1 = ttk.Entry(row, width=6)
    e1.pack(side=tk.LEFT)
    ttk.Label(row, text=" * x₁ + ").pack(side=tk.LEFT)
    e2 = ttk.Entry(row, width=6)
    e2.pack(side=tk.LEFT)
    ttk.Label(row, text=" * x₂ ").pack(side=tk.LEFT)

    var = tk.StringVar(value="≤")
    sign_box = ttk.Combobox(row, textvariable=var, values=["≤", "=", "≥"], width=3, state="readonly")
    sign_box.pack(side=tk.LEFT, padx=2)
    e3 = ttk.Entry(row, width=6)
    e3.pack(side=tk.LEFT)

    entry_a1.append(e1)
    entry_a2.append(e2)
    entry_b.append(e3)
    sign_var.append(var)

# --- Tryby rozwiązania ---
frame_mode = ttk.LabelFrame(root, text="Tryb rozwiązania")
frame_mode.pack(fill=tk.X, padx=10, pady=5)

ttk.Button(frame_mode, text="🔍 Krok po kroku", command=lambda: solve_and_draw(True)).pack(side=tk.LEFT, padx=10, pady=5)
ttk.Button(frame_mode, text="⚡ Ostateczny wynik", command=lambda: solve_and_draw(False)).pack(side=tk.LEFT, padx=10, pady=5)

# --- Wynik ---
result_label = ttk.Label(root, text="Wprowadź dane i wybierz tryb rozwiązania.", font=("Arial", 11))
result_label.pack(pady=10)

# --- Wykres ---
frame_plot = ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True)

root.mainloop()
