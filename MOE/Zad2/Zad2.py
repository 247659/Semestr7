import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog

# --- Klasa wizualizująca kroki Simplexa ---
class SimplexVisualizer(tk.Toplevel):
    def __init__(self, root, history):
        super().__init__(root)
        self.title("Kroki metody Simplex")
        self.geometry("800x600")
        self.history = history
        self.index = 0

        # Nagłówek
        self.label = ttk.Label(self, text=f"Iteracja 1 z {len(history)}", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)

        # Tabela
        self.tree = ttk.Treeview(self, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Przyciski nawigacji
        nav = ttk.Frame(self)
        nav.pack(pady=5)
        ttk.Button(nav, text="⟨ Poprzedni", command=self.prev).pack(side=tk.LEFT, padx=10)
        ttk.Button(nav, text="Następny ⟩", command=self.next).pack(side=tk.LEFT, padx=10)

        self.show_iteration(0)

    def show_iteration(self, i):
        self.index = i
        tableau, basis = self.history[i]
        rows, cols = tableau.shape
        headers = [f"x{j+1}" for j in range(cols-1)] + ["RHS"]

        # usuń poprzednią tabelę
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = headers
        for h in headers:
            self.tree.heading(h, text=h)
            self.tree.column(h, anchor="center", width=80)

        # wiersze
        for r in range(rows-1):
            self.tree.insert("", "end", values=[f"{v:.3f}" for v in tableau[r]])
        self.tree.insert("", "end", values=["—"]*(cols-1) + ["—"])
        self.tree.insert("", "end", values=[f"{v:.3f}" for v in tableau[-1]])

        self.label.config(text=f"Iteracja {i+1} z {len(self.history)}  |  Baza: {', '.join(['x'+str(b+1) for b in basis])}")

    def next(self):
        if self.index < len(self.history)-1:
            self.show_iteration(self.index + 1)

    def prev(self):
        if self.index > 0:
            self.show_iteration(self.index - 1)


# --- Zmieniona implementacja Simplex z zapisem kroków ---
def simplex_with_steps(c, A, b, eps=1e-9):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[m, :n] = -c
    basis = [n + i for i in range(m)]

    history = [(tableau.copy(), basis.copy())]

    while True:
        obj_row = tableau[m, :-1]
        if np.all(obj_row >= -eps):
            break
        entering = np.argmin(obj_row)
        col = tableau[:m, entering]
        if np.all(col <= eps):
            raise Exception("Problem nieograniczony.")
        ratios = np.array([(tableau[i, -1] / col[i]) if col[i] > eps else np.inf for i in range(m)])
        leaving_row = int(np.argmin(ratios))
        pivot = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot
        for i in range(m + 1):
            if i != leaving_row:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving_row, :]
        basis[leaving_row] = entering
        history.append((tableau.copy(), basis.copy()))

    x = np.zeros(n + m)
    for i in range(m):
        idx = basis[i]
        if idx < n + m:
            x[idx] = tableau[i, -1]
    opt_value = tableau[m, -1]
    return opt_value, x[:n], history


# --- Funkcje geometryczne (bez zmian) ---
def intersection_of_lines(l1, l2, eps=1e-9):
    a1, a2, b1 = l1
    c1, c2, b2 = l2
    det = a1*c2 - a2*c1
    if abs(det) < eps: return None
    x = (b1*c2 - a2*b2)/det
    y = (a1*b2 - b1*c1)/det
    return (x, y)

def point_satisfies_all(x, y, ineqs, tol=1e-7):
    for A, b in ineqs:
        if A[0]*x + A[1]*y - b > tol: return False
    if x < -tol or y < -tol: return False
    return True


# --- GUI ---
root = tk.Tk()
root.title("Simplex – wizualizacja kroków")
root.geometry("1020x900")

def show_help():
    messagebox.showinfo("Pomoc", "Program rozwiązuje zadania optymalizacji liniowej metodą Simplex (z wizualizacją).")

frame_top = ttk.Frame(root)
frame_top.pack(fill=tk.X, padx=10, pady=5)
ttk.Button(frame_top, text="Pomoc", command=show_help).pack(side=tk.LEFT)

# --- Funkcja celu ---
frame_objective = ttk.LabelFrame(root, text="Funkcja celu")
frame_objective.pack(fill=tk.X, padx=10, pady=5)
ttk.Label(frame_objective,text="Z = ").pack(side=tk.LEFT)
entry_c1 = ttk.Entry(frame_objective,width=8); entry_c1.pack(side=tk.LEFT)
ttk.Label(frame_objective,text=" * x₁ + ").pack(side=tk.LEFT)
entry_c2 = ttk.Entry(frame_objective,width=8); entry_c2.pack(side=tk.LEFT)
ttk.Label(frame_objective,text=" * x₂ → maksymalizuj").pack(side=tk.LEFT)

# --- Liczba ograniczeń ---
frame_num_constraints = ttk.Frame(root)
frame_num_constraints.pack(fill=tk.X, padx=10, pady=5)
ttk.Label(frame_num_constraints,text="Liczba ograniczeń:").pack(side=tk.LEFT)
num_constraints_var = tk.IntVar(value=3)
spin_constraints = ttk.Spinbox(frame_num_constraints, from_=1, to=10, textvariable=num_constraints_var, width=4)
spin_constraints.pack(side=tk.LEFT, padx=5)

# --- Ograniczenia ---
frame_constraints = ttk.LabelFrame(root, text="Ograniczenia (a₁*x₁ + a₂*x₂ ≤ b)")
frame_constraints.pack(fill=tk.X,padx=10,pady=5)
entry_a1, entry_a2, entry_b = [], [], []

def build_constraints(n):
    for w in frame_constraints.winfo_children(): w.destroy()
    entry_a1.clear(); entry_a2.clear(); entry_b.clear()
    for i in range(n):
        row = ttk.Frame(frame_constraints); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=f"{i+1}. ").pack(side=tk.LEFT)
        e1 = ttk.Entry(row, width=8); e1.pack(side=tk.LEFT)
        ttk.Label(row, text=" * x₁ + ").pack(side=tk.LEFT)
        e2 = ttk.Entry(row, width=8); e2.pack(side=tk.LEFT)
        ttk.Label(row, text=" * x₂ ≤ ").pack(side=tk.LEFT)
        e3 = ttk.Entry(row, width=8); e3.pack(side=tk.LEFT)
        entry_a1.append(e1); entry_a2.append(e2); entry_b.append(e3)

build_constraints(num_constraints_var.get())
num_constraints_var.trace_add("write", lambda *a: build_constraints(num_constraints_var.get()))

# --- Przycisk rozwiązania ---
frame_plot = ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True)
result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"), foreground="blue")
result_label.pack(pady=10)

def solve():
    try:
        c1, c2 = float(entry_c1.get()), float(entry_c2.get())
        A, b = [], []
        for i in range(len(entry_a1)):
            A.append([float(entry_a1[i].get()), float(entry_a2[i].get())])
            b.append(float(entry_b[i].get()))
        A, b = np.array(A), np.array(b)

        val, x, hist = simplex_with_steps([c1,c2], A, b)
        x1, x2 = x
        result_label.config(text=f"Optimum: x₁={x1:.2f}, x₂={x2:.2f}, Z={val:.2f}")

        ttk.Button(root, text="Pokaż kroki Simplexa", command=lambda: SimplexVisualizer(root, hist)).pack(pady=5)

        # wykres
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6,6))
        x_line = np.linspace(0, max(10, x1*2), 300)
        for i in range(len(A)):
            y_line = (b[i]-A[i,0]*x_line)/A[i,1]
            ax.plot(x_line, y_line, label=f"{A[i,0]}x₁+{A[i,1]}x₂={b[i]}")
        ax.plot(x1, x2, 'ro', label='Optimum')
        ax.text(x1+0.3, x2, f"Z={val:.2f}", color='red')
        ax.legend()
        ax.set_xlim(0, max(10, x1*2)); ax.set_ylim(0, max(10, x2*2))
        for w in frame_plot.winfo_children(): w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Błąd", str(e))

ttk.Button(root, text="Oblicz", command=solve).pack(pady=10)
root.mainloop()
