import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------
# Simplex z zapisem kroków
# -------------------------
def simplex_with_steps(c, A, b, eps=1e-9):
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    # standardowa macierz tableau z dodatkową macierzą tożsamości i wektorem RHS
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[m, :n] = -c
    basis = [n + i for i in range(m)]

    history = [(tableau.copy(), basis.copy())]

    while True:
        obj_row = tableau[m, :-1]
        # jeśli wszystkie współczynniki >= 0 (dla maksymalizacji) — optimum
        if np.all(obj_row >= -eps):
            break

        entering = int(np.argmin(obj_row))
        col = tableau[:m, entering]

        # jeśli kolumna nie ma dodatnich elementów => nieograniczony
        if np.all(col <= eps):
            raise Exception("Problem nieograniczony.")

        # stosunek RHS / kolumna dla kolumn dodatnich
        ratios = np.array([(tableau[i, -1] / col[i]) if col[i] > eps else np.inf for i in range(m)])
        leaving_row = int(np.argmin(ratios))
        pivot = tableau[leaving_row, entering]
        tableau[leaving_row, :] /= pivot

        for i in range(m + 1):
            if i != leaving_row:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving_row, :]

        basis[leaving_row] = entering
        history.append((tableau.copy(), basis.copy()))

    # odczyt rozwiązania
    x = np.zeros(n + m)
    for i in range(m):
        idx = basis[i]
        if idx < n + m:
            x[idx] = tableau[i, -1]
    opt_value = tableau[m, -1]
    return opt_value, x[:n], history

# -------------------------
# Narzędzia geometryczne
# -------------------------
def intersection_of_lines(l1, l2, eps=1e-9):
    # l: (a1, a2, b) meaning a1*x + a2*y = b
    a1, a2, b1 = l1
    c1, c2, b2 = l2
    det = a1*c2 - a2*c1
    if abs(det) < eps:
        return None
    x = (b1*c2 - a2*b2)/det
    y = (a1*b2 - b1*c1)/det
    return (x, y)

def point_satisfies_all(pt, ineqs, tol=1e-7):
    x, y = pt
    if x < -tol or y < -tol:
        return False
    for A, b in ineqs:
        if A[0]*x + A[1]*y - b > tol:
            return False
    return True

def monotone_chain_convex_hull(points):
    # points: list of (x,y)
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

# -------------------------
# GUI
# -------------------------
root = tk.Tk()
root.title("Simplex – wizualizacja kroków (A3)")
root.geometry("1100x900")

button_show_steps = None  # globalna zmienna dla przycisku "Pokaż kroki"

def show_help():
    messagebox.showinfo("Pomoc",
        "Program rozwiązuje zadania optymalizacji liniowej metodą Simplex (2 zmienne).\n"
        "- Dodaj ograniczenia (≤, ≥, =). Dodawaj/usuń wiersze.\n"
        "- Rysowany jest obszar dopuszczalny, punkt optymalny i (jeśli istnieje) odcinek optymalnych rozwiązań."
    )

# Górny panel
frame_top = ttk.Frame(root)
frame_top.pack(fill=tk.X, padx=10, pady=5)
ttk.Button(frame_top, text="Pomoc", command=show_help).pack(side=tk.LEFT)

# Funkcja celu
frame_objective = ttk.LabelFrame(root, text="Funkcja celu")
frame_objective.pack(fill=tk.X, padx=10, pady=5)
ttk.Label(frame_objective, text="Z = ").pack(side=tk.LEFT)
entry_c1 = ttk.Entry(frame_objective, width=8); entry_c1.pack(side=tk.LEFT)
ttk.Label(frame_objective, text=" * x₁ + ").pack(side=tk.LEFT)
entry_c2 = ttk.Entry(frame_objective, width=8); entry_c2.pack(side=tk.LEFT)
ttk.Label(frame_objective, text=" * x₂ → maksymalizuj").pack(side=tk.LEFT)

# Constraints frame - dynamiczne dodawanie/usuwanie
frame_constraints_control = ttk.Frame(root)
frame_constraints_control.pack(fill=tk.X, padx=10, pady=2)

ttk.Label(frame_constraints_control, text="Ograniczenia (a₁*x₁ + a₂*x₂ [≤/≥/=] b)").pack(side=tk.LEFT)
btn_add = ttk.Button(frame_constraints_control, text="Dodaj ograniczenie")
btn_add.pack(side=tk.LEFT, padx=5)
btn_remove = ttk.Button(frame_constraints_control, text="Usuń ostatnie")
btn_remove.pack(side=tk.LEFT, padx=5)

frame_constraints = ttk.LabelFrame(root, text="Ograniczenia")
frame_constraints.pack(fill=tk.X, padx=10, pady=5)

entry_rows = []  # lista słowników: {'frame', 'a1', 'a2', 'sign', 'b'}

def add_constraint_row(a1_val="", a2_val="", b_val="", sign_val="≤"):
    row = ttk.Frame(frame_constraints)
    row.pack(fill=tk.X, pady=2)
    idx = len(entry_rows) + 1
    ttk.Label(row, text=f"{idx}. ").pack(side=tk.LEFT)
    e1 = ttk.Entry(row, width=8); e1.pack(side=tk.LEFT); e1.insert(0, str(a1_val))
    ttk.Label(row, text=" * x₁ + ").pack(side=tk.LEFT)
    e2 = ttk.Entry(row, width=8); e2.pack(side=tk.LEFT); e2.insert(0, str(a2_val))
    ttk.Label(row, text=" * x₂ ").pack(side=tk.LEFT)

    sign = ttk.Combobox(row, width=3, values=["≤", "≥", "="])
    sign.set(sign_val)
    sign.pack(side=tk.LEFT)

    ttk.Label(row, text=" ").pack(side=tk.LEFT)
    e3 = ttk.Entry(row, width=8); e3.pack(side=tk.LEFT); e3.insert(0, str(b_val))

    entry_rows.append({'frame': row, 'a1': e1, 'a2': e2, 'sign': sign, 'b': e3})

def remove_last_constraint_row():
    if entry_rows:
        last = entry_rows.pop()
        last['frame'].destroy()

# start z 3 ograniczeniami domyślnie
add_constraint_row("1", "1", "10", "≤")
add_constraint_row("1", "0", "8", "≤")
add_constraint_row("0", "1", "6", "≤")

btn_add.config(command=lambda: add_constraint_row())
btn_remove.config(command=lambda: remove_last_constraint_row())

# miejsce na wykres
frame_plot = ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

result_label = ttk.Label(root, text="", font=("Arial", 12, "bold"), foreground="blue")
result_label.pack(pady=5)

# Funkcja obliczająca i rysująca
def compute_feasible_vertices(A, b):
    # A, b to lista par (A_row, b_val) gdzie A_row = [a1,a2]
    ineqs = [(np.array(row), val) for row, val in zip(A, b)]
    # tworzymy listę równań linii: a1*x + a2*y = b
    lines = []
    for row, val in zip(A, b):
        a1, a2 = row
        lines.append((a1, a2, val))
    # dodajemy osie x=0 (y-axis?) i y=0 jako ograniczenia (x>=0,y>=0)
    # ich równania: x = 0 -> 1*x + 0*y = 0 ; y = 0 -> 0*x + 1*y = 0
    axes = [(1,0,0), (0,1,0)]
    candidate_pts = []

    # intersections between every pair of lines (including axes)
    all_lines = lines + axes
    for i in range(len(all_lines)):
        for j in range(i+1, len(all_lines)):
            p = intersection_of_lines(all_lines[i], all_lines[j])
            if p is None:
                continue
            if point_satisfies_all(p, ineqs):
                candidate_pts.append((round(p[0],10), round(p[1],10)))

    # also include intersections with axes where line crosses axis within feasible region
    # (already covered by intersections with axes)
    # remove duplicates
    candidate_pts = list(set(candidate_pts))
    return candidate_pts

def solve():
    global button_show_steps
    try:
        # pobierz współczynniki funkcji celu
        c1, c2 = float(entry_c1.get()), float(entry_c2.get())

        # zbuduj A i b z obsługą ≤ ≥ =
        A_list = []
        b_list = []
        for r in entry_rows:
            a = float(r['a1'].get())
            c = float(r['a2'].get())
            d = float(r['b'].get())
            sign = r['sign'].get()
            if sign == "≤":
                A_list.append([a, c]); b_list.append(d)
            elif sign == "≥":
                A_list.append([-a, -c]); b_list.append(-d)
            elif sign == "=":
                A_list.append([a, c]); b_list.append(d)
                A_list.append([-a, -c]); b_list.append(-d)

        if len(A_list) == 0:
            messagebox.showwarning("Brak ograniczeń", "Brak zdefiniowanych ograniczeń.")
            return

        A_arr = np.array(A_list)
        b_arr = np.array(b_list)

        # spróbuj rozwiązać Simplex (wykrywa nieograniczoność przez wyjątek)
        try:
            val, x, history = simplex_with_steps([c1, c2], A_arr, b_arr)
        except Exception as e:
            # Jeśli błąd to nieograniczony
            messagebox.showerror("Wynik", f"Błąd podczas Simplex: {str(e)}")
            # ale nadal spróbujemy policzyć obszar dopuszczalny do rysunku
            val, x, history = None, None, None

        # usuń poprzedni przycisk 'Pokaż kroki' jeśli istnieje
        if button_show_steps is not None and button_show_steps.winfo_exists():
            button_show_steps.destroy()
            button_show_steps = None

        # jeśli mamy historię, utwórz przycisk podglądu
        if history is not None:
            button_show_steps = ttk.Button(root, text="Pokaż kroki Simplexa", command=lambda: SimplexVisualizer(root, history))
            button_show_steps.pack(pady=5)

        # oblicz wierzchołki obszaru dopuszczalnego
        feasible_pts = compute_feasible_vertices(A_list, b_list)

        # jeśli brak punktów dopuszczalnych -> obszar może być pusty
        if not feasible_pts:
            # sprawdź punkty osi (0,0) jeśli spełniają - może to być jedyny punkt
            if point_satisfies_all((0.0,0.0), [(np.array(row), val) for row,val in zip(A_list, b_list)]):
                feasible_pts = [(0.0,0.0)]
            else:
                # spróbuj wykryć brak rozwiązań
                messagebox.showinfo("Brak rozwiązań", "Brak punktów w obszarze dopuszczalnym (zbiór może być pusty).")
                feasible_pts = []

        # komputer hull dla rysowania obszaru
        hull = monotone_chain_convex_hull(feasible_pts) if feasible_pts else []

        # przygotuj wykres
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6,6))

        # zakresy osi na podstawie punktów z hull lub domyślnie
        if hull:
            xs = [p[0] for p in hull]
            ys = [p[1] for p in hull]
            x_min, x_max = max(0, min(xs)-1), max(max(xs)+1, 5)
            y_min, y_max = max(0, min(ys)-1), max(max(ys)+1, 5)
        else:
            x_min, x_max, y_min, y_max = 0, 10, 0, 10

        # rysuj linie ograniczeń
        x_line = np.linspace(x_min, x_max, 400)
        for i, row in enumerate(A_list):
            a1, a2 = row
            bi = b_list[i]
            # unikaj dzielenia przez zero
            if abs(a2) > 1e-12:
                y_line = (bi - a1 * x_line) / a2
                ax.plot(x_line, y_line, linewidth=1)
            else:
                # pionowa linia x = b/a1
                if abs(a1) > 1e-12:
                    xv = bi / a1
                    ax.axvline(x=xv, linewidth=1)

        # wypełnienie obszaru dopuszczalnego (jeśli hull ma >=3 punkty)
        if len(hull) >= 3:
            poly = plt.Polygon(hull, alpha=0.2, label="Obszar dopuszczalny")
            ax.add_patch(poly)
        elif len(hull) == 2:
            ax.plot([hull[0][0], hull[1][0]], [hull[0][1], hull[1][1]], 'k-', linewidth=2)
            ax.fill([hull[0][0], hull[1][0], hull[1][0], hull[0][0]],
                    [hull[0][1], hull[1][1], 0, 0], alpha=0.1)

        # rysuj wszystkie wierzchołki dopuszczalne
        if feasible_pts:
            pts_np = np.array(feasible_pts)
            ax.scatter(pts_np[:,0], pts_np[:,1], s=30)

        multiple_optima = False
        opt_segment_pts = []

        if history is not None:
            # analiza ostatniego tableau na wielokrotne rozwiązania
            tableau, basis = history[-1]
            obj_row = tableau[-1, :-1]
            # non-basic vars are those not in basis (indices)
            non_basic_vars = [i for i in range(len(obj_row)) if i not in basis]
            alt_dir = None
            for j in non_basic_vars:
                if abs(obj_row[j]) < 1e-9:
                    multiple_optima = True
                    alt_dir = j
                    break

            if val is not None and x is not None:
                x1, x2 = x
                # znajdź wierzchołki obszaru, których wartość funkcji celu ~= val
                opt_vertices = []
                for p in feasible_pts:
                    z = c1 * p[0] + c2 * p[1]
                    if abs(z - val) < 1e-6:
                        opt_vertices.append(p)
                # jeśli występuje wiele takich punktów, narysuj ich obwiednię
                if len(opt_vertices) >= 2:
                    # hull tych punktów
                    seg_hull = monotone_chain_convex_hull(opt_vertices)
                    opt_segment_pts = seg_hull
                # aktualizuj label
                if multiple_optima:
                    result_label.config(text=f"Optimum: x₁={x1:.4f}, x₂={x2:.4f}, Z={val:.4f}  —  NIESKOŃCZENIE WIELE ROZWIĄZAŃ")
                else:
                    result_label.config(text=f"Optimum: x₁={x1:.4f}, x₂={x2:.4f}, Z={val:.4f}")
                # rysuj punkt optymalny
                ax.plot(x1, x2, 'ro', label='Optimum')
                ax.text(x1 + (x_max-x_min)*0.02, x2 + (y_max-y_min)*0.02, f"Z={val:.2f}", color='red')
        else:
            result_label.config(text="Brak rozwiązania Simplex (np. nieograniczony).")

        # rysuj odcinek/obszar rozwiązań optymalnych (jeśli znalezione)
        if opt_segment_pts:
            seg = np.array(opt_segment_pts)
            ax.plot(seg[:,0], seg[:,1], 'g--', linewidth=3, label="Optymalne rozwiązania")
            ax.scatter(seg[:,0], seg[:,1], color='green', s=30)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend()

        # wstaw canvas do GUI
        for w in frame_plot.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    except Exception as e:
        messagebox.showerror("Błąd", str(e))

# Przycisk Oblicz
ttk.Button(root, text="Oblicz", command=solve).pack(pady=8)

# -------------------------
# Klasa wizualizująca kroki Simplexa
# -------------------------
class SimplexVisualizer(tk.Toplevel):
    def __init__(self, root, history):
        super().__init__(root)
        self.title("Kroki metody Simplex")
        self.geometry("900x600")
        self.history = history
        self.index = 0

        self.label = ttk.Label(self, text=f"Iteracja 1 z {len(history)}", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)

        self.tree = ttk.Treeview(self, show='headings')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

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

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = headers
        for h in headers:
            self.tree.heading(h, text=h)
            self.tree.column(h, anchor="center", width=80)

        for r in range(rows-1):
            self.tree.insert("", "end", values=[f"{v:.6f}" for v in tableau[r]])
        self.tree.insert("", "end", values=["—"]*(cols-1) + ["—"])
        self.tree.insert("", "end", values=[f"{v:.6f}" for v in tableau[-1]])

        self.label.config(text=f"Iteracja {i+1} z {len(self.history)}  |  Baza: {', '.join(['x'+str(b+1) for b in basis])}")

    def next(self):
        if self.index < len(self.history)-1:
            self.show_iteration(self.index + 1)

    def prev(self):
        if self.index > 0:
            self.show_iteration(self.index - 1)

# Uruchom aplikację
root.mainloop()
