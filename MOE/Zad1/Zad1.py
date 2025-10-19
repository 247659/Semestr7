import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import linprog

# --- Funkcje pomocnicze geometryczne ---
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

def convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def compute_feasible_polygon(ineqs):
    all_ineqs = list(ineqs)
    all_ineqs.append(((-1,0),0))  # x>=0
    all_ineqs.append(((0,-1),0))  # y>=0
    lines=[]
    for A,b in all_ineqs:
        lines.append((A[0],A[1],b))
    pts=[]
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            P=intersection_of_lines(lines[i],lines[j])
            if P is None: continue
            x,y=P
            if point_satisfies_all(x,y,all_ineqs):
                pts.append((x,y))
    if not pts: return []
    return convex_hull(pts)

# --- GUI ---
root = tk.Tk()
root.title("Optymalizacja liniowa")
root.geometry("1020x900")

# --- Górny pasek z przyciskiem Pomoc ---
frame_top = ttk.Frame(root)
frame_top.pack(fill=tk.X, padx=10, pady=5)

ttk.Button(frame_top, text="Pomoc", command=lambda: show_help()).pack(side=tk.LEFT)

# --- Funkcja help ---
def show_help():
    help_text = (
        "Program: Optymalizacja liniowa\n\n"
        "Opis:\n"
        "  Program pozwala rozwiązywać zadania optymalizacji liniowej "
        "w dwóch zmiennych (x₁, x₂) z ograniczeniami nierównościowymi (≤, ≥) oraz równaniami (=).\n\n"
        "Instrukcja obsługi:\n"
        "  1. Wprowadź współczynniki funkcji celu c1 i c2.\n"
        "  2. Wybierz liczbę ograniczeń.\n"
        "  3. Wprowadź współczynniki a1, a2, b dla każdego ograniczenia i wybierz typ (≤, ≥, =).\n"
        "  4. Wybierz typ zmiennych (ciągłe lub całkowite).\n"
        "  5. Wybierz tryb:\n"
        "       - 'Rozpocznij krok po kroku' – przechodzenie etapami.\n"
        "       - 'Ostateczny wynik' – wyświetlenie gotowego rozwiązania od razu.\n"
        "  6. W kroku po kroku używaj przycisku 'Dalej', aby przechodzić między etapami.\n\n"
        "Zakres działania:\n"
        "  - Maksymalizacja funkcji celu Z = c1*x1 + c2*x2\n"
        "  - Obsługa ograniczeń typu ≤, ≥ i =\n"
        "  - Wyznaczanie obszaru dopuszczalnego w pierwszej ćwiartce\n"
        "  - Interaktywny wykres z matplotlib w Tkinter\n"
    )
    messagebox.showinfo("Pomoc", help_text)



# --- Funkcja celu ---
frame_objective = ttk.LabelFrame(root, text="Funkcja celu")
frame_objective.pack(fill=tk.X, padx=10, pady=5)
ttk.Label(frame_objective,text="Z = ").pack(side=tk.LEFT)
entry_c1 = ttk.Entry(frame_objective,width=8); entry_c1.pack(side=tk.LEFT)
ttk.Label(frame_objective,text=" * x₁ + ").pack(side=tk.LEFT)
entry_c2 = ttk.Entry(frame_objective,width=8); entry_c2.pack(side=tk.LEFT)
ttk.Label(frame_objective,text=" * x₂ → maksymalizuj").pack(side=tk.LEFT)

# --- Kontrola liczby ograniczeń ---
frame_num_constraints = ttk.Frame(root)
frame_num_constraints.pack(fill=tk.X, padx=10, pady=5)
ttk.Label(frame_num_constraints,text="Liczba ograniczeń:").pack(side=tk.LEFT)
num_constraints_var = tk.IntVar(value=4)
spin_constraints = ttk.Spinbox(frame_num_constraints, from_=1, to=12, textvariable=num_constraints_var, width=4)
spin_constraints.pack(side=tk.LEFT, padx=5)

# --- Ograniczenia dynamiczne ---
frame_constraints = ttk.LabelFrame(root,text="Ograniczenia (a₁*x₁ + a₂*x₂ ≤, ≥, = b)")
frame_constraints.pack(fill=tk.X,padx=10,pady=5)
entry_a1, entry_a2, entry_b, sign_var, sign_box_widgets = [],[],[],[],[]

def build_constraint_rows(n):
    for w in frame_constraints.winfo_children(): w.destroy()
    entry_a1.clear(); entry_a2.clear(); entry_b.clear(); sign_var.clear(); sign_box_widgets.clear()
    for i in range(n):
        row=ttk.Frame(frame_constraints); row.pack(fill=tk.X,pady=2,padx=4)
        ttk.Label(row,text=f"{i+1:2d}. ").pack(side=tk.LEFT)
        e1=ttk.Entry(row,width=8); e1.pack(side=tk.LEFT)
        ttk.Label(row,text=" * x₁ + ").pack(side=tk.LEFT)
        e2=ttk.Entry(row,width=8); e2.pack(side=tk.LEFT)
        ttk.Label(row,text=" * x₂ ").pack(side=tk.LEFT)
        var=tk.StringVar(value="≤")
        sign_box=ttk.Combobox(row,textvariable=var,values=["≤","=","≥"],width=3,state="readonly")
        sign_box.pack(side=tk.LEFT,padx=6)
        e3=ttk.Entry(row,width=8); e3.pack(side=tk.LEFT)
        entry_a1.append(e1); entry_a2.append(e2); entry_b.append(e3); sign_var.append(var); sign_box_widgets.append(sign_box)

build_constraint_rows(num_constraints_var.get())
num_constraints_var.trace_add("write", lambda *a: build_constraint_rows(num_constraints_var.get()))

# --- Typ zmiennych ---
frame_vars=ttk.LabelFrame(root,text="Typ zmiennych")
frame_vars.pack(fill=tk.X,padx=10,pady=5)
var_type=tk.StringVar(value="continuous")
ttk.Radiobutton(frame_vars,text="Ciągłe",variable=var_type,value="continuous").pack(side=tk.LEFT,padx=10,pady=5)
ttk.Radiobutton(frame_vars,text="Naturalne (całkowite ≥0)",variable=var_type,value="integer").pack(side=tk.LEFT,padx=10,pady=5)

# --- Tryb rozwiązania ---
frame_mode=ttk.LabelFrame(root,text="Tryb rozwiązania")
frame_mode.pack(fill=tk.X,padx=10,pady=5)

step_btn = ttk.Button(frame_mode,text="Dalej")
step_btn.pack(side=tk.LEFT,padx=5)
ttk.Button(frame_mode,text="Rozpocznij krok po kroku",command=lambda: solver.start(final=False)).pack(side=tk.LEFT,padx=5)
ttk.Button(frame_mode,text="Ostateczny wynik",command=lambda: solver.start(final=True)).pack(side=tk.LEFT,padx=5)

# --- Wynik nad wykresem ---
result_plot_label = ttk.Label(root, text="", font=("Arial", 12), foreground="blue")
result_plot_label.pack(pady=5)

# --- Wykres ---
frame_plot=ttk.Frame(root)
frame_plot.pack(fill=tk.BOTH,expand=True)

# --- Krok po kroku ---
class StepSolver:
    def __init__(self):
        self.step = 0
        self.steps_total = 4
        self.A_ub,self.b_ub,self.A_eq,self.b_eq=[],[],[],[]
        self.c1=self.c2=self.x1_opt=self.x2_opt=self.z_opt=0
        self.solver_info=""
        self.feasible_poly=[]
        self.canvas=None
        self.fig=None
        self.ax=None
        self.final=False

    def start(self,final=False):
        self.final=final
        try:
            # Wczytanie danych
            self.c1=float(entry_c1.get()); self.c2=float(entry_c2.get())
            n_constraints=min(len(entry_a1),len(entry_a2),len(entry_b),len(sign_var))
            self.A_ub=[]; self.b_ub=[]; self.A_eq=[]; self.b_eq=[]
            for i in range(n_constraints):
                a1=float(entry_a1[i].get()); a2=float(entry_a2[i].get()); b=float(entry_b[i].get())
                sign=sign_var[i].get()
                if sign=="≤": self.A_ub.append([a1,a2]); self.b_ub.append(b)
                elif sign=="≥": self.A_ub.append([-a1,-a2]); self.b_ub.append(-b)
                elif sign=="=": self.A_eq.append([a1,a2]); self.b_eq.append(b)
            if not(self.A_ub or self.A_eq): messagebox.showerror("Błąd","Brak ograniczeń"); return
            # Rozwiązanie zależnie od typu zmiennych
            if var_type.get()=="continuous":
                res = linprog([-self.c1,-self.c2],
                              A_ub=self.A_ub if self.A_ub else None,
                              b_ub=self.b_ub if self.b_ub else None,
                              A_eq=self.A_eq if self.A_eq else None,
                              b_eq=self.b_eq if self.A_eq else None,
                              bounds=(0,None),
                              method="highs")
                if not res.success: messagebox.showerror("Błąd","Brak dopuszczalnego rozwiązania"); return
                self.x1_opt,self.x2_opt=res.x
                self.z_opt=-res.fun
                self.solver_info="Rozwiązanie ciągłe"
            else:
                # ILP brute-force
                feasible_points=[]; z_best=-float("inf")
                for x1 in range(51):
                    for x2 in range(51):
                        ok=True
                        for i,A in enumerate(self.A_ub):
                            if A[0]*x1+A[1]*x2-self.b_ub[i]>1e-9: ok=False; break
                        if ok and self.A_eq:
                            for i,A in enumerate(self.A_eq):
                                if abs(A[0]*x1+A[1]*x2-self.b_eq[i])>1e-9: ok=False; break
                        if ok:
                            z=self.c1*x1+self.c2*x2
                            feasible_points.append((x1,x2,z))
                            if z>z_best: z_best=z; self.x1_opt=x1; self.x2_opt=x2
                if not feasible_points: messagebox.showerror("Błąd","Brak dopuszczalnych punktów całkowitych"); return
                self.z_opt=z_best
                self.solver_info="Rozwiązanie całkowite"
            # Poligon dopuszczalny
            ineqs=[]
            for i,A in enumerate(self.A_ub): ineqs.append((A,self.b_ub[i]))
            for i, A in enumerate(self.A_eq):
                bi = self.b_eq[i]
                # Dwa ograniczenia dla "="
                ineqs.append((A, bi))  # A*x <= b
                ineqs.append(([-A[0], -A[1]], -bi))  # -A*x <= -b → A*x >= b
            self.feasible_poly=compute_feasible_polygon(ineqs)
            # Wykres
            plt.close('all')
            self.fig,self.ax=plt.subplots(figsize=(6,6))
            for widget in frame_plot.winfo_children(): widget.destroy()
            self.canvas=FigureCanvasTkAgg(self.fig,master=frame_plot)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
            self.step=0 if not final else self.steps_total
            self.update_plot()
        except Exception as e:
            messagebox.showerror("Błąd",str(e))

    def next_step(self):
        if self.step<self.steps_total: self.step+=1; self.update_plot()

    def update_plot(self):
        # Czyścimy poprzednie rysunki
        self.ax.clear()

        x = np.linspace(0, max(10, self.x1_opt * 2), 400)

        # --- Ograniczenia ≤ i ≥ ---
        if self.step >= 1:
            for i, A in enumerate(self.A_ub):
                y = (self.b_ub[i] - A[0] * x) / A[1] if A[1] != 0 else None
                if y is not None:
                    y_plot = np.clip(y, 0, None)
                    self.ax.plot(x, y_plot, label=f"{A[0]}x₁+{A[1]}x₂≤{self.b_ub[i]}")
                else:
                    xpos = max(0, self.b_ub[i] / A[0])
                    self.ax.axvline(xpos, label=f"x₁≤{xpos:.2f}")

            # --- Ograniczenia równości ---
            for i, A in enumerate(self.A_eq):
                bi = self.b_eq[i]
                y = (bi - A[0] * x) / A[1] if A[1] != 0 else None
                if y is not None:
                    y_plot = np.clip(y, 0, None)
                    self.ax.plot(x, y_plot, '--', color='blue', label=f"{A[0]}x₁+{A[1]}x₂={bi}")
                else:
                    xpos = max(0, bi / A[0])
                    self.ax.axvline(xpos, linestyle='--', color='blue', label=f"x₁={xpos:.2f}")

        # --- Obszar dopuszczalny ---
        if self.step >= 2 and self.feasible_poly:
            xs, ys = zip(*self.feasible_poly)
            xs_clip = [max(0, x) for x in xs]
            ys_clip = [max(0, y) for y in ys]
            self.ax.fill(xs_clip, ys_clip, color='lightgreen', alpha=0.4, label="Obszar dopuszczalny")

        # --- Funkcja celu ---
        if self.step >= 3:
            y = (self.z_opt - self.c1 * x) / self.c2
            y_plot = np.clip(y, 0, None)
            self.ax.plot(x, y_plot, 'r-', label=f"Funkcja celu Z={self.z_opt:.2f}")

        # --- Punkt optimum ---
        if self.step >= 4:
            self.ax.plot(max(0, self.x1_opt), max(0, self.x2_opt), 'ro',
                         label=f"Optimum ({self.x1_opt:.2f},{self.x2_opt:.2f})")
            self.ax.text(max(0, self.x1_opt) + 0.2, max(0, self.x2_opt), f"Z={self.z_opt:.2f}", color='red')
            result_plot_label.config(
                text=f"Optimum: x₁ = {self.x1_opt:.2f}, x₂ = {self.x2_opt:.2f}, Z = {self.z_opt:.2f}"
            )

        # --- Wyśrodkowanie na pierwszej ćwiartce ---
        self.ax.set_xlim(0, max(10, self.x1_opt * 1.5))
        self.ax.set_ylim(0, max(10, self.x2_opt * 1.5))

        self.ax.legend()
        self.canvas.draw()


solver = StepSolver()
step_btn.config(command=solver.next_step)

root.mainloop()
