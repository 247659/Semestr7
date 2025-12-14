import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import copy


# =============================================================================
# LOGIKA (BACKEND) - KLASA SOLVERA
# =============================================================================

class TransportationSolver:
    def __init__(self, costs, supply, demand, goal="MIN"):
        self.goal = goal  # "MIN" lub "MAX"

        # Konwersja na float
        self.orig_costs_input = np.array(costs, dtype=float)
        self.orig_supply = np.array(supply, dtype=float)
        self.orig_demand = np.array(demand, dtype=float)

        # JEŚLI MAKSYMALIZACJA -> Zamieniamy koszty na przeciwne (mnożymy przez -1)
        if self.goal == "MAX":
            self.costs = self.orig_costs_input * -1
        else:
            self.costs = self.orig_costs_input.copy()

        self.supply = self.orig_supply.copy()
        self.demand = self.orig_demand.copy()

        self.rows, self.cols = self.costs.shape
        self.allocation = np.zeros((self.rows, self.cols))

        # Historia kroków do wizualizacji
        self.history = []

        self.balance_problem()

        # Zapisz stan początkowy
        goal_desc = "Maksymalizacja (Zyski zamienione na ujemne)" if self.goal == "MAX" else "Minimalizacja"
        self.log_step(f"Stan początkowy ({goal_desc}, po zbilansowaniu)")

    def log_step(self, description):
        """Zapisuje aktualny stan macierzy alokacji do historii."""
        self.history.append((description, self.allocation.copy()))

    def balance_problem(self):
        s_sum = np.sum(self.supply)
        d_sum = np.sum(self.demand)

        if abs(s_sum - d_sum) < 1e-9:
            return

        if s_sum > d_sum:
            diff = s_sum - d_sum
            col_zeros = np.zeros((self.rows, 1))
            self.costs = np.hstack((self.costs, col_zeros))
            self.demand = np.append(self.demand, diff)
            self.cols += 1
            self.allocation = np.zeros((self.rows, self.cols))

        elif d_sum > s_sum:
            diff = d_sum - s_sum
            row_zeros = np.zeros((1, self.cols))
            self.costs = np.vstack((self.costs, row_zeros))
            self.supply = np.append(self.supply, diff)
            self.rows += 1
            self.allocation = np.zeros((self.rows, self.cols))

    def calculate_total_result(self):
        """Oblicza wynik. Jeśli MAX, odwraca wynik z powrotem na dodatni."""
        raw_result = np.sum(self.allocation * self.costs)
        if self.goal == "MAX":
            return -raw_result  # Zwracamy zysk jako liczbę dodatnią
        return raw_result

    # --- METODY ROZWIĄZYWANIA ---

    def solve_nwcm(self):
        self.history = []
        self.log_step("Start metody Kąta Północno-Zachodniego")

        supply = self.supply.copy()
        demand = self.demand.copy()
        i, j = 0, 0
        while i < self.rows and j < self.cols:
            quantity = min(supply[i], demand[j])
            self.allocation[i, j] = quantity
            supply[i] -= quantity
            demand[j] -= quantity

            self.log_step(f"NWCM: Przydzielono {quantity} na trasie Dost{i + 1}->Odb{j + 1}")

            if supply[i] == 0:
                i += 1
            else:
                j += 1
        return self.allocation

    def solve_lcm(self):
        self.history = []
        self.log_step("Start metody Minimalnego Elementu Macierzy")

        supply = self.supply.copy()
        demand = self.demand.copy()
        costs_temp = self.costs.copy()

        while np.sum(supply) > 0 and np.sum(demand) > 0:
            min_val = np.min(costs_temp)
            if min_val == np.inf: break

            rows_idx, cols_idx = np.where(costs_temp == min_val)
            r, c = rows_idx[0], cols_idx[0]

            quantity = min(supply[r], demand[c])
            self.allocation[r, c] = quantity
            supply[r] -= quantity
            demand[c] -= quantity

            val_display = min_val if self.goal == "MIN" else -min_val
            type_display = "koszt" if self.goal == "MIN" else "zysk"

            self.log_step(
                f"LCM: Wybrano {type_display} {val_display}. Przydzielono {quantity} (Dost{r + 1}->Odb{c + 1})")

            if supply[r] == 0: costs_temp[r, :] = np.inf
            if demand[c] == 0: costs_temp[:, c] = np.inf

        return self.allocation

    def solve_row_min(self):
        self.history = []
        self.log_step("Start metody Minimum Wierszowego")
        supply = self.supply.copy()
        demand = self.demand.copy()

        for r in range(self.rows):
            while supply[r] > 0:
                valid_cols = [c for c in range(self.cols) if demand[c] > 0]
                if not valid_cols: break

                best_c = valid_cols[0]
                min_c_val = self.costs[r, best_c]
                for c in valid_cols:
                    if self.costs[r, c] < min_c_val:
                        min_c_val = self.costs[r, c]
                        best_c = c

                quantity = min(supply[r], demand[best_c])
                self.allocation[r, best_c] = quantity
                supply[r] -= quantity
                demand[best_c] -= quantity

                self.log_step(f"RowMin: Wiersz {r + 1}. Przydzielono {quantity} do Odb{best_c + 1}")
        return self.allocation

    def solve_vam(self):
        self.history = []
        self.log_step("Start Aproksymacji Vogela (VAM)")

        supply = self.supply.copy()
        demand = self.demand.copy()
        costs = self.costs.copy()

        row_active = [True] * self.rows
        col_active = [True] * self.cols

        while np.sum(supply) > 0 and np.sum(demand) > 0:
            row_penalties = []
            for r in range(self.rows):
                if not row_active[r]:
                    row_penalties.append(-1);
                    continue
                valid = sorted([costs[r, c] for c in range(self.cols) if col_active[c]])
                if len(valid) >= 2:
                    row_penalties.append(valid[1] - valid[0])
                elif len(valid) == 1:
                    row_penalties.append(valid[0])
                else:
                    row_penalties.append(0)

            col_penalties = []
            for c in range(self.cols):
                if not col_active[c]:
                    col_penalties.append(-1);
                    continue
                valid = sorted([costs[r, c] for r in range(self.rows) if row_active[r]])
                if len(valid) >= 2:
                    col_penalties.append(valid[1] - valid[0])
                elif len(valid) == 1:
                    col_penalties.append(valid[0])
                else:
                    col_penalties.append(0)

            max_r_p = max(row_penalties)
            max_c_p = max(col_penalties)

            if max_r_p == -1 and max_c_p == -1: break

            if max_r_p >= max_c_p:
                r = row_penalties.index(max_r_p)
                c_cands = [c for c in range(self.cols) if col_active[c]]
                c = min(c_cands, key=lambda x: costs[r, x])
                desc = f"VAM: Max kara wiersza. Wybrano Dost{r + 1}."
            else:
                c = col_penalties.index(max_c_p)
                r_cands = [r for r in range(self.rows) if row_active[r]]
                r = min(r_cands, key=lambda x: costs[x, c])
                desc = f"VAM: Max kara kolumny. Wybrano Odb{c + 1}."

            qty = min(supply[r], demand[c])
            self.allocation[r, c] = qty
            supply[r] -= qty
            demand[c] -= qty

            self.log_step(f"{desc} Przydzielono {qty} na trasie Dost{r + 1}->Odb{c + 1}")

            if supply[r] <= 1e-9: row_active[r] = False
            if demand[c] <= 1e-9: col_active[c] = False

        return self.allocation

    def solve_optimal_modi(self):
        self.log_step("--- Rozpoczęcie optymalizacji MODI ---")

        for iter_num in range(100):
            u = [None] * self.rows
            v = [None] * self.cols
            u[0] = 0

            changed = True
            while changed:
                changed = False
                for r in range(self.rows):
                    for c in range(self.cols):
                        if self.allocation[r, c] > 0:
                            if u[r] is not None and v[c] is None:
                                v[c] = self.costs[r, c] - u[r]
                                changed = True
                            elif u[r] is None and v[c] is not None:
                                u[r] = self.costs[r, c] - v[c]
                                changed = True

            u = [x if x is not None else 0 for x in u]
            v = [x if x is not None else 0 for x in v]

            negatives = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.allocation[r, c] == 0:
                        delta = self.costs[r, c] - (u[r] + v[c])
                        if delta < -1e-9:
                            negatives.append((delta, r, c))

            if not negatives:
                self.log_step("MODI: Brak możliwości poprawy. Rozwiązanie optymalne.")
                break

            negatives.sort(key=lambda x: x[0])
            best_delta, start_r, start_c = negatives[0]

            path = self.find_path(start_r, start_c)
            if not path: break

            minus_cells = []
            for i in range(1, len(path), 2):
                r_p, c_p = path[i]
                minus_cells.append(self.allocation[r_p, c_p])

            theta = min(minus_cells)

            for i, (r_p, c_p) in enumerate(path):
                if i % 2 == 0:
                    self.allocation[r_p, c_p] += theta
                else:
                    self.allocation[r_p, c_p] -= theta

            self.log_step(f"MODI: Przesunięto {theta} jednostek wzdłuż cyklu.")

        return self.allocation

    def find_path(self, start_r, start_c):
        stack = [[(start_r, start_c)]]
        while stack:
            path = stack.pop()
            curr_r, curr_c = path[-1]
            if len(path) >= 4:
                if path[-2][1] == curr_c and curr_r == start_r: return path
                if path[-2][0] == curr_r and curr_c == start_c: return path

            last_move_was_vertical = False
            if len(path) > 1:
                if path[-1][0] != path[-2][0]: last_move_was_vertical = True

            next_steps = []
            if len(path) == 1 or last_move_was_vertical:
                for c in range(self.cols):
                    if c != curr_c and self.allocation[curr_r, c] > 0: next_steps.append((curr_r, c))
            else:
                for r in range(self.rows):
                    if r != curr_r and self.allocation[r, curr_c] > 0: next_steps.append((r, curr_c))

            for step in next_steps:
                if step not in path: stack.append(path + [step])
        return None


# =============================================================================
# GUI - WIZUALIZACJA
# =============================================================================

class StepViewer(tk.Toplevel):
    def __init__(self, parent, history, rows, cols):
        super().__init__(parent)
        self.title("Kroki Rozwiązania")
        self.geometry("600x500")
        self.history = history
        self.rows = rows
        self.cols = cols
        self.current_step = 0

        ctrl = ttk.Frame(self)
        ctrl.pack(pady=10)
        ttk.Button(ctrl, text="<<", command=self.go_start).pack(side="left")
        ttk.Button(ctrl, text="<", command=self.go_prev).pack(side="left")
        self.lbl = ttk.Label(ctrl, text="0/0")
        self.lbl.pack(side="left", padx=10)
        ttk.Button(ctrl, text=">", command=self.go_next).pack(side="left")
        ttk.Button(ctrl, text=">>", command=self.go_end).pack(side="left")

        self.desc = ttk.Label(self, text="", foreground="blue")
        self.desc.pack()
        self.txt = tk.Text(self)
        self.txt.pack(fill="both", expand=True)
        self.show()

    def show(self):
        d, alloc = self.history[self.current_step]
        self.lbl.config(text=f"{self.current_step + 1}/{len(self.history)}")
        self.desc.config(text=d)

        t = ""
        headers = [""] + [f"Odb{j + 1}" for j in range(self.cols)]
        fmt = "{:<8}" + "".join([f"{{:^8}}" for _ in range(self.cols)])
        t += fmt.format(*headers) + "\n" + "-" * 50 + "\n"
        for i in range(self.rows):
            vals = [f"{v:g}" if v > 0 else "." for v in alloc[i]]
            t += fmt.format(f"Dst{i + 1}", *vals) + "\n"
        self.txt.delete(1.0, tk.END)
        self.txt.insert(tk.END, t)

    def go_prev(self):
        if self.current_step > 0: self.current_step -= 1; self.show()

    def go_next(self):
        if self.current_step < len(self.history) - 1: self.current_step += 1; self.show()

    def go_start(self):
        self.current_step = 0; self.show()

    def go_end(self):
        self.current_step = len(self.history) - 1; self.show()


# =============================================================================
# GUI - GŁÓWNA APLIKACJA
# =============================================================================

class TransportationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zagadnienie Transportowe - MAX/MIN")
        self.root.geometry("1100x600")  # Szersze okno dla lepszego układu
        self.solver = None

        # Pasek pomocy (Góra)
        ttk.Button(root, text="Pomoc / Instrukcja", command=self.show_help).pack(anchor="e", padx=10, pady=5)

        # 1. Konfiguracja (Góra)
        fr1 = ttk.LabelFrame(root, text="1. Konfiguracja")
        fr1.pack(fill="x", padx=10)
        ttk.Label(fr1, text="Dostawcy:").pack(side="left")
        self.s_r = ttk.Spinbox(fr1, from_=2, to=10, width=5);
        self.s_r.set(3);
        self.s_r.pack(side="left")
        ttk.Label(fr1, text="Odbiorcy:").pack(side="left")
        self.s_c = ttk.Spinbox(fr1, from_=2, to=10, width=5);
        self.s_c.set(3);
        self.s_c.pack(side="left")
        ttk.Button(fr1, text="Generuj Tabelę", command=self.create_grid).pack(side="left", padx=10)

        # 2. Metoda i CEL (Góra)
        fr2 = ttk.LabelFrame(root, text="2. Metoda i Cel Optymalizacji")
        fr2.pack(fill="x", padx=10, pady=5)

        goal_frame = ttk.Frame(fr2)
        goal_frame.pack(side="left", padx=10)
        ttk.Label(goal_frame, text="Cel:").pack(anchor="w")
        self.goal_var = tk.StringVar(value="MIN")
        ttk.Radiobutton(goal_frame, text="Minimalizacja (Koszty)", value="MIN", variable=self.goal_var).pack(anchor="w")
        ttk.Radiobutton(goal_frame, text="Maksymalizacja (Zyski)", value="MAX", variable=self.goal_var).pack(anchor="w")

        ttk.Separator(fr2, orient="vertical").pack(side="left", fill="y", padx=10)

        alg_frame = ttk.Frame(fr2)
        alg_frame.pack(side="left")
        self.m_var = tk.StringVar(value="VAM")
        opts = [("Kąt Pn-Zach", "NWCM"), ("Min. Element", "LCM"), ("Min. Wierszowe", "ROW_MIN"), ("Vogel (VAM)", "VAM")]
        for t, v in opts: ttk.Radiobutton(alg_frame, text=t, value=v, variable=self.m_var).pack(side="left", padx=5)

        self.do_opt = tk.BooleanVar(value=True)
        ttk.Checkbutton(fr2, text="Doptymalizuj (MODI)", variable=self.do_opt).pack(side="left", padx=10)

        ttk.Button(fr2, text="ROZWIĄŻ", command=self.solve).pack(side="right", padx=10)
        self.btn_steps = ttk.Button(fr2, text="Kroki", state="disabled", command=self.steps);
        self.btn_steps.pack(side="right")

        # --- GŁÓWNA ZAWARTOŚĆ (Podział na LEWO i PRAWO) ---
        main_content = ttk.Frame(root)
        main_content.pack(fill="both", expand=True, padx=10, pady=5)

        # LEWA STRONA: Tabela Danych
        left_frame = ttk.LabelFrame(main_content, text="3. Tabela Danych (Edycja)")
        left_frame.pack(side="left", fill="both", expand=True)

        self.canv = tk.Canvas(left_frame)
        self.canv.pack(side="left", fill="both", expand=True)

        sb_y = ttk.Scrollbar(left_frame, orient="vertical", command=self.canv.yview)
        sb_y.pack(side="right", fill="y")

        sb_x = ttk.Scrollbar(left_frame, orient="horizontal", command=self.canv.xview)
        sb_x.pack(side="bottom", fill="x")

        self.canv.config(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        self.frame_grid = ttk.Frame(self.canv)
        self.canv.create_window((0, 0), window=self.frame_grid, anchor="nw")
        self.frame_grid.bind("<Configure>", lambda e: self.canv.config(scrollregion=self.canv.bbox("all")))

        # PRAWA STRONA: Wyniki
        right_frame = ttk.LabelFrame(main_content, text="4. Wyniki")
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self.res_txt = tk.Text(right_frame, width=40, font=("Consolas", 10))
        self.res_txt.pack(fill="both", expand=True, padx=5, pady=5)

        self.create_grid()

    def show_help(self):
        msg = (
            "1. Wybierz 'Maksymalizacja' dla zadań z zyskami (wpisuj liczby dodatnie).\n"
            "2. Wybierz 'Minimalizacja' dla zadań z kosztami.\n"
            "3. Wpisz dane w tabelę po lewej.\n"
            "4. Kliknij ROZWIĄŻ. Wynik pojawi się w oknie po PRAWEJ stronie."
        )
        messagebox.showinfo("Pomoc", msg)

    def create_grid(self):
        for w in self.frame_grid.winfo_children(): w.destroy()
        self.cost_ents = [];
        self.supp_ents = [];
        self.dem_ents = []
        self.btn_steps.config(state="disabled")
        try:
            R, C = int(self.s_r.get()), int(self.s_c.get())
        except:
            return

        # Nagłówki
        tk.Label(self.frame_grid, text="Dost/Odb").grid(row=0, column=0)
        for j in range(C): tk.Label(self.frame_grid, text=f"Odb {j + 1}").grid(row=0, column=j + 1)
        tk.Label(self.frame_grid, text="PODAŻ", fg="blue").grid(row=0, column=C + 1)

        for i in range(R):
            tk.Label(self.frame_grid, text=f"Dost {i + 1}").grid(row=i + 1, column=0)
            row = []
            for j in range(C):
                e = ttk.Entry(self.frame_grid, width=6, justify="center")
                e.insert(0, "0")
                e.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                row.append(e)
            self.cost_ents.append(row)
            s = tk.Entry(self.frame_grid, width=8, fg="blue");
            s.insert(0, "0");
            s.grid(row=i + 1, column=C + 1)
            self.supp_ents.append(s)

        tk.Label(self.frame_grid, text="POPYT", fg="green").grid(row=R + 1, column=0)
        for j in range(C):
            d = tk.Entry(self.frame_grid, width=6, fg="green");
            d.insert(0, "0");
            d.grid(row=R + 1, column=j + 1)
            self.dem_ents.append(d)

    def solve(self):
        try:
            C = [[float(e.get()) for e in r] for r in self.cost_ents]
            S = [float(e.get()) for e in self.supp_ents]
            D = [float(e.get()) for e in self.dem_ents]
            goal = self.goal_var.get()

            self.solver = TransportationSolver(C, S, D, goal)

            m = self.m_var.get()
            if m == "NWCM":
                self.solver.solve_nwcm()
            elif m == "LCM":
                self.solver.solve_lcm()
            elif m == "ROW_MIN":
                self.solver.solve_row_min()
            elif m == "VAM":
                self.solver.solve_vam()

            res_start = self.solver.calculate_total_result()
            res_opt = res_start
            opt_done = False

            if self.do_opt.get():
                self.solver.solve_optimal_modi()
                res_opt = self.solver.calculate_total_result()
                opt_done = True

            self.show_res(res_start, res_opt, opt_done, goal)
            self.btn_steps.config(state="normal")

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def show_res(self, s, o, done, goal):
        lbl = "KOSZT" if goal == "MIN" else "ZYSK"
        t = f"Cel: {goal} ({'Minimalizacja Kosztów' if goal == 'MIN' else 'Maksymalizacja Zysku'})\n"
        t += f"Początkowy {lbl}: {s:.2f}\n"
        if done: t += f"Optymalny {lbl} (MODI): {o:.2f}\n"

        t += "\nPLAN PRZEWOZÓW:\n"
        cols = self.solver.cols
        head = [""] + [f"Odb{j + 1}" for j in range(cols)]
        fmt = "{:<8}" + "".join(["{:^8}" for _ in range(cols)])
        t += fmt.format(*head) + "\n"
        for i in range(self.solver.rows):
            vals = [f"{v:g}" if v > 0 else "." for v in self.solver.allocation[i]]
            t += fmt.format(f"Dst{i + 1}", *vals) + "\n"

        self.res_txt.delete(1.0, tk.END)
        self.res_txt.insert(tk.END, t)

    def steps(self):
        if self.solver: StepViewer(self.root, self.solver.history, self.solver.rows, self.solver.cols)


if __name__ == "__main__":
    root = tk.Tk()
    app = TransportationApp(root)
    root.mainloop()