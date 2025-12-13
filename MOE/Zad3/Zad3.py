import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import copy


# =============================================================================
# LOGIKA (BACKEND) - KLASA SOLVERA
# =============================================================================

class TransportationSolver:
    def __init__(self, costs, supply, demand):
        self.orig_costs = np.array(costs, dtype=float)
        self.orig_supply = np.array(supply, dtype=float)
        self.orig_demand = np.array(demand, dtype=float)

        # Kopia robocza
        self.costs = self.orig_costs.copy()
        self.supply = self.orig_supply.copy()
        self.demand = self.orig_demand.copy()
        self.rows, self.cols = self.costs.shape
        self.allocation = np.zeros((self.rows, self.cols))

        # Historia kroków do wizualizacji (lista krotek: (opis_kroku, kopia_alokacji))
        self.history = []

        self.balance_problem()

        # Zapisz stan początkowy (pusta tabela)
        self.log_step("Stan początkowy (po zbilansowaniu)")

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

    def calculate_total_cost(self):
        return np.sum(self.allocation * self.costs)

    # --- METODY ROZWIĄZYWANIA (Z LOGOWANIEM KROKÓW) ---

    def solve_nwcm(self):
        self.history = []  # Reset historii dla nowej metody
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

            self.log_step(f"LCM: Wybrano koszt {min_val}. Przydzielono {quantity} na trasie Dost{r + 1}->Odb{c + 1}")

            if supply[r] == 0:
                costs_temp[r, :] = np.inf
            if demand[c] == 0:
                costs_temp[:, c] = np.inf

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

                self.log_step(
                    f"RowMin: Wiersz {r + 1}, min koszt {min_c_val}. Przydzielono {quantity} do Odb{best_c + 1}")

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
            # Obliczanie kar (kod identyczny jak wcześniej, pomijam dla czytelności komentarze)
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
                desc = f"VAM: Max kara wiersza ({max_r_p}). Wybrano Dost{r + 1}."
            else:
                c = col_penalties.index(max_c_p)
                r_cands = [r for r in range(self.rows) if row_active[r]]
                r = min(r_cands, key=lambda x: costs[x, c])
                desc = f"VAM: Max kara kolumny ({max_c_p}). Wybrano Odb{c + 1}."

            qty = min(supply[r], demand[c])
            self.allocation[r, c] = qty
            supply[r] -= qty
            demand[c] -= qty

            self.log_step(f"{desc} Przydzielono {qty} na trasie Dost{r + 1}->Odb{c + 1}")

            if supply[r] <= 1e-9: row_active[r] = False
            if demand[c] <= 1e-9: col_active[c] = False

        return self.allocation

    def solve_optimal_modi(self):
        # Uwaga: nie resetujemy historii, bo MODI jest kontynuacją metody startowej
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
                self.log_step("MODI: Brak ujemnych kosztów zredukowanych. Rozwiązanie optymalne.")
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

            # Aktualizacja alokacji
            for i, (r_p, c_p) in enumerate(path):
                if i % 2 == 0:
                    self.allocation[r_p, c_p] += theta
                else:
                    self.allocation[r_p, c_p] -= theta

            self.log_step(
                f"MODI Iteracja {iter_num + 1}: Przesunięto {theta} jednostek wzdłuż cyklu (Delta={best_delta:.2f})")

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
                if path[-1][0] != path[-2][0]:
                    last_move_was_vertical = True

            next_steps = []
            if len(path) == 1 or last_move_was_vertical:
                for c in range(self.cols):
                    if c != curr_c and self.allocation[curr_r, c] > 0:
                        next_steps.append((curr_r, c))
            else:
                for r in range(self.rows):
                    if r != curr_r and self.allocation[r, curr_c] > 0:
                        next_steps.append((r, curr_c))

            for step in next_steps:
                if step not in path:
                    stack.append(path + [step])
        return None


# =============================================================================
# GUI - OKNO WIZUALIZACJI KROKÓW (NOWOŚĆ DLA OCENY 4/5)
# =============================================================================

class StepViewer(tk.Toplevel):
    def __init__(self, parent, history, rows, cols):
        super().__init__(parent)
        self.title("Rozwiązanie Krok po Kroku")
        self.geometry("600x500")

        self.history = history
        self.rows = rows
        self.cols = cols
        self.current_step = 0

        # Panel sterowania
        control_frame = ttk.Frame(self)
        control_frame.pack(side="top", fill="x", pady=10)

        ttk.Button(control_frame, text="<< Początek", command=self.go_start).pack(side="left", padx=5)
        ttk.Button(control_frame, text="< Poprzedni", command=self.go_prev).pack(side="left", padx=5)

        self.lbl_step = ttk.Label(control_frame, text="Krok 0 / 0", font=("Arial", 10, "bold"))
        self.lbl_step.pack(side="left", padx=20)

        ttk.Button(control_frame, text="Następny >", command=self.go_next).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Koniec >>", command=self.go_end).pack(side="left", padx=5)

        # Opis kroku
        self.lbl_desc = ttk.Label(self, text="", foreground="blue", wraplength=550)
        self.lbl_desc.pack(pady=5)

        # Widok tabeli
        self.text_area = tk.Text(self, font=("Consolas", 10))
        self.text_area.pack(fill="both", expand=True, padx=10, pady=10)

        self.show_current_step()

    def show_current_step(self):
        desc, allocation = self.history[self.current_step]
        self.lbl_step.config(text=f"Krok {self.current_step + 1} z {len(self.history)}")
        self.lbl_desc.config(text=desc)

        # Rysowanie tabeli tekstowej
        txt = ""
        headers = [""] + [f"Odb{j + 1}" for j in range(self.cols)]
        col_width = 8
        row_fmt = "{:<8}" + "".join([f"{{:^{col_width}}}" for _ in range(self.cols)])

        txt += row_fmt.format(*headers) + "\n"
        txt += "-" * (8 + self.cols * col_width) + "\n"

        for i in range(self.rows):
            row_vals = []
            for j in range(self.cols):
                val = allocation[i, j]
                if val > 0:
                    row_vals.append(f"{val:g}")
                else:
                    row_vals.append(".")
            txt += row_fmt.format(f"Dost{i + 1}", *row_vals) + "\n"

        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, txt)

    def go_prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.show_current_step()

    def go_next(self):
        if self.current_step < len(self.history) - 1:
            self.current_step += 1
            self.show_current_step()

    def go_start(self):
        self.current_step = 0
        self.show_current_step()

    def go_end(self):
        self.current_step = len(self.history) - 1
        self.show_current_step()


# =============================================================================
# GUI - GŁÓWNA APLIKACJA
# =============================================================================

class TransportationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Zagadnienie Transportowe - Wersja Pełna (5.0)")
        self.root.geometry("1000x800")
        self.solver = None  # Przechowuje instancję solvera po obliczeniach

        # --- GÓRNY PASEK Z POMOCĄ ---
        top_bar = ttk.Frame(root)
        top_bar.pack(fill="x", padx=5, pady=5)
        ttk.Button(top_bar, text="Instrukcja / Pomoc", command=self.show_help).pack(side="right")

        # 1. Panel Konfiguracyjny
        config_frame = ttk.LabelFrame(root, text="1. Konfiguracja Macierzy")
        config_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(config_frame, text="Liczba Dostawców:").pack(side="left", padx=5)
        self.spin_rows = ttk.Spinbox(config_frame, from_=2, to=10, width=5)
        self.spin_rows.set(3)
        self.spin_rows.pack(side="left", padx=5)

        ttk.Label(config_frame, text="Liczba Odbiorców:").pack(side="left", padx=5)
        self.spin_cols = ttk.Spinbox(config_frame, from_=2, to=10, width=5)
        self.spin_cols.set(4)
        self.spin_cols.pack(side="left", padx=5)

        ttk.Button(config_frame, text="Generuj Tabelę", command=self.create_grid).pack(side="left", padx=20)

        # 2. Panel Metod
        method_frame = ttk.LabelFrame(root, text="2. Wybór Metody")
        method_frame.pack(fill="x", padx=10, pady=5)

        self.method_var = tk.StringVar(value="NWCM")

        methods = [
            ("Kąt Północno-Zachodni (NWCM)", "NWCM"),
            ("Minimalny Element (LCM)", "LCM"),
            ("Minimum Wierszowe", "ROW_MIN"),
            ("Aproksymacja Vogela (VAM)", "VAM"),
        ]

        for text, val in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.method_var, value=val).pack(side="left", padx=5)

        self.opt_var = tk.BooleanVar()
        ttk.Checkbutton(method_frame, text="Doptymalizuj (MODI)", variable=self.opt_var).pack(side="left", padx=20)

        # Przyciski akcji
        action_frame = ttk.Frame(method_frame)
        action_frame.pack(side="right", padx=10)

        ttk.Button(action_frame, text="Oblicz (Wynik Końcowy)", command=self.solve).pack(side="left", padx=5)
        self.btn_steps = ttk.Button(action_frame, text="Pokaż Kroki", command=self.open_steps, state="disabled")
        self.btn_steps.pack(side="left", padx=5)

        # 3. Kontener na tabelę
        self.grid_container = ttk.Frame(root)
        self.grid_container.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(self.grid_container)
        self.scrollbar_y = ttk.Scrollbar(self.grid_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar_y.pack(side="right", fill="y")

        self.entries_cost = []
        self.entries_supply = []
        self.entries_demand = []

        # 4. Panel Wyników
        res_frame = ttk.LabelFrame(root, text="3. Wynik Końcowy")
        res_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.result_text = tk.Text(res_frame, height=10, font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.create_grid()

    def show_help(self):
        """Wyświetla okno z instrukcją (Wymóg na ocenę 5)."""
        help_text = (
            "INSTRUKCJA OBSŁUGI PROGRAMU\n\n"
            "1. PRZEZNACZENIE:\n"
            "Program służy do rozwiązywania Zagadnienia Transportowego (minimalizacja kosztów).\n\n"
            "2. KROKI POSTĘPOWANIA:\n"
            "   a) Ustal liczbę dostawców i odbiorców, kliknij 'Generuj Tabelę'.\n"
            "   b) Wypełnij tabelę kosztami jednostkowymi (środek), podażą (prawa kolumna) i popytem (dolny wiersz).\n"
            "   c) Wybierz metodę rozwiązania startowego (np. NWCM, VAM).\n"
            "   d) (Opcjonalnie) Zaznacz 'Doptymalizuj (MODI)', aby znaleźć rozwiązanie optymalne.\n"
            "   e) Kliknij 'Oblicz (Wynik Końcowy)'.\n\n"
            "3. WIZUALIZACJA KROKÓW:\n"
            "   Po obliczeniu wyniku, aktywuje się przycisk 'Pokaż Kroki'.\n"
            "   Kliknij go, aby otworzyć okno, w którym możesz prześledzić cały proces alokacji towaru.\n\n"
            "4. UWAGI:\n"
            "   Program automatycznie bilansuje zadania (dodaje fikcyjnych dostawców/odbiorców)."
        )
        messagebox.showinfo("Pomoc", help_text)

    def create_grid(self):
        for widget in self.scrollable_frame.winfo_children(): widget.destroy()
        self.entries_cost = []
        self.entries_supply = []
        self.entries_demand = []
        self.btn_steps.config(state="disabled")  # Reset przycisku kroków

        try:
            r = int(self.spin_rows.get())
            c = int(self.spin_cols.get())
        except ValueError:
            return

        # Generowanie nagłówków i pól (kod identyczny jak wcześniej)
        ttk.Label(self.scrollable_frame, text="Dost\\Odb").grid(row=0, column=0, padx=5, pady=5)
        for j in range(c): ttk.Label(self.scrollable_frame, text=f"Odb {j + 1}", font=("Arial", 9, "bold")).grid(row=0,
                                                                                                                 column=j + 1)
        ttk.Label(self.scrollable_frame, text="PODAŻ", foreground="blue", font=("Arial", 9, "bold")).grid(row=0,
                                                                                                          column=c + 1)

        for i in range(r):
            ttk.Label(self.scrollable_frame, text=f"Dost {i + 1}", font=("Arial", 9, "bold")).grid(row=i + 1, column=0,
                                                                                                   padx=5)
            row_entries = []
            for j in range(c):
                e = ttk.Entry(self.scrollable_frame, width=6, justify="center")
                e.insert(0, str(np.random.randint(1, 20)))
                e.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                row_entries.append(e)
            self.entries_cost.append(row_entries)
            s = ttk.Entry(self.scrollable_frame, width=8, justify="center", foreground="blue")
            s.insert(0, str(np.random.randint(50, 100)))
            s.grid(row=i + 1, column=c + 1, padx=5)
            self.entries_supply.append(s)

        ttk.Label(self.scrollable_frame, text="POPYT", foreground="green", font=("Arial", 9, "bold")).grid(row=r + 1,
                                                                                                           column=0)
        for j in range(c):
            d = ttk.Entry(self.scrollable_frame, width=6, justify="center", foreground="green")
            d.insert(0, str(np.random.randint(50, 100)))
            d.grid(row=r + 1, column=j + 1, padx=2, pady=5)
            self.entries_demand.append(d)

    def get_data(self):
        try:
            costs = [[float(e.get()) for e in row] for row in self.entries_cost]
            supply = [float(e.get()) for e in self.entries_supply]
            demand = [float(e.get()) for e in self.entries_demand]
            return costs, supply, demand
        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby.")
            return None, None, None

    def solve(self):
        costs, supply, demand = self.get_data()
        if costs is None: return

        try:
            self.solver = TransportationSolver(costs, supply, demand)
            method = self.method_var.get()

            if method == "NWCM":
                self.solver.solve_nwcm()
            elif method == "LCM":
                self.solver.solve_lcm()
            elif method == "ROW_MIN":
                self.solver.solve_row_min()
            elif method == "VAM":
                self.solver.solve_vam()

            start_cost = self.solver.calculate_total_cost()
            final_cost = start_cost
            was_optimized = False

            if self.opt_var.get():
                self.solver.solve_optimal_modi()
                final_cost = self.solver.calculate_total_cost()
                was_optimized = True

            self.display_results(start_cost, final_cost, was_optimized)
            self.btn_steps.config(state="normal")  # Aktywuj przycisk kroków

        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def display_results(self, start_cost, final_cost, was_optimized):
        txt = f"METODA: {self.method_var.get()}\n"
        txt += f"Koszt początkowy: {start_cost:.2f}\n"
        if was_optimized:
            txt += f"Koszt optymalny (MODI): {final_cost:.2f}\n"

        # Wyświetlenie końcowej tabeli w oknie głównym
        txt += "\nTABELA KOŃCOWA:\n"
        headers = [""] + [f"Odb{j + 1}" for j in range(self.solver.cols)]
        col_width = 8
        row_fmt = "{:<8}" + "".join([f"{{:^{col_width}}}" for _ in range(self.solver.cols)])
        txt += row_fmt.format(*headers) + "\n"
        txt += "-" * (8 + self.solver.cols * col_width) + "\n"

        for i in range(self.solver.rows):
            row_vals = [f"{v:g}" if v > 0 else "." for v in self.solver.allocation[i]]
            txt += row_fmt.format(f"Dost{i + 1}", *row_vals) + "\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, txt)

    def open_steps(self):
        if self.solver and self.solver.history:
            StepViewer(self.root, self.solver.history, self.solver.rows, self.solver.cols)
        else:
            messagebox.showinfo("Info", "Brak kroków do wyświetlenia.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TransportationApp(root)
    root.mainloop()