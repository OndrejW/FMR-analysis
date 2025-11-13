import re
import io
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# --- SciPy / Matplotlib ---
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
import matplotlib
matplotlib.use("TkAgg")  # embed in Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseButton

# --- Tkinter UI ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# =========================
#   MODELS
# =========================
def lorentz(x, A, x0, gamma, C):
    """Absorption Lorentz: y = C + A * ((g/2)^2 / ((x-x0)^2 + g2)); gamma is FWHM"""
    g2 = (gamma / 2.0) ** 2
    return C + A * (g2 / ((x - x0) ** 2 + g2))

def dlorentz_dx(x, A, x0, gamma, C):
    """First derivative of Lorentz (dispersion-like)."""
    g2 = (gamma / 2.0) ** 2
    return C + A * (-2.0) * g2 * (x - x0) / (((x - x0) ** 2 + g2) ** 2)

MODEL_FUNCS = {
    "Derivative of Lorentz (dispersion-like)": dlorentz_dx,
    "Lorentz (absorption)": lorentz,
}

# =========================
#   HELPERS
# =========================
FREQ_REGEX = re.compile(r"f\s*([0-9]*\.?[0-9]+)\s*GHz", re.IGNORECASE)

def parse_frequency_from_name(name: str) -> Optional[float]:
    m = FREQ_REGEX.search(name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def read_two_column_txt_bytes(file_bytes: bytes):
    raw = file_bytes.decode(errors="ignore")
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty file")
    # try direct
    try:
        data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
        if data.ndim == 1 or data.shape[1] < 2:
            raise ValueError("Expected two columns")
    except Exception:
        # skip first line (header)
        if len(lines) < 2:
            raise
        data = np.loadtxt(io.StringIO("\n".join(lines[1:])), delimiter=",")
        if data.ndim == 1 or data.shape[1] < 2:
            raise ValueError("Expected two columns after skipping header")
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    return x, y

def auto_initial_guess(x: np.ndarray, y: np.ndarray, model_name: str):
    C = float(np.median(y))
    y0 = y - C
    span = float(np.max(x) - np.min(x)) or 1.0

    if model_name == "Lorentz (absorption)":
        i_max = int(np.argmax(y0))
        i_min = int(np.argmin(y0))
        if abs(y0[i_max]) >= abs(y0[i_min]):
            x0 = float(x[i_max]); A = float(y0[i_max])
        else:
            x0 = float(x[i_min]); A = float(y0[i_min])
        gamma = 0.1 * span
    else:
        dy = np.gradient(y0, x)
        i_slope = int(np.argmax(np.abs(dy)))
        sign = np.sign(y0)
        changes = np.where(np.diff(sign) != 0)[0]
        if len(changes) > 0:
            zidx = min(changes, key=lambda k: abs(k - i_slope))
            x0 = float(np.interp(0.0, [y0[zidx], y0[zidx + 1]], [x[zidx], x[zidx + 1]]))
        else:
            x0 = float(x[i_slope])
        A = float(np.max(np.abs(y0)))
        gamma = 0.1 * span

    gamma = max(gamma, 1e-6)
    return A, x0, gamma, C

def r2(y, yfit):
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

# =========================
#   DATA MODEL
# =========================
@dataclass
class FitState:
    file_path: str
    file_name: str
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray        # True = include in fit; False = excluded
    model: str
    freq_GHz: float
    A_init: float
    x0_init: float
    gamma_init: float
    C_init: float
    # results
    A: Optional[float] = None
    x0: Optional[float] = None
    gamma: Optional[float] = None
    C: Optional[float] = None
    R2: Optional[float] = None
    # Uncertainties (95%): single ± values
    A_unc: Optional[float] = None
    x0_unc: Optional[float] = None
    gamma_unc: Optional[float] = None

# =========================
#   MAIN APP
# =========================
class FMRFitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FMR Lorentz Fitter (Tk)")
        self.geometry("1380x900")

        # State
        self.fits: Dict[str, FitState] = {}  # key by file path
        self.selected_key: Optional[str] = None

        # Global bounds
        self.min_gamma = tk.DoubleVar(value=1e-6)
        self.max_abs_A = tk.DoubleVar(value=1e9)

        # Masking controls
        self.click_masking_enabled = tk.BooleanVar(value=False)
        self.area_masking_enabled = tk.BooleanVar(value=False)
        self.area_action = tk.StringVar(value="mask")  # mask | unmask | toggle
        self._rect_selector: Optional[RectangleSelector] = None

        # ---- Layout ----
        self._build_menu()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._build_bottom_table()

        self._refresh_file_list()
        self._refresh_table()

    # -------- UI Builders --------
    def _build_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=False)
        filemenu.add_command(label="Open files…", command=self.open_files)
        filemenu.add_separator()
        filemenu.add_command(label="Export CSV…", command=self.export_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def _build_left_panel(self):
        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=8, pady=8)

        ttk.Label(left, text="Files").pack(anchor="w")
        lb_frame = ttk.Frame(left); lb_frame.pack(fill="y", expand=True)
        self.file_list = tk.Listbox(lb_frame, height=18, exportselection=False)
        vs = ttk.Scrollbar(lb_frame, orient="vertical", command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=vs.set)
        self.file_list.grid(row=0, column=0, sticky="ns"); vs.grid(row=0, column=1, sticky="ns")
        lb_frame.grid_columnconfigure(0, weight=1)
        self.file_list.bind("<<ListboxSelect>>", self.on_select_file)

        btns = ttk.Frame(left); btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Open files…", command=self.open_files).pack(side="left", padx=2)
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(side="left", padx=2)
        ttk.Button(btns, text="Remove all", command=self.remove_all).pack(side="left", padx=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        gb = ttk.LabelFrame(left, text="Global bounds"); gb.pack(fill="x")
        ttk.Label(gb, text="Min linewidth (FWHM):").grid(row=0, column=0, sticky="w")
        ttk.Entry(gb, textvariable=self.min_gamma, width=14).grid(row=0, column=1, sticky="e")
        ttk.Label(gb, text="Max |A|:").grid(row=1, column=0, sticky="w")
        ttk.Entry(gb, textvariable=self.max_abs_A, width=14).grid(row=1, column=1, sticky="e")

        ttk.Button(left, text="Fit ALL", command=self.fit_all).pack(fill="x", pady=8)

        # Masking controls
        mask_box = ttk.LabelFrame(left, text="Masking")
        mask_box.pack(fill="x", pady=6)
        ttk.Checkbutton(mask_box, text="Enable click-masking", variable=self.click_masking_enabled).pack(anchor="w")
        ttk.Button(mask_box, text="Clear masks (selected)", command=self.clear_masks_selected).pack(fill="x", pady=3)
        ttk.Separator(mask_box, orient="horizontal").pack(fill="x", pady=4)
        ttk.Checkbutton(mask_box, text="Enable area selection", variable=self.area_masking_enabled,
                        command=self._toggle_rect_selector).pack(anchor="w")
        action_row = ttk.Frame(mask_box); action_row.pack(fill="x", pady=3)
        ttk.Label(action_row, text="Area action:").pack(side="left")
        ttk.Radiobutton(action_row, text="Mask", value="mask", variable=self.area_action).pack(side="left")
        ttk.Radiobutton(action_row, text="Unmask", value="unmask", variable=self.area_action).pack(side="left")
        ttk.Radiobutton(action_row, text="Toggle", value="toggle", variable=self.area_action).pack(side="left")
        ttk.Button(mask_box, text="Disable selection", command=self._disable_rect_selector).pack(fill="x", pady=3)

    def _build_center_panel(self):
        center = ttk.Frame(self); center.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        ctrl = ttk.LabelFrame(center, text="Per-file controls"); ctrl.pack(fill="x")
        self.model_var = tk.StringVar(value=list(MODEL_FUNCS.keys())[0])
        self.freq_var = tk.DoubleVar(value=0.0)
        self.A0_var = tk.DoubleVar(value=0.0)
        self.x00_var = tk.DoubleVar(value=0.0)
        self.g0_var = tk.DoubleVar(value=0.0)
        self.C0_var = tk.DoubleVar(value=0.0)

        r = 0
        ttk.Label(ctrl, text="Model:").grid(row=r, column=0, sticky="w", padx=3, pady=2)
        ttk.OptionMenu(ctrl, self.model_var, self.model_var.get(), *MODEL_FUNCS.keys()).grid(row=r, column=1, sticky="w")
        ttk.Label(ctrl, text="Frequency (GHz):").grid(row=r, column=2, sticky="w", padx=3)
        ttk.Entry(ctrl, textvariable=self.freq_var, width=14).grid(row=r, column=3, sticky="w")
        r += 1
        ttk.Label(ctrl, text="A init:").grid(row=r, column=0, sticky="w", padx=3)
        ttk.Entry(ctrl, textvariable=self.A0_var, width=14).grid(row=r, column=1, sticky="w")
        ttk.Label(ctrl, text="x0 init (Hres):").grid(row=r, column=2, sticky="w", padx=3)
        ttk.Entry(ctrl, textvariable=self.x00_var, width=14).grid(row=r, column=3, sticky="w")
        r += 1
        ttk.Label(ctrl, text="gamma init (FWHM):").grid(row=r, column=0, sticky="w", padx=3)
        ttk.Entry(ctrl, textvariable=self.g0_var, width=14).grid(row=r, column=1, sticky="w")
        ttk.Label(ctrl, text="offset init (C):").grid(row=r, column=2, sticky="w", padx=3)
        ttk.Entry(ctrl, textvariable=self.C0_var, width=14).grid(row=r, column=3, sticky="w")
        r += 1

        btnrow = ttk.Frame(ctrl); btnrow.grid(row=r, column=0, columnspan=4, sticky="w", pady=4)
        ttk.Button(btnrow, text="Refit selected", command=self.fit_selected).pack(side="left", padx=2)
        ttk.Button(btnrow, text="Update initials", command=self.update_initials_from_ui).pack(side="left", padx=2)
        ttk.Button(btnrow, text="Auto-guess initials", command=self.autoguess_selected).pack(side="left", padx=2)

        plots = ttk.Frame(center); plots.pack(fill="both", expand=True, pady=6)
        self.fig = Figure(figsize=(7.6, 4.8), dpi=100)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_res = self.fig.add_subplot(212, sharex=self.ax_main)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plots)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Click-masking hookup
        self._mpl_cid = self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_right_panel(self):
        right = ttk.Frame(self); right.pack(side="right", fill="y", padx=8, pady=8)

        res = ttk.LabelFrame(right, text="Selected fit results (95% ± unc)")
        res.pack(fill="x")

        self.lab_freq = ttk.Label(res, text="f [GHz]: —")
        self.lab_hres = ttk.Label(res, text="Hres: —")
        self.lab_fwhm = ttk.Label(res, text="FWHM: —")
        self.lab_amp = ttk.Label(res, text="Amplitude: —")
        self.lab_off = ttk.Label(res, text="Offset: —")
        self.lab_r2 = ttk.Label(res, text="R²: —")
        for w in (self.lab_freq, self.lab_hres, self.lab_fwhm, self.lab_amp, self.lab_off, self.lab_r2):
            w.pack(anchor="w")

        act = ttk.LabelFrame(right, text="Results CSV")
        act.pack(fill="x", pady=10)
        ttk.Button(act, text="Copy table to clipboard", command=self.copy_csv_clipboard).pack(fill="x", pady=2)
        ttk.Button(act, text="Save CSV…", command=self.export_csv).pack(fill="x", pady=2)

        ttk.Label(right, text="Tip: Use area selection to (un)mask many points at once.").pack(anchor="w", pady=8)

    def _build_bottom_table(self):
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="both", padx=8, pady=8, expand=True)

        ttk.Label(bottom, text="Fitted parameters table (editable; scrollable):").pack(anchor="w")
        cols = (
            "file", "frequency_GHz",
            "amplitude", "amplitude_unc",
            "resonance_field", "resonance_field_unc",
            "linewidth_FWHM", "linewidth_FWHM_unc",
            "offset", "R2", "model"
        )
        tree_frame = ttk.Frame(bottom); tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            w = 180 if c in ("file",) else 140
            self.tree.column(c, width=w, anchor="w")
        vs = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew"); vs.grid(row=0, column=1, sticky="ns"); hs.grid(row=1, column=0, sticky="ew")
        tree_frame.grid_rowconfigure(0, weight=1); tree_frame.grid_columnconfigure(0, weight=1)
        self.tree.bind("<Double-1>", self._on_table_double_click)

    # -------- Actions --------
    def open_files(self):
        paths = filedialog.askopenfilenames(
            title="Select FMR text files",
            filetypes=[("Text/CSV", "*.txt *.csv"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            try:
                with open(p, "rb") as fh:
                    x, y = read_two_column_txt_bytes(fh.read())
                fname = p.split("/")[-1].split("\\")[-1]
                freq = parse_frequency_from_name(fname)
                if freq is None: freq = float("nan")
                model = list(MODEL_FUNCS.keys())[0]
                A0, x00, g0, C0 = auto_initial_guess(x, y, model)
                mask = np.ones_like(x, dtype=bool)
                self.fits[p] = FitState(
                    file_path=p, file_name=fname, x=x, y=y, mask=mask,
                    model=model, freq_GHz=freq,
                    A_init=A0, x0_init=x00, gamma_init=g0, C_init=C0
                )
            except Exception as e:
                messagebox.showerror("Load failed", f"{p}\n\n{e}")
        self._refresh_file_list()
        if self.selected_key is None and self.fits:
            self._select_first()
        self._refresh_table()
        self._update_plot_and_labels()

    def remove_selected(self):
        key = self.selected_key
        if key and key in self.fits:
            del self.fits[key]
            self.selected_key = None
            self._refresh_file_list()
            self._refresh_table()
            self._update_plot_and_labels()

    def remove_all(self):
        if not self.fits:
            return
        if messagebox.askyesno("Remove all", "Remove all loaded files?"):
            self.fits.clear()
            self.selected_key = None
            self._refresh_file_list()
            self._refresh_table()
            self._update_plot_and_labels()

    def on_select_file(self, event):
        idx = self.file_list.curselection()
        if not idx:
            return
        key = self.file_list.get(idx[0])
        self.selected_key = key
        self._load_selected_into_controls()
        self._update_plot_and_labels()

    def _select_first(self):
        if not self.fits: return
        first_key = list(self.fits.keys())[0]
        self.selected_key = first_key
        self._refresh_file_list()
        self.file_list.selection_clear(0, tk.END)
        try:
            idx = list(self.fits.keys()).index(first_key)
            self.file_list.selection_set(idx)
        except Exception:
            pass
        self._load_selected_into_controls()

    def _load_selected_into_controls(self):
        fs = self._get_selected()
        if not fs: return
        self.model_var.set(fs.model)
        self.freq_var.set(0.0 if math.isnan(fs.freq_GHz) else fs.freq_GHz)
        self.A0_var.set(fs.A_init); self.x00_var.set(fs.x0_init)
        self.g0_var.set(fs.gamma_init); self.C0_var.set(fs.C_init)

    def update_initials_from_ui(self):
        fs = self._get_selected()
        if not fs: return
        fs.model = self.model_var.get()
        fs.freq_GHz = self.freq_var.get()
        fs.A_init = self.A0_var.get()
        fs.x0_init = self.x00_var.get()
        fs.gamma_init = self.g0_var.get()
        fs.C_init = self.C0_var.get()
        self._refresh_table()

    def autoguess_selected(self):
        fs = self._get_selected()
        if not fs: return
        A0, x00, g0, C0 = auto_initial_guess(fs.x, fs.y, self.model_var.get())
        self.A0_var.set(A0); self.x00_var.set(x00); self.g0_var.set(g0); self.C0_var.set(C0)

    def fit_all(self):
        if not self.fits: return
        for _, fs in self.fits.items():
            self._fit_one(fs)
        self._refresh_table()
        self._update_plot_and_labels()

    def fit_selected(self):
        fs = self._get_selected()
        if not fs: return
        self.update_initials_from_ui()
        self._fit_one(fs)
        self._refresh_table()
        self._update_plot_and_labels()

    def _fit_one(self, fs: FitState):
        try:
            f = MODEL_FUNCS.get(fs.model, lorentz)
            xm = fs.x[fs.mask]; ym = fs.y[fs.mask]
            if xm.size < 4:
                raise ValueError("Not enough unmasked points to fit (need ≥ 4).")

            lower = [-abs(self.max_abs_A.get()), -np.inf, max(self.min_gamma.get(), 1e-12), -np.inf]
            upper = [ abs(self.max_abs_A.get()),  np.inf, np.inf,                          np.inf]
            popt, pcov = curve_fit(
                f, xm, ym,
                p0=[fs.A_init, fs.x0_init, fs.gamma_init, fs.C_init],
                bounds=(lower, upper),
                maxfev=10000
            )
            fs.A, fs.x0, fs.gamma, fs.C = map(float, popt)
            yfit_full = f(fs.x, *popt)
            fs.R2 = float(r2(fs.y, yfit_full))

            # --- 95% uncertainties (±) ---
            se = np.sqrt(np.diag(pcov))
            dof = max(1, xm.size - len(popt))
            tcrit = student_t.ppf(0.975, dof)  # two-sided 95%
            unc = tcrit * se
            fs.A_unc, fs.x0_unc, fs.gamma_unc = float(unc[0]), float(unc[1]), float(unc[2])

        except Exception as e:
            messagebox.showerror("Fit failed", f"{fs.file_name}\n\n{e}")

    # -------- Masking (click + area) --------
    def _on_plot_click(self, event):
        if not self.click_masking_enabled.get():
            return
        fs = self._get_selected()
        if not fs or event.inaxes != self.ax_main:
            return
        xdata, ydata = fs.x, fs.y
        if xdata.size == 0: return
        trans = self.ax_main.transData.transform
        pts = trans(np.column_stack([xdata, ydata]))
        ex, ey = event.x, event.y
        d2 = (pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2
        idx = int(np.argmin(d2))
        if d2[idx] <= 9**2:
            fs.mask[idx] = ~fs.mask[idx]
            self._update_plot_and_labels()

    def _toggle_rect_selector(self):
        """Enable/disable the RectangleSelector for area masking, compatible with multiple Matplotlib versions."""
        # Turn off an existing selector first
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
                self._rect_selector.disconnect_events()
            except Exception:
                pass
            self._rect_selector = None

        if self.area_masking_enabled.get():
            # Matplotlib >=3.3 doesn’t need 'drawtype'; some versions error if it’s present.
            # Keep args minimal & broadly compatible.
            try:
                self._rect_selector = RectangleSelector(
                    self.ax_main,
                    self._on_rect_select,           # (eclick, erelease)
                    useblit=False,                  # safer across platforms/backends
                    button=[MouseButton.LEFT],      # or just button=1 if you prefer
                    minspanx=0, minspany=0,
                    spancoords='data',
                    interactive=False
                )
            except TypeError:
                # Very old MPL fallback (some expected different button type)
                self._rect_selector = RectangleSelector(
                    self.ax_main,
                    self._on_rect_select,
                    useblit=False,
                    button=1,                       # plain left button
                    minspanx=0, minspany=0,
                    spancoords='data',
                    interactive=False
                )

            try:
                self._rect_selector.set_active(True)
            except Exception:
                pass
        # force a redraw so the selector box appears immediately when used
        self.canvas.draw_idle()

    def _disable_rect_selector(self):
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
                self._rect_selector.disconnect_events()
            except Exception:
                pass
            self._rect_selector = None
        self.area_masking_enabled.set(False)
        self.canvas.draw_idle()

    def _on_rect_select(self, eclick, erelease):
        fs = self._get_selected()
        if not fs: return
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        ymin, ymax = sorted([eclick.ydata, erelease.ydata])
        inside = (fs.x >= xmin) & (fs.x <= xmax) & (fs.y >= ymin) & (fs.y <= ymax)
        if self.area_action.get() == "mask":
            fs.mask[inside] = False
        elif self.area_action.get() == "unmask":
            fs.mask[inside] = True
        else:  # toggle
            fs.mask[inside] = ~fs.mask[inside]
        self._update_plot_and_labels()

    def clear_masks_selected(self):
        fs = self._get_selected()
        if not fs: return
        fs.mask[:] = True
        self._update_plot_and_labels()

    # -------- Table handling --------
    def _row_dicts(self):
        out = []
        for fs in self.fits.values():
            if fs.A is None:
                continue
            out.append({
                "file": fs.file_name,
                "frequency_GHz": fs.freq_GHz,
                "amplitude": fs.A,
                "amplitude_unc": fs.A_unc,
                "resonance_field": fs.x0,
                "resonance_field_unc": fs.x0_unc,
                "linewidth_FWHM": fs.gamma,
                "linewidth_FWHM_unc": fs.gamma_unc,
                "offset": fs.C,
                "R2": fs.R2,
                "model": fs.model,
                "_key": fs.file_path,
            })
        return out

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in self._row_dicts():
            values = (
                row["file"], row["frequency_GHz"],
                row["amplitude"], row["amplitude_unc"],
                row["resonance_field"], row["resonance_field_unc"],
                row["linewidth_FWHM"], row["linewidth_FWHM_unc"],
                row["offset"], row["R2"], row["model"]
            )
            self.tree.insert("", "end", iid=row["_key"], values=values)

    def _on_table_double_click(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or not column:
            return
        col_index = int(column.replace("#", "")) - 1
        bbox = self.tree.bbox(item, column)
        if not bbox:
            return
        x, y, w, h = bbox
        value = self.tree.set(item, self.tree["columns"][col_index])

        entry = tk.Entry(self.tree)
        entry.insert(0, str(value))
        entry.select_range(0, tk.END)
        entry.focus()
        entry.place(x=x, y=y, width=w, height=h)

        def commit(_=None):
            new_val = entry.get()
            entry.destroy()
            vals = list(self.tree.item(item, "values"))
            vals[col_index] = new_val
            self.tree.item(item, values=vals)
            self._apply_table_row_to_state(item, vals)

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)

    def _apply_table_row_to_state(self, key, vals):
        if key not in self.fits:
            return
        fs = self.fits[key]
        try:
            fs.freq_GHz             = float(vals[1])
            fs.A                    = float(vals[2]); fs.A_unc = float(vals[3])
            fs.x0                   = float(vals[4]); fs.x0_unc = float(vals[5])
            fs.gamma                = float(vals[6]); fs.gamma_unc = float(vals[7])
            fs.C                    = float(vals[8])
            fs.R2                   = float(vals[9]) if vals[9] != "nan" else float("nan")
            fs.model                = str(vals[10])
        except Exception:
            pass
        self._update_plot_and_labels()

    def copy_csv_clipboard(self):
        rows = self._row_dicts()
        if not rows:
            messagebox.showinfo("Copy CSV", "No fitted results to copy."); return
        df = pd.DataFrame(rows).drop(columns=["_key"])
        csv_text = df.to_csv(index=False)
        self.clipboard_clear(); self.clipboard_append(csv_text)
        messagebox.showinfo("Copy CSV", "Results copied to clipboard.")

    def export_csv(self):
        rows = self._row_dicts()
        if not rows:
            messagebox.showinfo("Save CSV", "No fitted results to save."); return
        df = pd.DataFrame(rows).drop(columns=["_key"])
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path: return
        try:
            df.to_csv(path, index=False)
            messagebox.showinfo("Save CSV", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save CSV failed", str(e))

    # -------- Plot + labels --------
    def _update_plot_and_labels(self):
        self.ax_main.clear(); self.ax_res.clear()
        fs = self._get_selected()
        if fs:
            inc = fs.mask
            self.ax_main.plot(fs.x[inc], fs.y[inc], "o", ms=3, label="Data (used)")
            if (~inc).any():
                self.ax_main.plot(fs.x[~inc], fs.y[~inc], "x", ms=5, label="Masked")

            if fs.A is not None:
                f = MODEL_FUNCS.get(fs.model, lorentz)
                xx = np.linspace(fs.x.min(), fs.x.max(), 1500)
                yy = f(xx, fs.A, fs.x0, fs.gamma, fs.C)
                self.ax_main.plot(xx, yy, label="Fit")
                self.ax_main.axvline(fs.x0, linestyle="--", alpha=0.6, label="Resonance")
                resid = fs.y - f(fs.x, fs.A, fs.x0, fs.gamma, fs.C)
                self.ax_res.plot(fs.x, resid, ".", ms=3)
                self.ax_res.axhline(0, ls="--", alpha=0.5)
                r2txt = f"{fs.R2:.4f}" if fs.R2 is not None else "—"
            else:
                r2txt = "—"

            self.ax_main.set_title(fs.file_name)
            self.ax_main.set_xlabel("Field"); self.ax_main.set_ylabel("Signal")
            self.ax_main.legend(loc="best")
            self.ax_res.set_xlabel("Field"); self.ax_res.set_ylabel("Residuals")

            ftxt = "—" if math.isnan(fs.freq_GHz) else f"{fs.freq_GHz:.6g}"
            self.lab_freq.config(text=f"f [GHz]: {ftxt}")

            def fmt_pm(val, unc):
                if val is None: return "—"
                if unc is None: return f"{val:.6g}"
                return f"{val:.6g} ± {unc:.2g}"

            self.lab_hres.config(text=f"Hres: {fmt_pm(fs.x0, fs.x0_unc)}")
            self.lab_fwhm.config(text=f"FWHM: {fmt_pm(fs.gamma, fs.gamma_unc)}")
            self.lab_amp.config(text=f"Amplitude: {fmt_pm(fs.A, fs.A_unc)}")
            self.lab_off.config(text=f"Offset: {('—' if fs.C is None else f'{fs.C:.6g}')}")
            self.lab_r2.config(text=f"R²: {r2txt}")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _refresh_file_list(self):
        self.file_list.delete(0, tk.END)
        for key in self.fits.keys():
            self.file_list.insert(tk.END, key)

    def _get_selected(self) -> Optional[FitState]:
        if self.selected_key and self.selected_key in self.fits:
            return self.fits[self.selected_key]
        idxs = self.file_list.curselection()
        if idxs:
            key = self.file_list.get(idxs[0])
            return self.fits.get(key)
        return None


if __name__ == "__main__":
    app = FMRFitApp()
    app.mainloop()
