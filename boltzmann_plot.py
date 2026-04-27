import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import re # Добавь это в самое начало файла!

# ====== НАСТРОЙКИ ======
in_xlsx  = "aligned_results.xlsx"
out_dir  = "results"
r_ref_mm = 0.0  # Опорная точка для построения одного графика Больцмана
os.makedirs(out_dir, exist_ok=True)

K_B_EV = 8.617333262e-5  # eV/K

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.figsize": (7, 6),
    "figure.dpi": 150,
    "figure.autolayout": True
})

# База данных линий: длина волны_нм : (Энергия_верхнего_уровня_эВ, g*f)
LINE_DB = {
    510.5537: (3.816948, 0.0197),
    515.3230: (6.191593, 1.64659),
    521.8197: (6.192444, 1.97166876),
    578.2127: (3.78615 , 0.013),
    465.1119: (7.737547, 1.4217765),
    #570.0237: (3.816948, 0.00565054),
}

def annotate_inside_axes(ax, x, y, text):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # По умолчанию — вправо и вверх
    dx, dy = 6, 6
    ha, va = 'left', 'bottom'

    # Если точка близко к правому краю — смещаем влево
    if x > x_max - 0.12 * x_range:
        dx = -6
        ha = 'right'
    elif x < x_min + 0.12 * x_range:
        dx = 6
        ha = 'left'

    # Если точка близко к верхнему краю — смещаем вниз
    if y > y_max - 0.12 * y_range:
        dy = -6
        va = 'top'
    elif y < y_min + 0.12 * y_range:
        dy = 6
        va = 'bottom'

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords='offset points',
        ha=ha,
        va=va,
        clip_on=True
    )

def header_to_nm(col):
    s = str(col).strip().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*nm\b", s, flags=re.IGNORECASE)
    if m is None:
        m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m is None:
        return np.nan
    v = float(m.group(1))
    return v / 10.0 if v > 1000 else v


def nearest_nm(nm, tol=0.35):
    if not np.isfinite(nm):
        return None
    ks = np.array(list(LINE_DB.keys()), dtype=float)
    idx = int(np.argmin(np.abs(ks - nm)))
    key = float(ks[idx])
    if abs(key - nm) > tol:
        return None
    return key

def compute_profile(df, side: str, radius_col: str, line_cols: list[str]):
    # Преобразование данных в числа
    radii = pd.to_numeric(df[radius_col], errors="coerce").to_numpy(float)
    
    # Маппинг колонок на ключи из LINE_DB
    mapping = {}
    for c in line_cols:
        wave = header_to_nm(c)
        key = nearest_nm(wave)
        if key: mapping[c] = key

    # --- Построение графика Больцмана в опорной точке (r ≈ 0) ---
    idx_ref = int(np.nanargmin(np.abs(radii - r_ref_mm)))
    rows = []
    for col, nm in mapping.items():
        eps = float(pd.to_numeric(df[col].iloc[idx_ref], errors="coerce"))
        if not np.isfinite(eps) or eps <= 0:
            continue
        E, gf = LINE_DB[nm]
        Y = np.log(eps * (nm**3) / gf)
        rows.append((nm, E, gf, eps, Y))

    boltz = pd.DataFrame(rows, columns=["lambda_nm","E_eV","gf","epsilon","Y"]).sort_values("E_eV")
    
    T_ref, T_ref_err = np.nan, np.nan
    if boltz["E_eV"].nunique() >= 2:
        res = linregress(boltz["E_eV"], boltz["Y"])
        T_ref = -1.0 / (K_B_EV * res.slope) if res.slope != 0 else np.nan
        T_ref_err = res.stderr / (K_B_EV * res.slope**2) if res.slope != 0 else np.nan
        
        plt.figure(figsize=(7, 5))
        ax = plt.gca()

        plt.scatter(boltz["E_eV"], boltz["Y"], color='red', zorder=3)

        ex = np.array([boltz["E_eV"].min(), boltz["E_eV"].max()])
        plt.plot(ex, res.intercept + res.slope * ex, alpha=0.5)

        plt.title(f"Boltzmann Plot ({side}) r={radii[idx_ref]:.2f}mm\nT = {T_ref:.0f} K")
        plt.xlabel("Upper Energy E [eV]")
        plt.ylabel("ln(ε λ³ / gf)")
        plt.grid(True, linestyle=':')
        plt.draw()

        for _, r in boltz.iterrows():
            annotate_inside_axes(
                ax,
                r["E_eV"],
                r["Y"],
                f"{r['lambda_nm']:.1f}"
            )

        plt.savefig(os.path.join(out_dir, f"boltzmann_{side}.png"))
        plt.show()

    # --- Расчет профиля T(r) по всем точкам радиуса ---
    T_vals, Terr_vals = [], []
    for i in range(len(radii)):
        X, Ylist = [], []
        for col, nm in mapping.items():
            eps = pd.to_numeric(df[col].iloc[i], errors="coerce")
            if np.isfinite(eps) and eps > 0:
                E, gf = LINE_DB[nm]
                X.append(E)
                Ylist.append(np.log(eps * (nm**3) / gf))
        
        if len(set(X)) >= 2:
            lr = linregress(X, Ylist)
            Ti = -1.0/(K_B_EV * lr.slope) if lr.slope != 0 else np.nan
            dTi = lr.stderr/(K_B_EV * lr.slope**2) if lr.slope != 0 else np.nan
        else:
            Ti, dTi = np.nan, np.nan
        
        T_vals.append(Ti)
        Terr_vals.append(dTi)

    prof = pd.DataFrame({
        "Radius_mm": radii,
        "T_K": T_vals,
        "T_err": Terr_vals
    })
    prof.to_excel(os.path.join(out_dir, f"radial_T_{side}.xlsx"), index=False)
    return prof, (radii[idx_ref], T_ref, T_ref_err)



# ====== ЗАПУСК ======
# Загрузка данных
if not os.path.exists(in_xlsx):
    if os.path.exists("aligned_results.xlsx - Sheet1.csv"):
        df = pd.read_csv("aligned_results.xlsx - Sheet1.csv")
    else:
        print(f"Файл {in_xlsx} не найден!")
        exit()
else:
    df = pd.read_excel(in_xlsx)

# 1. Очистка заголовков от мусора и пробелов
df.columns = [str(c).strip() for c in df.columns]

# 2. Автоматический поиск колонок радиусов
# Ищем колонку, где есть слово 'Radius' и направление
try:
    radius_left = [c for c in df.columns if "Radius" in c and ("Left" in c or "left" in c)][0]
    radius_right = [c for c in df.columns if "Radius" in c and ("Right" in c or "right" in c)][0]
except IndexError:
    print("ОШИБКА: Не нашел колонки с радиусом!")
    print("Список всех колонок в файле:", df.columns.tolist())
    exit()

# 3. Поиск колонок с линиями (все, где есть Left/Right, но это не радиус)
left_cols  = [c for c in df.columns if ("Left" in c or "left" in c) and c != radius_left]
right_cols = [c for c in df.columns if ("Right" in c or "right" in c) and c != radius_right]

print(f"--- ДИАГНОСТИКА ---")
print(f"Радиус слева:  {radius_left}")
print(f"Радиус справа: {radius_right}")
print(f"Линии слева:   {left_cols}")
print(f"Линии справа:  {right_cols}")
print(f"-------------------")

# Если линий 0, принудительно выходим
if len(left_cols) == 0 or len(right_cols) == 0:
    print("ОШИБКА: Линии не найдены. Проверьте названия в Excel.")
    exit()

# Запуск расчетов
prof_left,  ref_left  = compute_profile(df, "left",  radius_left,  left_cols)
prof_right, ref_right = compute_profile(df, "right", radius_right, right_cols)

# --- Итоговый график T(r) ---
plt.figure(figsize=(9, 6))

if not prof_left["T_K"].isna().all():
    plt.errorbar(-prof_left["Radius_mm"], prof_left["T_K"], yerr=prof_left["T_err"], 
                 fmt='o', markersize=4, capsize=3, label="Left side", alpha=0.8, color='#1f77b4')

if not prof_right["T_K"].isna().all():
    plt.errorbar(prof_right["Radius_mm"], prof_right["T_K"], yerr=prof_right["T_err"], 
                 fmt='s', markersize=4, capsize=3, label="Right side", alpha=0.8, color='#ff7f0e')

plt.axvline(0, color='black', lw=1.2, linestyle='--')
plt.xlabel("Radius [mm]", fontsize=12)
plt.ylabel("Temperature [K]", fontsize=12)
plt.title("Radial Temperature Profile", fontsize=14)
plt.legend()
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "T_profile_final.png"), dpi=250)
plt.show()

print(f"\nРЕЗУЛЬТАТ В ЦЕНТРЕ:")
print(f"Слева:  T = {ref_left[1]:.0f} ± {ref_left[2]:.0f} K")
print(f"Справа: T = {ref_right[1]:.0f} ± {ref_right[2]:.0f} K")
