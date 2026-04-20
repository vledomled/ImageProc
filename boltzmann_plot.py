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

# База данных линий: длина волны_нм : (Энергия_верхнего_уровня_эВ, g*f)
LINE_DB = {
    510.5537: (3.816948, 0.0197),
    515.3230: (6.191593, 1.64659),
    521.8197: (6.192444, 1.97166876),
    578.2127: (3.78615 , 0.013),
    465.1119: (7.737547, 1.4217765),
    570.0237: (3.816948, 0.00565054),
}

def header_to_nm(col):
    """Извлекает длину волны из заголовка колонки типа 'Cu I 510.5 nm_Left'."""
    s = str(col).strip().replace(",", ".")
    # Оставляем только цифры и точку
    digits = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    try:
        v = float(digits)
        # Если вдруг в названии 5105 (в ангстремах), переводим в нм
        return v / 10.0 if v > 1000 else v
    except ValueError:
        return np.nan

def nearest_nm(nm):
    """Находит ближайшую длину волны в базе данных."""
    if np.isnan(nm): return None
    ks = np.array(list(LINE_DB.keys()), dtype=float)
    return float(ks[np.argmin(np.abs(ks - nm))])

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
        
        # Рисуем проверочный график Больцмана
        plt.figure(figsize=(7, 5))
        plt.scatter(boltz["E_eV"], boltz["Y"], color='red', zorder=3)
        for _, r in boltz.iterrows():
            plt.annotate(f"{r['lambda_nm']:.1f}", (r['E_eV'], r['Y']), xytext=(5,5), textcoords='offset points')
        
        ex = np.array([boltz["E_eV"].min(), boltz["E_eV"].max()])
        plt.plot(ex, res.intercept + res.slope*ex, 'k--', alpha=0.5)
        plt.title(f"Boltzmann Plot ({side}) r={radii[idx_ref]:.2f}mm\nT = {T_ref:.0f} K")
        plt.xlabel("Upper Energy E [eV]")
        plt.ylabel("ln(ε λ³ / gf)")
        plt.grid(True, linestyle=':')
        plt.savefig(os.path.join(out_dir, f"boltzmann_{side}.png"))
        plt.close()

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
