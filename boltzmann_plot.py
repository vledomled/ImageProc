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
c_m_s = 3e+8
h = 6.63e-34
part_func_txt = "partition_func.txt"
NE_CONST = 66700000000000

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

def load_partition_function_txt(path):

    if not os.path.exists(path):
        print(f"ПРЕДУПРЕЖДЕНИЕ: файл статсуммы не найден: {path}")
        return None

    try:
        pf = pd.read_csv(
            path,
            sep=r"[\s,;]+",
            comment="#",
            engine="python",
            header=None
        )
    except Exception as e:
        print(f"ОШИБКА чтения файла статсуммы {path}: {e}")
        return None

    if pf.shape[1] < 2:
        print(f"ОШИБКА: в {path} должно быть минимум 2 колонки: T_K и part_func")
        return None

    pf = pf.iloc[:, :2].copy()
    pf.columns = ["T_K", "part_func"]

    pf["T_K"] = pd.to_numeric(pf["T_K"], errors="coerce")
    pf["part_func"] = pd.to_numeric(pf["part_func"], errors="coerce")

    pf = pf.dropna(subset=["T_K", "part_func"])

    if pf.empty:
        print(f"ОШИБКА: не удалось прочитать численные значения из {path}")
        return None

    pf = (
        pf
        .groupby("T_K", as_index=False)["part_func"]
        .mean()
        .sort_values("T_K")
    )

    print("\n--- Статсумма загружена ---")
    print(f"Файл: {path}")
    print(f"T range: {pf['T_K'].min():.1f} ... {pf['T_K'].max():.1f} K")
    print(f"Точек: {len(pf)}")

    return pf


def get_partition_function_with_error(T_K, T_err, pf_df):

    if pf_df is None:
        return np.nan, np.nan

    if not np.isfinite(T_K):
        return np.nan, np.nan

    T_arr = pf_df["T_K"].to_numpy(float)
    U_arr = pf_df["part_func"].to_numpy(float)

    if len(T_arr) < 2:
        return np.nan, np.nan

    if T_K < T_arr.min() or T_K > T_arr.max():
        return np.nan, np.nan

    U = np.interp(T_K, T_arr, U_arr)

    if not np.isfinite(T_err):
        return U, np.nan

    dU_dT_arr = np.gradient(U_arr, T_arr)
    dU_dT = np.interp(T_K, T_arr, dU_dT_arr)

    U_err = abs(dU_dT) * abs(T_err)

    return U, U_err

def calc_electron_concentration(intercept_b, intercept_b_err, T_K, T_err, pf_df):

    if not np.isfinite(intercept_b) or not np.isfinite(T_K):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    U, U_err = get_partition_function_with_error(
        T_K=T_K,
        T_err=T_err,
        pf_df=pf_df
    )

    if not np.isfinite(U) or U <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    Ne = (
        np.exp(intercept_b)
        * U
        / NE_CONST
        / c_m_s
        / h
        * 1e-9
    )

    if (
        np.isfinite(Ne)
        and np.isfinite(intercept_b_err)
        and np.isfinite(U_err)
        and U > 0
    ):
        Ne_rel_err = np.sqrt(
            intercept_b_err ** 2
            + (U_err / U) ** 2
        )

        Ne_err = Ne * Ne_rel_err
    else:
        Ne_rel_err = np.nan
        Ne_err = np.nan

    return Ne, Ne_err, Ne_rel_err, U, U_err

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

def make_lognormal_yerr(y, rel_err):
    y = np.asarray(y, dtype=float)
    rel_err = np.asarray(rel_err, dtype=float)

    y_low = y * np.exp(-rel_err)
    y_high = y * np.exp(rel_err)

    lower = y - y_low
    upper = y_high - y

    return np.vstack([lower, upper])


def nearest_nm(nm, tol=0.35):
    if not np.isfinite(nm):
        return None
    ks = np.array(list(LINE_DB.keys()), dtype=float)
    idx = int(np.argmin(np.abs(ks - nm)))
    key = float(ks[idx])
    if abs(key - nm) > tol:
        return None
    return key

def compute_profile(df, side: str, radius_col: str, line_cols: list[str], part_func_df=None):
    # Преобразование данных в числа
    radii = pd.to_numeric(df[radius_col], errors="coerce").to_numpy(float)

    # Маппинг колонок на ключи из LINE_DB
    mapping = {}

    for c in line_cols:
        wave = header_to_nm(c)
        key = nearest_nm(wave)

        if key is not None:
            mapping[c] = key

    if len(mapping) == 0:
        print(f"ОШИБКА: для стороны {side} не найдено ни одной линии из LINE_DB")

        prof = pd.DataFrame({
            "Radius_mm": radii,
            "T_K": np.nan,
            "T_err": np.nan,
            "Intercept_b": np.nan,
            "Intercept_b_err": np.nan,
            "Partition_func": np.nan,
            "Partition_func_err": np.nan,
            "Electron_concentration": np.nan,
            "Electron_concentration_err": np.nan
        })

        prof.to_excel(os.path.join(out_dir, f"radial_T_{side}.xlsx"), index=False)
        return prof, (np.nan, np.nan, np.nan)

    # --- Построение графика Больцмана в опорной точке r_ref_mm ---
    idx_ref = int(np.nanargmin(np.abs(radii - r_ref_mm)))

    rows = []

    for col, nm in mapping.items():
        eps = float(pd.to_numeric(df[col].iloc[idx_ref], errors="coerce"))

        if not np.isfinite(eps) or eps <= 0:
            continue

        E, gf = LINE_DB[nm]
        Y = np.log(eps * (nm ** 3) / gf)

        rows.append((nm, E, gf, eps, Y))

    boltz = pd.DataFrame(
        rows,
        columns=["lambda_nm", "E_eV", "gf", "epsilon", "Y"]
    ).sort_values("E_eV")

    T_ref = np.nan
    T_ref_err = np.nan

    if not boltz.empty and boltz["E_eV"].nunique() >= 2:
        res = linregress(boltz["E_eV"], boltz["Y"])

        if res.slope != 0:
            T_ref = -1.0 / (K_B_EV * res.slope)
            T_ref_err = res.stderr / (K_B_EV * res.slope ** 2)

            plt.figure(figsize=(7, 5))
            ax = plt.gca()

            plt.scatter(
                boltz["E_eV"],
                boltz["Y"],
                color="red",
                zorder=3
            )

            ex = np.array([
                boltz["E_eV"].min(),
                boltz["E_eV"].max()
            ])

            plt.plot(
                ex,
                res.intercept + res.slope * ex,
                "k--",
                alpha=0.6
            )

            plt.title(
                f"Boltzmann Plot ({side}) r={radii[idx_ref]:.2f} mm\n"
                f"T = {T_ref:.0f} ± {T_ref_err:.0f} K"
            )

            plt.xlabel("Upper Energy E [eV]")
            plt.ylabel("ln(ε λ³ / gf)")
            plt.grid(True, linestyle=":")

            plt.draw()

            for _, row in boltz.iterrows():
                annotate_inside_axes(
                    ax,
                    row["E_eV"],
                    row["Y"],
                    f"{row['lambda_nm']:.1f}"
                )

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"boltzmann_{side}.png"), dpi=250)
            plt.show()

    # --- Расчёт профилей T(r), intercept, U(T), Ne(r) ---
    T_vals = []
    Terr_vals = []

    B_vals = []
    Berr_vals = []

    Part_vals = []
    Part_err_vals = []

    Ne_vals = []
    Ne_err_vals = []

    for i in range(len(radii)):
        X = []
        Ylist = []

        for col, nm in mapping.items():
            eps = pd.to_numeric(df[col].iloc[i], errors="coerce")

            if np.isfinite(eps) and eps > 0:
                E, gf = LINE_DB[nm]
                X.append(E)
                Ylist.append(np.log(eps * (nm ** 3) / gf))

        if len(set(X)) >= 2:
            lr = linregress(X, Ylist)

            if lr.slope != 0:
                Ti = -1.0 / (K_B_EV * lr.slope)
                dTi = lr.stderr / (K_B_EV * lr.slope ** 2)

                b = lr.intercept
                b_err = getattr(lr, "intercept_stderr", np.nan)

                Ne, Ne_err, part_func, part_func_err = calc_electron_concentration(
                    intercept_b=b,
                    intercept_b_err=b_err,
                    T_K=Ti,
                    T_err=dTi,
                    pf_df=part_func_df
                )

            else:
                Ti = np.nan
                dTi = np.nan
                b = np.nan
                b_err = np.nan
                part_func = np.nan
                part_func_err = np.nan
                Ne = np.nan
                Ne_err = np.nan

        else:
            Ti = np.nan
            dTi = np.nan
            b = np.nan
            b_err = np.nan
            part_func = np.nan
            part_func_err = np.nan
            Ne = np.nan
            Ne_err = np.nan

        T_vals.append(Ti)
        Terr_vals.append(dTi)

        B_vals.append(b)
        Berr_vals.append(b_err)

        Part_vals.append(part_func)
        Part_err_vals.append(part_func_err)

        Ne_vals.append(Ne)
        Ne_err_vals.append(Ne_err)

    prof = pd.DataFrame({
        "Radius_mm": radii,
        "T_K": T_vals,
        "T_err": Terr_vals,
        "Intercept_b": B_vals,
        "Intercept_b_err": Berr_vals,
        "Partition_func": Part_vals,
        "Partition_func_err": Part_err_vals,
        "Electron_concentration": Ne_vals,
        "Electron_concentration_err": Ne_err_vals
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
part_func_df = load_partition_function_txt(part_func_txt)
prof_left,  ref_left  = compute_profile(df, "left",  radius_left,  left_cols, part_func_df)
prof_right, ref_right = compute_profile(df, "right", radius_right, right_cols, part_func_df)

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

# --- График концентрации электронов ---
plt.figure(figsize=(9, 6))

if "Electron_concentration" in prof_left.columns:
    mask_l = (
        np.isfinite(prof_left["Radius_mm"])
        & np.isfinite(prof_left["Electron_concentration"])
        & np.isfinite(prof_left["Electron_concentration_err"])
        & (prof_left["Electron_concentration"] > 0)
        & (prof_left["Electron_concentration_err"] >= 0)
    )

    if mask_l.any():
        plt.errorbar(
            -prof_left.loc[mask_l, "Radius_mm"],
            prof_left.loc[mask_l, "Electron_concentration"],
            yerr=prof_left.loc[mask_l, "Electron_concentration_err"],
            fmt="o",
            markersize=4,
            capsize=3,
            label="Left side",
            alpha=0.8,
            color="#1f77b4"
        )

if "Electron_concentration" in prof_right.columns:
    mask_r = (
        np.isfinite(prof_right["Radius_mm"])
        & np.isfinite(prof_right["Electron_concentration"])
        & np.isfinite(prof_right["Electron_concentration_err"])
        & (prof_right["Electron_concentration"] > 0)
        & (prof_right["Electron_concentration_err"] >= 0)
    )

    if mask_r.any():
        plt.errorbar(
            prof_right.loc[mask_r, "Radius_mm"],
            prof_right.loc[mask_r, "Electron_concentration"],
            yerr=prof_right.loc[mask_r, "Electron_concentration_err"],
            fmt="s",
            markersize=4,
            capsize=3,
            label="Right side",
            alpha=0.8,
            color="#ff7f0e"
        )

#plt.yscale("log")
plt.ylim(1e20, 1e22)
plt.axvline(0, color="black", lw=1.2, linestyle="--")
plt.xlabel("Radius [mm]", fontsize=12)
plt.ylabel("Number density []", fontsize=12)
plt.legend()
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "electron_concentration_profile.png"), dpi=250)
plt.show()

print(f"\nРЕЗУЛЬТАТ В ЦЕНТРЕ:")
print(f"Слева:  T = {ref_left[1]:.0f} ± {ref_left[2]:.0f} K")
print(f"Справа: T = {ref_right[1]:.0f} ± {ref_right[2]:.0f} K")
