from scipy.interpolate import Akima1DInterpolator
from scipy.optimize import curve_fit
from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import logging


# ===================== НАСТРОЙКИ =====================

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.figsize": (7, 6),
    "figure.dpi": 150,
    "figure.autolayout": True
})


USE_BASELINE_IN_ABEL = True


# ===================== ГАУСС ДЛЯ ПРОСТРАНСТВЕННЫХ ПРОФИЛЕЙ =====================

def gaussian_model(r, y0, A, sigma):
    """
    Пространственная гауссова аппроксимация с ненулевым фоном.

    r     — радиус, мм
    y0    — baseline / фон
    A     — амплитуда пика над baseline
    sigma — ширина
    """
    return y0 + A * np.exp(-(r ** 2) / (2.0 * sigma ** 2))


def gaussian_peak_only(r, A, sigma):
    """
    Только гауссов пик без baseline.
    """
    return A * np.exp(-(r ** 2) / (2.0 * sigma ** 2))


def gaussian_half_peak_area(A, sigma, r_max):
    """
    Площадь под половиной гауссового пика от 0 до r_max.

    ВАЖНО:
    y0 сюда не входит, потому что y0 — это baseline.
    """
    if not np.isfinite(A) or not np.isfinite(sigma):
        return np.nan

    if sigma <= 0 or r_max <= 0:
        return np.nan

    return A * sigma * math.sqrt(math.pi / 2.0) * math.erf(
        r_max / (math.sqrt(2.0) * sigma)
    )


def estimate_y0_from_outer_edge(y_values):
    """
    Оценка y0 по внешнему краю половины профиля.

    После сортировки r идёт от центра к краю:
        r = 0       — центр
        r = r_max   — внешний край

    Поэтому baseline берём по последним точкам.
    """
    y_values = np.asarray(y_values, dtype=float)

    if len(y_values) == 0:
        return 0.0

    edge_n = max(1, len(y_values) // 10)
    edge_values = y_values[-edge_n:]

    return float(np.median(edge_values))


def fit_half_gaussian(r_branch, i_branch):
    """
    Аппроксимация одной половины пространственного профиля.

    На вход:
        r_branch — abs(radius), от 0 до края
        i_branch — интегральные площади спектральной линии

    На выход:
        словарь с параметрами фита и гладкой кривой
    """
    r_branch = np.asarray(r_branch, dtype=float)
    i_branch = np.asarray(i_branch, dtype=float)

    mask = np.isfinite(r_branch) & np.isfinite(i_branch)
    r_branch = r_branch[mask]
    i_branch = i_branch[mask]

    if len(r_branch) < 5:
        return None

    order = np.argsort(r_branch)
    r_branch = r_branch[order]
    i_branch = i_branch[order]

    r_max = float(np.max(r_branch))
    if r_max <= 0:
        return None

    y0_init = estimate_y0_from_outer_edge(i_branch)

    A_init = float(np.max(i_branch) - y0_init)
    if not np.isfinite(A_init) or A_init <= 0:
        A_init = float(np.ptp(i_branch)) if np.ptp(i_branch) > 0 else 1.0

    sigma_init = r_max / 2.0
    if not np.isfinite(sigma_init) or sigma_init <= 0:
        sigma_init = 0.1

    try:
        popt, pcov = curve_fit(
            gaussian_model,
            r_branch,
            i_branch,
            p0=[y0_init, A_init, sigma_init],
            bounds=(
                [-np.inf, 0.0, 1e-12],
                [np.inf, np.inf, np.inf]
            ),
            maxfev=20000
        )

    except Exception as e:
        logging.error(f"Ошибка гауссовой аппроксимации половины профиля: {e}")
        return None

    y0, A, sigma = [float(v) for v in popt]

    i_fit_full = gaussian_model(r_branch, y0, A, sigma)
    i_fit_peak = gaussian_peak_only(r_branch, A, sigma)

    residual = i_branch - i_fit_full
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((i_branch - np.mean(i_branch)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    area_peak = gaussian_half_peak_area(A, sigma, r_max)
    area_total_with_y0 = area_peak + y0 * r_max

    return {
        "y0": y0,
        "A": A,
        "Sigma": sigma,
        "R2": r2,
        "I_fit_full": i_fit_full,
        "I_fit_peak": i_fit_peak,
        "Area_peak": area_peak,
        "Area_total_with_y0": area_total_with_y0,
        "Covariance": pcov,
    }


# ===================== БОКАСТЕН / АБЕЛЬ =====================

def generate_bockasten_matrix(n):
    """
    Генерация матрицы коэффициентов a_jk по формулам Бокастена.
    """
    a = np.zeros((n, n))

    def eval_int(p, lower, upper, j_val):
        def F(x):
            if j_val == 0:
                if p == 2:
                    return 0.5 * x ** 2
                if p == 1:
                    return x
                if p == 0:
                    return math.log(x)
            else:
                root = math.sqrt(x ** 2 - j_val ** 2)

                if p == 2:
                    return (
                        0.5 * x * root
                        + 0.5 * j_val ** 2 * math.log(x + root)
                    )

                if p == 1:
                    return root

                if p == 0:
                    return math.log(x + root)

            raise ValueError(f"Unsupported p={p}")

        return F(upper) - F(lower)

    for m in range(n):
        if m == 0:
            coeffs = {
                0: [0, -3.5, 2.25],
                1: [0, 4, -3],
                2: [0, -0.5, 0.75]
            }

        elif m == n - 1:
            coeffs = {
                n - 2: [0.5 - n, 1, 0],
                n - 1: [2 * n - 2, -2, 0]
            }

        else:
            coeffs = {
                m - 1: [-0.5 * m ** 2 - m - 1 / 3, m + 1, -0.5],
                m: [1.5 * m ** 2 + 2 * m - 0.5, -(3 * m + 2), 1.5],
                m + 1: [-1.5 * m ** 2 - m + 1, 3 * m + 1, -1.5],
                m + 2: [0.5 * m ** 2 - 1 / 6, -m, 0.5]
            }

            coeffs = {k: v for k, v in coeffs.items() if k < n}

        for k, c_arr in coeffs.items():
            c0, c1, c2 = c_arr

            for j in range(n):
                if m < j:
                    continue

                val = (
                    c2 * eval_int(2, m, m + 1, j)
                    + c1 * eval_int(1, m, m + 1, j)
                )

                if not (j == 0 and m == 0 and c0 == 0):
                    val += c0 * eval_int(0, m, m + 1, j)

                a[j, k] += (-n / math.pi) * val

    return a


def bockasten_abel_transformation(N, coefficients, r_max):
    """
    Преобразование Абеля по Бокастену.

    N      — профиль интенсивности вдоль половины
    r_max  — максимальный радиус половины, мм
    """
    e = coefficients @ N
    return e / (r_max * 0.001)


# ===================== MAIN =====================

def main():
    file_path = Path("results") / "All_spectra_areas.xlsx"

    if not file_path.exists():
        file_path = Path("All_spectra_areas.xlsx")

        if not file_path.exists():
            logging.error(f"Файл данных не найден: {file_path}")
            return

    data = pd.read_excel(file_path)

    required_columns = {"Line", "Radius_mm", "Area"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        logging.error(f"В файле нет нужных колонок: {missing_columns}")
        return

    data = data.copy()
    data["Radius_mm"] = pd.to_numeric(data["Radius_mm"], errors="coerce")
    data["Area"] = pd.to_numeric(data["Area"], errors="coerce")
    data = data.dropna(subset=["Line", "Radius_mm", "Area"])

    lines = data["Line"].unique()

    out_dir = Path.cwd() / "results"
    out_dir.mkdir(exist_ok=True)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    color_map = {
        line: colors[idx % len(colors)]
        for idx, line in enumerate(lines)
    }

    abel_results = pd.DataFrame()
    raw_plots_data = []
    half_area_rows = []

    plt.figure(figsize=(8, 6))

    print("\n--- Раздельная гауссова аппроксимация половин и преобразование Абеля ---")

    for line_name in lines:
        line_data = data[data["Line"] == line_name].copy()

        radii_all = line_data["Radius_mm"].values.astype(float)
        intensities_all = line_data["Area"].values.astype(float)

        mask_left = radii_all <= 0
        mask_right = radii_all >= 0

        for branch_name, mask in (("left", mask_left), ("right", mask_right)):
            branch_df = pd.DataFrame({
                "R_abs": np.abs(radii_all[mask]),
                "I": intensities_all[mask]
            })

            branch_df = branch_df[
                np.isfinite(branch_df["R_abs"])
                & np.isfinite(branch_df["I"])
            ]

            if branch_df.empty:
                logging.warning(f"Нет данных для {line_name} ({branch_name})")
                continue

            # На случай дублей по одному радиусу усредняем значения.
            # Например, r=0 может попасть и в left, и в right.
            branch_df = (
                branch_df
                .groupby("R_abs", as_index=False)["I"]
                .mean()
                .sort_values("R_abs")
            )

            r_branch = branch_df["R_abs"].values
            i_branch = branch_df["I"].values
            n_points = len(r_branch)

            if n_points < 5:
                logging.warning(f"Мало точек для {line_name} ({branch_name}): {n_points}")
                continue

            r_max = float(r_branch.max())
            if r_max <= 0:
                logging.warning(f"Нулевой r_max для {line_name} ({branch_name})")
                continue

            fit = fit_half_gaussian(r_branch, i_branch)

            if fit is None:
                logging.warning(
                    f"{line_name} ({branch_name}): фит не выполнен, "
                    f"для Абеля будут использованы исходные точки"
                )

                i_branch_smooth = i_branch.copy()

                y0_fit = np.nan
                A_fit = np.nan
                sigma_fit = np.nan
                r2_fit = np.nan
                area_peak = np.nan
                area_total = np.nan

            else:
                if USE_BASELINE_IN_ABEL:
                    i_branch_smooth = fit["I_fit_full"]
                else:
                    i_branch_smooth = fit["I_fit_peak"]

                y0_fit = fit["y0"]
                A_fit = fit["A"]
                sigma_fit = fit["Sigma"]
                r2_fit = fit["R2"]
                area_peak = fit["Area_peak"]
                area_total = fit["Area_total_with_y0"]

            half_area_rows.append({
                "Line": line_name,
                "Half": branch_name,
                "N_points": n_points,
                "R_max_mm": r_max,
                "y0": y0_fit,
                "A": A_fit,
                "Sigma": sigma_fit,
                "R2": r2_fit,
                "Gaussian_peak_area_0_to_Rmax": area_peak,
                "Gaussian_total_area_with_y0_0_to_Rmax": area_total,
            })

            # --- Контрольный график аппроксимации ---
            r_plot = -r_branch if branch_name == "left" else r_branch

            plt.scatter(
                r_plot,
                i_branch,
                color=color_map[line_name],
                alpha=0.35,
                s=20
            )

            plt.plot(
                r_plot,
                i_branch_smooth,
                color=color_map[line_name],
                linestyle="--" if branch_name == "left" else "-",
                lw=2,
                label=f"{line_name} {branch_name}"
            )

            # --- Абель отдельно для каждой половины ---
            bockasten_coefficients = generate_bockasten_matrix(n_points)
            abelized_intensities = bockasten_abel_transformation(
                i_branch_smooth,
                bockasten_coefficients,
                r_max
            )

            col_prefix = f"{line_name}_{branch_name}"

            abel_block = pd.DataFrame({
                f"{col_prefix}_R": pd.Series(r_branch),
                f"{col_prefix}_Eps": pd.Series(abelized_intensities),
                f"{col_prefix}_I_raw": pd.Series(i_branch),
                f"{col_prefix}_I_smooth": pd.Series(i_branch_smooth),
                f"{col_prefix}_y0": pd.Series(np.full(n_points, y0_fit)),
                f"{col_prefix}_A": pd.Series(np.full(n_points, A_fit)),
                f"{col_prefix}_Sigma": pd.Series(np.full(n_points, sigma_fit)),
                f"{col_prefix}_R2": pd.Series(np.full(n_points, r2_fit)),
            })

            abel_results = pd.concat([abel_results, abel_block], axis=1)
            raw_plots_data.append((line_name, branch_name, r_branch, abelized_intensities))

    # ===================== СВОДКА ПЛОЩАДЕЙ ПОЛОВИН =====================

    half_area_df = pd.DataFrame(half_area_rows)

    if not half_area_df.empty:
        half_area_df["Mean_peak_area_for_line"] = (
            half_area_df
            .groupby("Line")["Gaussian_peak_area_0_to_Rmax"]
            .transform("mean")
        )

        half_area_df["Delta_from_mean"] = (
            half_area_df["Gaussian_peak_area_0_to_Rmax"]
            - half_area_df["Mean_peak_area_for_line"]
        )

        half_area_df["Delta_from_mean_percent"] = np.where(
            half_area_df["Mean_peak_area_for_line"].abs() > 0,
            100.0
            * half_area_df["Delta_from_mean"]
            / half_area_df["Mean_peak_area_for_line"],
            np.nan
        )

        out_area_summary = out_dir / "half_gaussian_area_summary.xlsx"
        half_area_df.to_excel(out_area_summary, index=False)

        print("\n--- Отличие площади каждой половины от среднего по линии ---")

        for _, row in half_area_df.iterrows():
            area = row["Gaussian_peak_area_0_to_Rmax"]
            mean_area = row["Mean_peak_area_for_line"]
            delta = row["Delta_from_mean"]
            delta_pct = row["Delta_from_mean_percent"]

            print(
                f"{row['Line']} | {row['Half']:>5}: "
                f"Area={area:.6g}, "
                f"Mean={mean_area:.6g}, "
                f"Delta={delta:+.6g} "
                f"({delta_pct:+.2f}%), "
                f"y0={row['y0']:.6g}, "
                f"A={row['A']:.6g}, "
                f"Sigma={row['Sigma']:.6g}, "
                f"R2={row['R2']:.4f}"
            )

        print(f"\nСводка площадей половин сохранена в {out_area_summary}")

    else:
        logging.warning("Сводка площадей пустая: half_area_df не создан.")

    # ===================== ГРАФИК КОНТРОЛЯ СГЛАЖИВАНИЯ =====================

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Integrated line area [a.u.]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8, loc="best")
    plt.savefig(out_dir / "smoothing_check.png", dpi=250)
    plt.show()

    # ===================== СОХРАНЕНИЕ АБЕЛЯ =====================

    if abel_results.empty:
        logging.error("Abel results пустой. Нечего сохранять.")
        return

    out_abel = out_dir / "abel_results_gaussian.xlsx"
    abel_results.to_excel(out_abel, index=False)

    print(f"\nПрофили ε(r) сохранены в {out_abel}")

    # ===================== ГРАФИК ε(r) =====================

    print("Построение профилей ε(r)...")

    plt.figure()

    for line_name, branch_name, r, eps in raw_plots_data:
        r_plot = -r if branch_name == "left" else r

        if branch_name == "right":
            label = line_name
        else:
            label = "_nolegend_"

        plt.plot(
            r_plot,
            eps,
            color=color_map[line_name],
            lw=2,
            label=label
        )

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Emissivity $\\varepsilon(r)$ [W/m$^3$]")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.savefig(out_dir / "epsilon_vs_radius.png", dpi=250)
    plt.show()

    # ===================== ВЫРАВНИВАНИЕ НА ОБЩУЮ СЕТКУ =====================

    left_limits = [
        np.max(r)
        for _, branch_name, r, _ in raw_plots_data
        if branch_name == "left" and len(r) > 0
    ]

    right_limits = [
        np.max(r)
        for _, branch_name, r, _ in raw_plots_data
        if branch_name == "right" and len(r) > 0
    ]

    limit_left = min(left_limits) if left_limits else 0
    limit_right = min(right_limits) if right_limits else 0

    print(
        f"\nГраницы выравнивания по минимальному размеру: "
        f"Left {limit_left:.3f} mm, Right {limit_right:.3f} mm"
    )

    grid_points = 100

    common_grid_left = (
        np.linspace(0, limit_left, grid_points)
        if limit_left > 0
        else np.array([])
    )

    common_grid_right = (
        np.linspace(0, limit_right, grid_points)
        if limit_right > 0
        else np.array([])
    )

    aligned_left_df = (
        pd.DataFrame({"Radius_mm": common_grid_left})
        if limit_left > 0
        else pd.DataFrame()
    )

    aligned_right_df = (
        pd.DataFrame({"Radius_mm": common_grid_right})
        if limit_right > 0
        else pd.DataFrame()
    )

    for line_name, branch_name, r, eps in raw_plots_data:
        r = np.asarray(r, dtype=float)
        eps = np.asarray(eps, dtype=float)

        mask = np.isfinite(r) & np.isfinite(eps)
        r = r[mask]
        eps = eps[mask]

        if len(r) < 2:
            logging.warning(f"Мало точек для интерполяции: {line_name} {branch_name}")
            continue

        order = np.argsort(r)
        r = r[order]
        eps = eps[order]

        # На случай повторяющихся радиусов
        interp_df = (
            pd.DataFrame({"R": r, "Eps": eps})
            .groupby("R", as_index=False)["Eps"]
            .mean()
            .sort_values("R")
        )

        r = interp_df["R"].values
        eps = interp_df["Eps"].values

        if len(r) < 2:
            continue

        interpolator = Akima1DInterpolator(r, eps)

        if branch_name == "left" and limit_left > 0:
            aligned_left_df[line_name] = interpolator(common_grid_left)

        elif branch_name == "right" and limit_right > 0:
            aligned_right_df[line_name] = interpolator(common_grid_right)

    if not aligned_left_df.empty and not aligned_right_df.empty:
        aligned_final = pd.concat(
            [
                aligned_left_df.add_suffix("_Left"),
                aligned_right_df.add_suffix("_Right")
            ],
            axis=1
        )

    elif not aligned_left_df.empty:
        aligned_final = aligned_left_df.add_suffix("_Left")

    elif not aligned_right_df.empty:
        aligned_final = aligned_right_df.add_suffix("_Right")

    else:
        logging.error("Не удалось создать aligned_final.")
        return

    out_aligned = out_dir / "aligned_results.xlsx"
    aligned_final.to_excel(out_aligned, index=False)

    print(f"Выровненные профили сохранены в {out_aligned}")

    # ===================== ГРАФИК ВЫРОВНЕННЫХ ПРОФИЛЕЙ =====================

    print("Построение выровненных графиков...")

    plt.figure()

    for line_name in lines:
        col_color = color_map[line_name]

        if (
            "Radius_mm_Left" in aligned_final.columns
            and f"{line_name}_Left" in aligned_final.columns
        ):
            r_l = aligned_final["Radius_mm_Left"]
            e_l = aligned_final[f"{line_name}_Left"]

            plt.plot(
                -r_l,
                e_l,
                linestyle="--",
                color=col_color,
                lw=2,
                label=f"{line_name} (Left)"
            )

        if (
            "Radius_mm_Right" in aligned_final.columns
            and f"{line_name}_Right" in aligned_final.columns
        ):
            r_r = aligned_final["Radius_mm_Right"]
            e_r = aligned_final[f"{line_name}_Right"]

            plt.plot(
                r_r,
                e_r,
                linestyle="-",
                color=col_color,
                lw=2,
                label=f"{line_name} (Right)"
            )

    plt.xlabel("Radius $r$ [mm]")
    plt.ylabel("Aligned $\\varepsilon(r)$ [W/m$^3$]")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig(out_dir / "aligned_epsilon.png", bbox_inches="tight", dpi=250)
    plt.show()


if __name__ == "__main__":
    main()