import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ====== НАСТРОЙКИ ======
in_xlsx  = "aligned_results.xlsx"
out_dir  = "results"
r_ref_mm = 0.0
os.makedirs(out_dir, exist_ok=True)

K_B_EV = 8.617333262e-5  # eV/K

# wavelength_nm : (E_eV, gf)
LINE_DB = {
    510.5537: (3.816948, 0.0197),
    515.3230: (6.191593, 1.64659),
    521.8197: (6.192444, 1.97166876),
    578.2127: (3.78615 , 0.013),
    465.1119: (7.737547, 1.4217765),
    570.0237: (3.816948, 0.00565054),
}

def header_to_nm(col):
    # "5105_left_L1" -> 5105 -> 510.5 nm
    s = str(col).strip().replace(",", ".")
    digits = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    v = float(digits) if digits else np.nan
    return v / 10.0 if v > 1000 else v

def nearest_nm(nm):
    ks = np.array(list(LINE_DB.keys()), dtype=float)
    return float(ks[np.argmin(np.abs(ks - nm))])

def compute_profile(df, side: str, radius_col: str, line_cols: list[str]):
    radii = pd.to_numeric(df[radius_col], errors="coerce").to_numpy(float)

    # mapping: col -> nearest wavelength key in LINE_DB
    mapping = {c: nearest_nm(header_to_nm(c)) for c in line_cols}

    # --- болцман в опорной точке ---
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

    uniqE_ref = int(boltz["E_eV"].nunique(dropna=True)) if len(boltz) else 0
    if uniqE_ref >= 2:
        res = linregress(boltz["E_eV"], boltz["Y"])
        slope, intercept, stderr = res.slope, res.intercept, res.stderr
        T_ref = -1.0/(K_B_EV * slope) if slope != 0 else np.nan
        T_ref_err = stderr/(K_B_EV * slope**2) if slope != 0 else np.nan
    else:
        slope = intercept = stderr = np.nan
        T_ref = T_ref_err = np.nan
        print(f"[{side}] skip Boltzmann at r≈{radii[idx_ref]:.3f} mm: unique E = {uniqE_ref} (<2)")

    # сохранить болцман-точки (даже если не строили — полезно для диагностики)
    boltz.to_excel(os.path.join(out_dir, f"boltzmann_points_{side}_r{radii[idx_ref]:.3f}mm.xlsx"), index=False)

    # график болцмана только если можно
    if uniqE_ref >= 2:
        xx = np.linspace(boltz["E_eV"].min(), boltz["E_eV"].max(), 200)
        yy = intercept + slope*xx
        plt.figure(figsize=(8,6))
        plt.scatter(boltz["E_eV"], boltz["Y"], c="k", marker="s")
        for x, y, nm in zip(boltz["E_eV"], boltz["Y"], boltz["lambda_nm"]):
            plt.annotate(f"{nm:.1f}", (x, y), xytext=(6,6), textcoords="offset points", fontsize=10)
        plt.plot(xx, yy, "-", color="0.6", lw=1.5)
        plt.grid(True, alpha=0.3)
        plt.xlabel("E, eV")
        plt.ylabel("ln(I(r)·λ³ / (gf))")
        plt.title(f"Boltzmann ({side}) at r≈{radii[idx_ref]:.3f} mm: T={T_ref:.0f}±{T_ref_err:.0f} K")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"boltzmann_lambda3_{side}_r{radii[idx_ref]:.3f}mm.png"), dpi=200)
        plt.close()

    # --- профиль T(r) ---
    T_vals, Terr_vals, n_unique_E = [], [], []
    for i in range(len(radii)):
        X, Ylist = [], []
        for col, nm in mapping.items():
            eps = pd.to_numeric(df[col].iloc[i], errors="coerce")
            if not np.isfinite(eps) or eps <= 0:
                continue
            E, gf = LINE_DB[nm]
            X.append(E)
            Ylist.append(np.log(eps * (nm**3) / gf))

        uniqE = len(set(X))
        n_unique_E.append(uniqE)

        if uniqE >= 2:
            # даже если точек много, но X все одинаковые — сюда не попадём
            lr = linregress(X, Ylist)
            b, errb = lr.slope, lr.stderr
            Ti = -1.0/(K_B_EV*b) if b != 0 else np.nan
            dTi = errb/(K_B_EV*b**2) if (b != 0 and np.isfinite(errb)) else np.nan
        else:
            Ti, dTi = np.nan, np.nan

        T_vals.append(Ti)
        Terr_vals.append(dTi)

    prof = pd.DataFrame({
        "Radius_mm": radii,
        "T_K": T_vals,
        "T_err": Terr_vals,
        "n_unique_E": n_unique_E,
    })
    prof.to_excel(os.path.join(out_dir, f"radial_T_lambda3_{side}.xlsx"), index=False)

    return prof, (radii[idx_ref], T_ref, T_ref_err)

# ====== MAIN ======
df = pd.read_excel(in_xlsx)

radius_left  = "Radius_Common_Left"
radius_right = "Radius_Common_Right"

left_cols  = [c for c in df.columns if ("_left_" in str(c)) and ("Radius" not in str(c))]
right_cols = [c for c in df.columns if ("_right_" in str(c)) and ("Radius" not in str(c))]

prof_left,  ref_left  = compute_profile(df, "left",  radius_left,  left_cols)
prof_right, ref_right = compute_profile(df, "right", radius_right, right_cols)

# общий график двух плеч
plt.figure(figsize=(9,7))
plt.errorbar(prof_left["Radius_mm"], prof_left["T_K"], yerr=prof_left["T_err"],
             fmt="o", capsize=3, label="Left")
plt.errorbar(prof_right["Radius_mm"], prof_right["T_K"], yerr=prof_right["T_err"],
             fmt="o", capsize=3, label="Right")
plt.grid(True, which="both", alpha=0.3)
plt.xlabel("r, mm")
plt.ylabel("T(r), K")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "T_profile_left_right.png"), dpi=200)
plt.show()

print(f"LEFT : r≈{ref_left[0]:.3f} mm  T = {ref_left[1]:.0f} ± {ref_left[2]:.0f} K")
print(f"RIGHT: r≈{ref_right[0]:.3f} mm  T = {ref_right[1]:.0f} ± {ref_right[2]:.0f} K")