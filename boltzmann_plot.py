# boltzmann_run_lambda3_and_conc.py
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ====== НАСТРОЙКИ ======
in_xlsx  = "abel_results_interpolated_akima.xlsx"  # файл после абелизации/интерполяции
out_dir  = "results"
r_ref_mm = 0.0                                     
os.makedirs(out_dir, exist_ok=True)

n_vals = np.array([
    6.77e20, 6.71e20, 6.77e20, 6.71e20, 6.80e20,
    6.74e20, 6.91e20, 6.88e20, 6.98e20, 6.12e20
], dtype=float)

K_B_EV = 8.617333262e-5  # eV/K
# wavelength_nm : (E_eV, gf)
LINE_DB = {
    510.5537: (3.816948, 0.0197),
    515.3230: (6.191593, 1.64659),
    521.8197: (6.192444, 1.97166876),
    578.2127: (3.78615 , 0.013),
    465.1119: (7.737547, 1.4217765),   # запасные
    570.0237: (3.816948, 0.00565054),
}

def header_to_nm(col):
    s = str(col).strip().replace(',', '.')
    digits = ''.join(ch for ch in s if (ch.isdigit() or ch=='.'))
    v = float(digits) if digits else np.nan
    return v/10.0 if v>1000 else v  # '5105' -> 510.5

def nearest_nm(nm):
    ks = np.array(list(LINE_DB.keys()))
    return float(ks[np.argmin(np.abs(ks - nm))])

# ====== ЗАГРУЗКА ======
df = pd.read_excel(in_xlsx)
radii_mm = pd.to_numeric(df["Radius_mm"], errors="coerce").to_numpy()
cols = [c for c in df.columns if c != "Radius_mm"]
mapping = {c: nearest_nm(header_to_nm(c)) for c in cols}

# ====== ДИАГРАММА БОЛЬЦМАНА (на r_ref_mm) ======
idx_ref = int(np.nanargmin(np.abs(radii_mm - r_ref_mm)))
rows = []
for col, nm in mapping.items():
    eps = float(pd.to_numeric(df[col].iloc[idx_ref], errors="coerce"))
    if not np.isfinite(eps) or eps <= 0:
        continue
    E, gf = LINE_DB[nm]
    lam_m = nm                   
    Y = np.log(eps * (lam_m**3) / gf)   # ln(ε(r) * λ^3 / gf)
    rows.append((nm, E, gf, eps, Y))

boltz = pd.DataFrame(rows, columns=["lambda_nm","E_eV","gf","epsilon","Y"]).sort_values("E_eV")
res = linregress(boltz["E_eV"], boltz["Y"])
slope, intercept, stderr = res.slope, res.intercept, res.stderr
T_ref = -1.0/(K_B_EV * slope) if slope != 0 else np.nan
T_ref_err = stderr/(K_B_EV * slope**2) if slope != 0 else np.nan

# график с подписями точек
xx = np.linspace(boltz["E_eV"].min(), boltz["E_eV"].max(), 200)
yy = intercept + slope*xx
plt.figure(figsize=(8,6))
plt.scatter(boltz["E_eV"], boltz["Y"], c='k', marker='s')
for x, y, nm in zip(boltz["E_eV"], boltz["Y"], boltz["lambda_nm"]):
    plt.annotate(f"{nm:.1f}", (x, y), xytext=(6,6), textcoords="offset points", fontsize=10)
plt.plot(xx, yy, '-', color='0.6', lw=1.5)
plt.grid(True, alpha=0.3)
plt.xlabel("E, eV"); plt.ylabel("ln(ε(r)·λ³ / (gf))")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, f"boltzmann_lambda3_r{radii_mm[idx_ref]:.2f}mm.png"), dpi=200)

boltz.to_excel(os.path.join(out_dir, f"boltzmann_points_r{radii_mm[idx_ref]:.2f}mm.xlsx"), index=False)

# ====== РАДИАЛЬНЫЙ ПРОФИЛЬ ТЕМПЕРАТУРЫ ======
T_vals, Terr_vals = [], []
for i in range(len(radii_mm)):
    X, Ylist = [], []
    for col, nm in mapping.items():
        eps = pd.to_numeric(df[col].iloc[i], errors="coerce")
        if not np.isfinite(eps) or eps <= 0: 
            continue
        E, gf = LINE_DB[nm]
        lam_m = nm
        Ylist.append(np.log(eps * (lam_m**3) / gf))
        X.append(E)
    if len(X) >= 2:
        lr = linregress(X, Ylist)
        b, errb = lr.slope, lr.stderr
        Ti = -1.0/(K_B_EV*b) if b != 0 else np.nan
        dTi = errb/(K_B_EV*b**2) if b != 0 else np.nan
    else:
        Ti, dTi = np.nan, np.nan
    T_vals.append(Ti); Terr_vals.append(dTi)

prof = pd.DataFrame({"Radius_mm": radii_mm, "T_K": T_vals, "T_err": Terr_vals})
prof.to_excel(os.path.join(out_dir, "radial_T_lambda3.xlsx"), index=False)

# ====== ОБЪЕДИНЁННЫЙ ГРАФИК: T(r) + n(r) (лог. ось справа) ======
# подгоним длину n_vals к числу радиусов (если нужно)
n_vals = np.asarray(n_vals, dtype=float)
if len(n_vals) != len(radii_mm):
    # простой вариант: линейно растянуть/обрезать до нужной длины
    idx = np.linspace(0, len(n_vals)-1, len(radii_mm))
    n_vals = np.interp(idx, np.arange(len(n_vals)), n_vals)

fig, axT = plt.subplots(figsize=(9,7))
axT.errorbar(prof["Radius_mm"], prof["T_K"], yerr=prof["T_err"],
             fmt='o', color='k', ecolor='k', elinewidth=1.2, capsize=3)
axT.set_xlabel("r, mm"); axT.set_ylabel("T(r), K")
axT.grid(True, which='both', alpha=0.3)
axT.set_ylim(4000, 6000)

axN = axT.twinx()
axN.plot(radii_mm, n_vals, '-', color='red', lw=2)   # просто линией
axN.set_xlim(0, 3)
axN.set_yscale('log')
axN.set_ylabel(r"n(r), m$^{-3}$", color='red')
axN.tick_params(axis='y', colors='red')
# диапазон как на твоём примере
axN.set_ylim(1e19, 1e22)
axN.set_yticks([1e19, 1e20, 1e21, 1e22])

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "T_and_n_dual_axis.png"), dpi=200)
plt.show()

print(f"r≈{radii_mm[idx_ref]:.3f} mm:  T = {T_ref:.0f} ± {T_ref_err:.0f} K")
