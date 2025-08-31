import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

# Paths
path_in = "abel_results.xlsx"
path_out = "abel_results_interpolated_akima.xlsx"
plot_path = "epsilon_vs_radius_interpolated_akima.png"

# Load
df = pd.read_excel(path_in)

# Collect (r, v, name) pairs
pairs = []
cols = list(df.columns)
i = 0
while i < len(cols):
    col = cols[i]
    if str(col).lower().startswith("radius") and i + 1 < len(cols):
        r = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna().to_numpy()
        v = pd.to_numeric(df.iloc[:, i+1], errors="coerce").dropna().to_numpy()
        n = min(len(r), len(v))
        r = r[:n]
        v = v[:n]
        name = str(cols[i+1])
        if n > 1:
            pairs.append((r, v, name))
        i += 2
    else:
        i += 1

# Common interval [max(min r), min(max r)]
common_rmin = max(r.min() for r,_,_ in pairs)
common_rmax = min(r.max() for r,_,_ in pairs)

# Number of points: take min number of samples among pairs
n_points = min(len(r) for r,_,_ in pairs)

# Common grid
common_r = np.linspace(common_rmin, common_rmax, n_points)

# Interpolate using Akima
aligned = pd.DataFrame({"Radius_mm": common_r})
for r, v, name in pairs:
    order = np.argsort(r)
    interp = Akima1DInterpolator(r[order], v[order])
    aligned[name] = interp(common_r)

# Save table
aligned.to_excel(path_out, index=False)

# Plot
plt.figure(figsize=(9, 6))
for name in aligned.columns[1:]:
    plt.plot(aligned["Radius_mm"], aligned[name], marker='o', label=name)
plt.xlabel("r, мм")
plt.ylabel("ε(r), W/m³")
plt.title("Интерполированные ε(r) (Akima) на общей сетке радиусов")
plt.grid(True)
if len(aligned.columns) > 2:
    plt.legend(title="Линия")
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.show()

path_out, plot_path
