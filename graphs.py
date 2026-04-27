# Load data, build a clean plot of ε(r) [W/m³] vs radius r [mm] from abel_results.xlsx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the uploaded file
path = "abel_results.xlsx"

# Read Excel
df = pd.read_excel(path)

# Now that we saw the actual structure: pairs are (Radius_n, wavelength_number).
# Let's build plots correctly: each pair of columns is (Radius_X, <wavelength>).
plt.figure(figsize=(12, 8))

pairs_plotted = 0
for col in df.columns:
    if str(col).lower().startswith("radius"):
        idx = df.columns.get_loc(col)
        if idx+1 < len(df.columns):
            r = pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna().to_numpy()
            e = pd.to_numeric(df.iloc[:, idx+1], errors="coerce").dropna().to_numpy()
            n = min(len(r), len(e))
            if n > 0:
                r = r[:n]
                e = e[:n]
                plt.plot(r, e, marker='o', label=f"{df.columns[idx+1]}")
                pairs_plotted += 1

size = {'size' : 14}

plt.xlabel("r, mm", fontdict=size)
plt.ylabel("ε(r), W / m³", fontdict=size)

plt.xlim(0, 4)
plt.grid(True)
if pairs_plotted > 1:
    plt.legend()

plt.show()
    
out_path2 = "epsilon_vs_radius_lines.png"
plt.tight_layout()
plt.savefig(out_path2, dpi=200)
out_path2

