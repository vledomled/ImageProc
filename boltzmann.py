import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Константы (в СИ)
nist = [1.421776496, 0.0197, 1.646589969, 1.97166875, 0.005650537, 0.013]
wave_nm = [465.1119, 510.5537, 515.323, 521.8197, 570.0237, 578.2127]
wave_m = np.array(wave_nm) * 1e-9  # в метры

# Загрузка таблицы
df = pd.read_excel("aligned.xlsx")  # путь к таблице

# Инициализация таблицы для log-значений
log_data = pd.DataFrame()
log_data['Radius'] = df['Radius']

# Расчёт log(I * λ³ / gA)
for i, col in enumerate(df.columns[1:7]):  # первые 6 линий
    I = df[col].values
    λ = wave_m[i]
    gA = nist[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_vals = np.log((I * λ**3) / gA)
        log_vals[~np.isfinite(log_vals)] = np.nan  # NaN вместо -inf
    log_data[col] = log_vals

# График
plt.figure(figsize=(12, 8))
for col in log_data.columns[1:]:
    plt.plot(log_data['Radius'], log_data[col], label=col)
plt.xlabel("Radius (мм)")
plt.ylabel("ln(I·λ³ / gA)")
plt.title("Больцмановская диаграмма по радиусу")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
