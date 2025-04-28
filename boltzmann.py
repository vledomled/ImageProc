import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, log



file_path = 'aligned.xlsx'  # ”кажите путь к вашему файлу
data = pd.read_excel(file_path)

radius = data.iloc[:, 0]
intensity_columns = data.iloc[:, 1:]

nist = [1.421776496, 1.646589969, 1.971668758, 0.005650537, 0.013]
wave = [465.1119, 515.323, 521.8197, 570.0237, 578.2127]
energy = [7.737547, 6.191593, 6.192444, 3.816948, 3.78615]


boltz = pd.DataFrame()

for col in range(len(intensity_columns.columns)):
    intensity = intensity_columns.iloc[:, col].values
    calculation = (intensity * ((wave[col] * 1e-9) ** 3)) / nist[col] 
    boltz[f'Line_{col+1}'] = np.round(np.log(calculation), 5)
    
for row in range(len(intensity_columns)):
    
    
print(boltz)