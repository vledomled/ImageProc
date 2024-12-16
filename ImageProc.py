import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd

# �������� Excel-�����
file_path = '5932 (2).xlsx'  # ���� � ����� Excel
data = pd.read_excel(file_path, header=None)  # ��������� ��� ����������

# ����� ������������ ������
row_number = 867  # ����� ������, ������� ����� �������� (��������, 10-� ������)
wavelengths = data.iloc[0, 1:]  # ������������, ��� ������ ������� � ����� ����
intensities = data.iloc[row_number, 1:].values  # ��������� �������� �� ������ (����� ������ ������)



def smooth(line, window, order):       # ������� �������� ��� �������������
    y_savgol = savgol_filter(line, window_length=window, polyorder=order)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Original Data (Noisy)", alpha=0.5)
    plt.plot(x, y_savgol, label="Savitzky-Golay Smoothing", color='red', linewidth=2)
    plt.xlabel("A")
    plt.ylabel("Intensity")
    plt.title("Savitzky-Golay Smoothing Example")
    plt.legend()
    plt.grid()
    plt.show()
    return y_savgol


smooth(intensities, 51, 3)




