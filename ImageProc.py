import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
from scipy.optimize import curve_fit

# �������� Excel-�����
file_path = '514.xlsx'  # ���� � ����� Excel
data = pd.read_excel(file_path, header=None)  # ��������� ��� ����������
num = data.shape

x = np.linspace(0, num[1]-1, num[1]-1)


wavelengths = data.iloc[:, 0]  #������ ������� � ����� ����
intensities = data.iloc[:, 1:]  # ��������� �������� �� �����

smoothed_data = pd.DataFrame(index=data.index, columns=data.columns)
smoothed_data.iloc[:, 0] = wavelengths

def smooth(line, window, order): 
    y_savgol = savgol_filter(line, window_length=window, polyorder=order)
    return y_savgol

for i in range(len(intensities)):
    row_data = intensities.iloc[i, :].values  # �������� ������� ������
    smoothed_row = savgol_filter(row_data, 51, 3)
    smoothed_data.iloc[i, 1:] = smoothed_row
    plt.plot(x, smoothed_row)

plt.xlabel("Wavelengths")
plt.ylabel("Intensity")
plt.title("Smoothed Spectral Lines")
plt.legend()
plt.grid()
plt.show()


output_file = 'smoothed_data.xlsx'
smoothed_data.to_excel(output_file, index=False)    

centers = []
pixel_positions = []

for i in range(len(intensities)):
    smoothed_row = smoothed_data.iloc[i, 1:].values  # ��������� ������ ������ (��� ������ numpy)
    center = np.argmax(smoothed_row)  # ������ ��������� �������������
    centers.append(center)  # ��������� �����
    
    # ���������� ���������� ������� � ����� 0.0155
    num_pixels = len(smoothed_row)
    positions = [(j - center) * 0.0155 for j in range(num_pixels)]
    pixel_positions.append(positions)

# ����� �����������
print("Centers", centers)

    
plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, line, label="Smoothed Line", color='red')
plt.axvline(0, color='blue', linestyle='--', label="Spatial Center")  # ����� ������
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Spectral Line with Spatial Center")
plt.legend()
plt.grid()
plt.show()

# ����� �����������
print(f"Center: {center}")


# ������� ��� ������ ���� ����� ������
def find_line_end(line, start_index, threshold):
    for i in range(start_index, len(intensities)):
        if line[i] < threshold:
            return i
    return len(line) - 1  # ���������� ��������� �������, ���� ����� �� ������

# ����� "����" ��� �������������
threshold = 3.5  # ������� ���� �������� ������

# ����� ���� ����� ������
line_end_index = find_line_end(line, center, threshold=threshold)

# ��������� 10 ��������������� ����� ����� ������� � �����
num_points = 10
right_branch_indices = np.linspace(center, line_end_index, num_points, dtype=int)
right_branch_positions = [pixel_positions[i] for i in right_branch_indices]
right_branch_values = [line[i] for i in right_branch_indices]

# ������������ ����������
plt.figure(figsize=(10, 6))
plt.plot(pixel_positions, line, label="Smoothed Line", color='red')
plt.axvline(0, color='blue', linestyle='--', label="Spatial Center")  # ����� ������
plt.scatter(right_branch_positions, right_branch_values, color='green', zorder=5, label="Right Branch Points")
plt.xlabel("Pixel Position (Step = 0.0155)")
plt.ylabel("Intensity")
plt.title("Right Branch of Spectral Line with Threshold")
plt.legend()
plt.grid()
plt.show()

# ����� �����������
print("Coordinates and values:")
for pos, val in zip(right_branch_positions, right_branch_values):
    print(f"Pos: {pos:.4f}, Value: {val:.4f}")




# plt.figure(figsize=(10, 6))
#     plt.plot(x, intensities, label="Original Data (Noisy)", alpha=0.5)
#     plt.plot(x, y_savgol, label="Savitzky-Golay Smoothing", color='red', linewidth=2)
#     plt.xlabel("Pixels")
#     plt.ylabel("Intensity")
#     plt.title("Savitzky-Golay Smoothing Example")
#     plt.legend()
#     plt.grid()
#     plt.show()




# # �������� �������
# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# # ���������� ������ �����
# x_data = np.array(right_branch_positions)  # ������� �����
# y_data = np.array(right_branch_values)     # �������� �������������

# # ��������� ����������� ��� ���������� A, mu, sigma
# initial_guess = [max(y_data), x_data[np.argmax(y_data)], 0.1]

# # �������� ������ �������� ������
# params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

# # ���������� ���������
# A_fit, mu_fit, sigma_fit = params
# print(f"Gauss:\A: {A_fit:.4f}\nCenter: {mu_fit:.4f}\nSigma: {sigma_fit:.4f}")

# # ���������� �������� ����� � �������������
# x_fit = np.linspace(min(x_data), max(x_data), 100)
# y_fit = gaussian(x_fit, *params)

# plt.figure(figsize=(10, 6))
# plt.scatter(x_data, y_data, color='red', label="Data Points")  # �������� �����
# plt.plot(x_fit, y_fit, color='blue', label="Gaussian Fit")     # �������� �������������
# plt.xlabel("Pixel Position")
# plt.ylabel("Intensity")
# plt.title("Gaussian Fit of Right Branch")
# plt.legend()
# plt.grid()
# plt.show()

