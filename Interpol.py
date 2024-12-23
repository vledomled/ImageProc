
from scipy.interpolate import Akima1DInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# ������� ��� ���������� e(r) � �������������� ������������� ���������
def bockasten_abel_transformation(N, coefficients, r_max):
    n = len(N)
    e = np.zeros(n)
    for j in range(n):
        for k in range(n):
            e[j] += coefficients[k, j] * N[k]
        e[j] /= r_max  # ��������� ������������ ������
    return e

# �������� ������������� ��������� (������ ��� n=10)
bockasten_coefficients = np.array([
    [7.625972, 0.463415, 0, 0, 0, 0, 0, 0, 0, 0],
    [-5.800962, 3.606300, 0.323954, 0, 0, 0, 0, 0, 0, 0],
    [-0.584698, -2.951278, 2.653847, 0.263182, 0, 0, 0, 0, 0, 0],
    [-0.339474, -0.182401, -2.058371, 2.198581, 0.227286, 0, 0, 0, 0, 0],
    [-0.197038, -0.214891, -0.138728, -1.666071, 1.918418, 0.202929, 0, 0, 0, 0],
    [-0.126877, -0.134649, -0.162498, -0.112322, -1.434904, 1.723807, 0.185020, 0, 0, 0],
    [-0.088278, -0.092042, -0.105026, -0.133815, -0.095626, -1.278587, 1.578512, 0.171141, 0, 0],
    [-0.064907, -0.066934, -0.073682, -0.087694, -0.115548, -0.084151, -1.164009, 1.464693, 0.159977, 0],
    [-0.048250, -0.049410, -0.053181, -0.060617, -0.074289, -0.100408, -0.072617, -1.070717, 1.381857, 0.251406],
    [-0.044883, -0.045711, -0.048354, -0.053365, -0.061987, -0.076986, -0.104895, -0.086465, -1.037290, 0.984158]
])

# �������� ������
file_path = '5932.xlsx'  # ������� ���� � ������ �����
data = pd.read_excel(file_path)

# ������������, ��� � �������� �������� �������, � ������ � �������������
radius_columns = data.iloc[:, ::2]
intensity_columns = data.iloc[:, 1::2]

# ���������� ����������
abel_results = pd.DataFrame()
gauss_results = pd.DataFrame()

# ���������� ������ �����
for i, column in enumerate(intensity_columns.columns):
    radii = radius_columns.iloc[:, i].dropna().values  # ������� ������� �����
    intensities = intensity_columns.iloc[:, i].dropna().values  # ������������� ������� �����

    # ���������� ������ �� ��������
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # ����������� ���������� ����� ��� ������������� � �������� ��������
    fine_radii = np.linspace(radii.min(), radii.max(), 1000)  # 1000 ����� ��� ����� ������� �������������

    # ������� ������
    popt, _ = curve_fit(gaussian, radii, intensities, p0=[np.max(intensities), radii[np.argmax(intensities)], 0.5])
    smoothed_intensities = gaussian(fine_radii, *popt)

    # �������� ������������� � �������� ��������
    matched_intensities = gaussian(radii, *popt)
    gauss_results[f'Radius_{i+1}'] = radii
    gauss_results[f'Line_{i+1}'] = matched_intensities
    

    # ���������� �������������� �����
    r_max = radii.max()/1000  # ������������ ������ ��� ������� �����
    abelized_intensities = bockasten_abel_transformation(matched_intensities, bockasten_coefficients, r_max)
    abel_results[f'Radius_{i+1}'] = radii
    abel_results[f'Line_{i+1}'] = abelized_intensities
    

    # ������������
    plt.figure(figsize=(10, 6))
    plt.plot(radii, intensities, 'o', label='Original Data', color='blue')
    plt.plot(fine_radii, smoothed_intensities, label='Gaussian Fit (1000 points)', color='green')
    plt.plot(radii, abelized_intensities, 's', label='Abelized Data (Original Radii)', color='red')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Intensity')
    plt.title(f'Gaussian Fit and Abel Transformation for Line {i+1}')
    plt.legend()
    plt.grid()
    #plt.show()

# ���������� ����������� ����������
abel_results.to_excel('abel_results.xlsx', index=False)
gauss_results.to_excel('gauss_results_abel.xlsx', index=False)



# �������� ������
file_path = 'abel_results.xlsx'  # �������� �� ���� � ������ �����
data = pd.read_excel(file_path)

# ���������� �������� � ��������������
radius_columns = data.iloc[:, ::2]  # ������� � �������� ��������
intensity_columns = data.iloc[:, 1::2]  # ������������� � ������ ��������

# ������ � ������ ������
line_index = 0  # ������ �����
radii_line = radius_columns.iloc[:, line_index].dropna().values  # ������� ������ �����
intensities_line = intensity_columns.iloc[:, line_index].dropna().values  # ������������� ������ �����

# ���������� ������ ��� ������ �����
sorted_indices = np.argsort(radii_line)
radii_line = radii_line[sorted_indices]
intensities_line = intensities_line[sorted_indices]

# ������������ ��� ������ �����
interpolator_line = Akima1DInterpolator(radii_line, intensities_line)
interpolated_intensities_line = interpolator_line(radii_line)

# ���������� �������� ������ ����� ��� ������������� � ������ ������
reference_radii = radii_line

# �������� ������� ��� �������������� ���� ����� � �������� ������ �����
aligned_intensity_data = pd.DataFrame({'Radius': reference_radii})

# ������ �� ������ ����� � ������������ ��������������
for i, column in enumerate(intensity_columns.columns):
    # ������� � ������������� ������� �����
    radii = radius_columns.iloc[:, i].dropna().values
    intensities = intensity_columns.iloc[:, i].dropna().values

    # ���������� ������
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # ������������ ��� �������� ������ �����
    interpolator = Akima1DInterpolator(radii, intensities)
    interpolated_intensities = interpolator(reference_radii)

    # ���������� ������ � �������
    aligned_intensity_data[f'Line_{i+1}'] = interpolated_intensities

# ���������� ������
aligned_intensity_data.to_excel('aligned.xlsx', index=False)

# ������������
plt.figure(figsize=(12, 8))
for column in aligned_intensity_data.columns[1:]:
    plt.plot(aligned_intensity_data['Radius'], aligned_intensity_data[column], label=column)
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Aligned Intensities for All Lines Based on Line 1 Radii')
plt.legend()
plt.grid()

# ������������ ��� ������ ����� (��������� �������� � ����������������� ������)
plt.figure(figsize=(10, 6))
plt.plot(radii_line, intensities_line, 'o-', label='Original Data (Line 1)')
plt.plot(radii_line, interpolated_intensities_line, 'x-', label='Interpolated Data (Line 1)')
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Comparison of Original and Interpolated Data for Line 1')
plt.legend()
plt.grid()
plt.show()


