
from scipy.interpolate import Akima1DInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# �������� ������
file_path = '5932.xlsx'  # �������� �� ���� � ������ �����
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
aligned_intensity_data.to_excel('aligned_intensities_to_line_1.xlsx', index=False)

# ������������
plt.figure(figsize=(12, 8))
for column in aligned_intensity_data.columns[1:]:
    plt.plot(aligned_intensity_data['Radius'], aligned_intensity_data[column], label=column)
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Aligned Intensities for All Lines Based on Line 1 Radii')
plt.legend()
plt.grid()
plt.show()

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