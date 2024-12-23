
from scipy.interpolate import Akima1DInterpolator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model


file_path = '5932.xlsx'  # �������� �� ���� � ������ �����
data = pd.read_excel(file_path)

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

gauss_model = Model(gaussian)

intensity_columns = data.iloc[:, 1::2]  # ׸���� ������� - �������������
radius_columns = data.iloc[:, ::2]  # �������� ������� - �������

fit_results = []  # ������ ��� �������� ���������� �������������

# ������������� ������ �����
for i, column in enumerate(intensity_columns.columns):
    radii = radius_columns.iloc[:, i].dropna().values  # ������� ������� �����
    intensities = intensity_columns.iloc[:, i].dropna().values  # ������������� ������� �����

    # ���������� ������ �� ��������
    sorted_indices = np.argsort(radii)
    radii = radii[sorted_indices]
    intensities = intensities[sorted_indices]

    # ����������� ���������� ����� ��� ������������� � �������� ��������
    fine_radii = np.linspace(radii.min(), radii.max(), 1000)  # 1000 ����� ��� ����� ������� �������������

    # ��������� �����������
    initial_params = gauss_model.make_params(A=np.max(intensities), mu=radii[np.argmax(intensities)], sigma=0.5)

    # �������
    result = gauss_model.fit(intensities, x=radii, params=initial_params)

    # ���������� ����������
    fit_results.append({
        'Line': column,
        'Amplitude (A)': result.params['A'].value,
        'Mean (mu)': result.params['mu'].value,
        'Sigma': result.params['sigma'].value,
        'Fit Success': result.success
    })

    # ������������ �������������
    plt.figure(figsize=(10, 6))
    plt.scatter(radii, intensities, label='Original Data', color='blue')
    plt.plot(fine_radii, gaussian(fine_radii, result.params['A'].value, result.params['mu'].value, result.params['sigma'].value),
             label='Gaussian Fit (Smoothed)', color='red')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Intensity')
    plt.title(f'Gaussian Fit for Line {i+1}')
    plt.legend()
    plt.grid()
    plt.show()

# ���������� ���������� �������������
fit_results_df = pd.DataFrame(fit_results)
fit_results_df.to_excel('gaussian_fit_results.xlsx', index=False)


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
#plt.show()

# ������������ ��� ������ ����� (��������� �������� � ����������������� ������)
plt.figure(figsize=(10, 6))
plt.plot(radii_line, intensities_line, 'o-', label='Original Data (Line 1)')
plt.plot(radii_line, interpolated_intensities_line, 'x-', label='Interpolated Data (Line 1)')
plt.xlabel('Radius (mm)')
plt.ylabel('Intensity')
plt.title('Comparison of Original and Interpolated Data for Line 1')
plt.legend()
plt.grid()
#plt.show()


