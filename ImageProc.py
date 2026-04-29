import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from lmfit import Model
import matplotlib.pyplot as plt
import subprocess

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.figsize": (8, 8),  
    "figure.dpi": 120,         
    "figure.autolayout": True  
})

def process_raw_file(raw_path: str, output_csv: str):
    cmd = [
        "./raw_processor.exe", 
        raw_path, 
        "temp.ppm", 
        output_csv
    ]
    
    print(f"Running C processor for {raw_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("C processing finished successfully.")
    else:
        print("Error in C code:")
        print(result.stderr)
        raise RuntimeError("C processing failed")

def load_data(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(filepath, header=None)
    except Exception as e:
        logging.error(f"Failed to load file {filepath}: {e}")
        raise

def load_data_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV matrix instead of Excel."""
    try:
        return pd.read_csv(filepath, header=None, sep=',')
    except Exception as e:
        logging.error(f"Failed to load CSV file {filepath}: {e}")
        raise

def cut_line(data: pd.DataFrame, bottom: float, top: float) -> pd.DataFrame:
    return data[(data.iloc[:, 0] >= bottom) & (data.iloc[:, 0] <= top)]

def smooth_data(intensities: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    smoothed = pd.DataFrame()
    for idx in range(len(intensities)):
        row = intensities.iloc[idx].values
        if len(row) < window_length:
            raise ValueError(f"Row {idx} too short for smoothing window.")
        smoothed_row = savgol_filter(row, window_length=window_length, polyorder=polyorder)
        smoothed[f"Line_{idx+1}"] = smoothed_row
    return smoothed

def find_top_lines(smoothed: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    sums = smoothed.sum(axis=0)
    top_indices = sums.nlargest(top_n).index
    return smoothed[top_indices]

def find_cutoff_right(data, start_index: int, threshold: float) -> int:
    """Right boundary (first index where signal drops below threshold)."""
    for i in range(start_index, len(data)):
        if data[i] < threshold:
            return i
    return len(data) - 1

def find_cutoff_left(data, start_index: int, threshold: float) -> int:
    """Left boundary (first index where signal drops below threshold)."""
    for i in range(start_index, -1, -1):
        if data[i] < threshold:
            return i
    return 0

def gaussian(x, y0, A, mu, sigma):
    return y0 + A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def fit_gaussian(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]

    if x_values.size < 4:
        logging.warning(f"Not enough points for fit: {x_values.size}")
        return None

    # Начальные оценки, похожие по логике на Origin
    edge_n = max(1, len(y_values) // 10)

    left_edge = y_values[:edge_n]
    right_edge = y_values[-edge_n:]

    y0_init = np.mean(np.r_[left_edge, right_edge])
    A_init = np.max(y_values) - y0_init
    mu_init = x_values[np.argmax(y_values)]
    sigma_init = (x_values.max() - x_values.min()) / 6
    if sigma_init <= 0:
        sigma_init = 0.1

    gauss_mod = Model(gaussian)

    params = gauss_mod.make_params(
        y0=y0_init,
        A=A_init,
        mu=mu_init,
        sigma=sigma_init
    )

    # Чтобы sigma не ушла в отрицательное значение
    params['sigma'].min = 1e-12

    try:
        result = gauss_mod.fit(y_values, params, x=x_values)

        if result.success:
            if hasattr(result, 'rsquared'):
                r2 = result.rsquared
            else:
                ss_res = np.sum(result.residual ** 2)
                ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'y0': result.params['y0'].value,
                'A': result.params['A'].value,
                'Mu': result.params['mu'].value,
                'Sigma': result.params['sigma'].value,
                'R2': r2,
                'RedChi': result.redchi
            }

        return None

    except Exception as e:
        logging.error(f"Fit error: {e}")
        return None

def main():
    raw_name = input('Enter RAW file name (e.g., _DSC4840.NEF): ').strip()
    
    temp_csv = Path("temp_matrix.csv")
    
    try:
        process_raw_file(raw_name, str(temp_csv))
    except Exception as e:
        logging.error(f"C Processing failed: {e}")
        return

    data = load_data_csv(temp_csv)
    total_pixels = data.shape[1] - 1  
    
    cu_lines = {
        #"Cu I 465.1": {"bottom": 464.5, "top": 465.6},
        "Cu I 510.5 nm": {"bottom": 510.2, "top": 511.6},
        "Cu I 515.3 nm": {"bottom": 514.8, "top": 515.8},
        "Cu I 521.8 nm": {"bottom": 521.3, "top": 522.3},
        "Cu I 570.0 nm": {"bottom": 569.4, "top": 570.8},
        #"Cu I 578.2 nm": {"bottom": 577.0, "top": 579},

    }

    window_length = 75
    polyorder = 3
    num_points_side = 40  
    
    line_profiles = {}
    centers = {}
    
    print("\n--- Spatial Center Analysis ---")
    base_center = None
    
    for line_name, bounds in cu_lines.items():
        line_data = cut_line(data, bounds["bottom"], bounds["top"])
        if line_data.empty:
            logging.warning(f"No data found for {line_name} in given range.")
            continue
            
        intensities = line_data.iloc[:, 1:]
        smoothed = smooth_data(intensities, window_length, polyorder)
        
        spatial_profile = smoothed.sum(axis=1).values
        line_profiles[line_name] = spatial_profile
        
        max_pos = int(np.argmax(spatial_profile))
        centers[line_name] = max_pos
        
        if base_center is None:
            base_center = max_pos
            print(f"{line_name}: Center = {max_pos} px (Reference line)")
        else:
            diff_px = max_pos - base_center
            diff_pct = (diff_px / total_pixels) * 100 
            print(f"{line_name}: Center = {max_pos} px | Shift: {diff_px:+} px ({diff_pct:+.2f}%)")

    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard qualitative colormap
    for idx, (line_name, profile) in enumerate(line_profiles.items()):
        center = centers[line_name]
        c = colors[idx % len(colors)]
        
        norm_profile = profile / np.max(profile) 
        plt.plot(norm_profile, label=f"{line_name} (Center: {center})", color=c, lw=2)
        plt.axvline(center, linestyle='--', color=c, alpha=0.7)

    #plt.title("Spatial Profiles of Cu Lines")
    plt.xlabel("Spatial Coordinate [pixels]")
    plt.ylabel("Relative Intensity [a.u.]")
    plt.legend()
    plt.tight_layout() # Ensures labels are not cut off
    plt.show()

    print("\n--- Gaussian Fitting (Spectra) ---")
    all_fit_results = []
    R2_THRESHOLD = 0.95 

    for line_name, bounds in cu_lines.items():
        line_data = cut_line(data, bounds["bottom"], bounds["top"])
        wavelengths = line_data.iloc[:, 0].values
        intensities = line_data.iloc[:, 1:]
        smoothed = smooth_data(intensities, window_length, polyorder)
        
        center = centers[line_name]
        
        print(f"\nFinding physical edges for {line_name} (Criterion: R^2 > {R2_THRESHOLD})...")
        
        def is_plasma_edge(px):
            if px < 0 or px >= len(smoothed): 
                return False
            y_vals = smoothed.iloc[px].values
            if np.max(y_vals) < 1e-3: 
                return False
            fit = fit_gaussian(wavelengths, y_vals)
            if not fit: 
                return False
            return fit.get('R2', 0) >= R2_THRESHOLD

        right_cutoff_idx = center
        for px in range(center + 1, len(smoothed)):
            if is_plasma_edge(px):
                right_cutoff_idx = px
            else:
                if not is_plasma_edge(px + 1):
                    break  

        left_cutoff_idx = center
        for px in range(center - 1, -1, -1):
            if is_plasma_edge(px):
                left_cutoff_idx = px
            else:
                if not is_plasma_edge(px - 1):
                    break

        print(f"Edges found: left {left_cutoff_idx} px, right {right_cutoff_idx} px (width {right_cutoff_idx - left_cutoff_idx} px)")

        left_indices = np.linspace(left_cutoff_idx, center, num_points_side + 1, dtype=int)
        right_indices = np.linspace(center, right_cutoff_idx, num_points_side + 1, dtype=int)
        selected_indices = np.unique(np.concatenate([left_indices, right_indices]))
        
        if len(selected_indices) < 4:
            logging.warning(f"{line_name}: Profile is too narrow.")
            continue
            
        print(f"Performing final fit for {len(selected_indices)} spectra...")
        
        for s_idx in selected_indices:
            radius_mm = (s_idx - center) * 0.0155
            spectrum_y = smoothed.iloc[s_idx].values
            spectrum_x = wavelengths
            
            if np.max(spectrum_y) < 1e-3:
                continue
                
            fit = fit_gaussian(spectrum_x, spectrum_y)
            
            if fit:
                area = fit['A'] * fit['Sigma'] * np.sqrt(2 * np.pi)
                all_fit_results.append({
                    'Line': line_name,
                    'Pixel': s_idx,
                    'Radius_mm': radius_mm,
                    'Amplitude': fit['A'],
                    'Mu': fit['Mu'],
                    'Sigma': fit['Sigma'],
                    'Area': area,
                    'R2': fit.get('R2', 0),
                    'RedChi': fit.get('RedChi', 0)
                })
                
                # Plotting only for center and outer edge
                if s_idx == center or s_idx == selected_indices[0]:
                    x_dense = np.linspace(np.min(spectrum_x), np.max(spectrum_x), 500)
                    plt.figure()
                    plt.scatter(spectrum_x, spectrum_y, label='Experimental data', color='black', s=15, zorder=3)
                    plt.plot(x_dense, gaussian(x_dense, fit['y0'], fit['A'], fit['Mu'], fit['Sigma']), 
                             label=f'Gaussian fit ($R^2={fit.get("R2", 0):.3f}$)', color='red', linewidth=2, zorder=2)
                    
                    if s_idx == center:
                        pos_label = "Center"
                    else:
                        pos_label = f"Edge ($R = {radius_mm:.3f}$ mm)"
                        
                    #plt.title(f"{line_name} | {pos_label}")
                    plt.xlabel("Wavelength [nm]")
                    plt.ylabel("Intensity [W/m$^2$/nm]")
                    plt.legend(frameon=False) 
                    plt.tight_layout()
                    plt.show()
            else:
                logging.warning(f"Fit failed for {line_name} at pixel {s_idx}")

    if all_fit_results:
        (Path.cwd() / 'results').mkdir(exist_ok=True)
        fit_df = pd.DataFrame(all_fit_results)
        output_file = Path('results') / 'All_spectra_areas.xlsx'
        fit_df.to_excel(output_file, index=False)
        print(f"\nDone! Integral areas and quality metrics saved to {output_file}")


if __name__ == "__main__":
    main()