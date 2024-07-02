from pathlib import Path
from results import get_results
import numpy as np
from dotmap import DotMap
import pandas as pd

model_name = "final_width_location2"


def get_stats(results, filter = None) -> tuple[float, float, float, float]:
    if filter:
        error_widths = [result.errors_width[filter(result)] for result in results]    
        error_locations = [result.errors_location[filter(result)] for result in results]    
    else:
        error_widths = [result.errors_width for result in results] 
        error_locations = [result.errors_location for result in results]     
    
    result = DotMap()

    error_widths_means = [np.mean(np.abs(err)) for err in error_widths]
    result.min_err_width = np.min(error_widths_means)
    result.max_err_width = np.max(error_widths_means)
    result.mean_width = np.mean(error_widths_means)
    result.std_width = np.std(error_widths_means)
    
    error_locations_means = [np.mean(np.abs(err)) for err in error_locations]
    result.min_err_location = np.min(error_locations_means)
    result.max_err_location = np.max(error_locations_means)
    result.mean_location = np.mean(error_locations_means)
    result.std_location = np.std(error_locations_means)
    
    return result


results = get_results(model_name)

s = get_stats(results)
print(f"[all] Width MAE: {s.mean_width:0.2f} (std: {s.std_width:0.2f}, min: {s.min_err_width:0.2f}, max: {s.max_err_width:0.2f})")
print(f"[all] Location MAE: {s.mean_location:0.2f} (std: {s.std_location:0.2f}, min: {s.min_err_location:0.2f}, max: {s.max_err_location:0.2f})")
print(f"[all] Location MAE is better than Width MAE by {(s.mean_width / s.mean_location - 1)*100:0.2f}%")
print()

s = get_stats(results, lambda r: r.actual_width < 30)
print(f"[width<30] Width MAE: {s.mean_width:0.2f} (std: {s.std_width:0.2f})")
print(f"[wodth<30] Location MAE: {s.mean_location:0.2f} (std: {s.std_location:0.2f})")
print()

s = get_stats(results, lambda r: r.actual_width > 80)
print(f"[width>80] Width MAE: {s.mean_width:0.2f} (std: {s.std_width:0.2f})")
print(f"[wodth>80] Location MAE: {s.mean_location:0.2f} (std: {s.std_location:0.2f})")
print()

best_epochs = []
for rep in range(10):
    df = pd.read_csv(f'data/train/out_models/final_width_location2/{rep}/properties.csv', index_col=0)
    result_best_epoch = df[df.property == 'result_best_epoch'].value[1]
    best_epochs.append(int(result_best_epoch))

print(f'Best epoch: min: {np.min(best_epochs)}, max: {np.max(best_epochs)}, median: {np.median(best_epochs)}')