from pathlib import Path
from results import get_results
import numpy as np

model_name = "multi_inout_1_model_final_3_nogcc_nosame"


def get_stats(results, filter = None) -> tuple[float, float, float, float]:
    if filter:
        error_widths = [result.errors_width[filter(result)] for result in results]    
        error_locations = [result.errors_location[filter(result)] for result in results]    
    else:
        error_widths = [result.errors_width for result in results] 
        error_locations = [result.errors_location for result in results]     
    
    error_widths_means = [np.mean(np.abs(err)) for err in error_widths]
    mean_width = np.mean(error_widths_means)
    std_width = np.std(error_widths_means)
    
    error_locations_means = [np.mean(np.abs(err)) for err in error_locations]
    mean_location = np.mean(error_locations_means)
    std_location = np.std(error_locations_means)
    
    return (mean_width, std_width, mean_location, std_location)


results = get_results(model_name)

mean_width, std_width, mean_location, std_location = get_stats(results)
print(f"[all] Width MAE: {mean_width:0.2f} (std: {std_width:0.2f})")
print(f"[all] Location MAE: {mean_location:0.2f} (std: {std_location:0.2f})")
print(f"[all] Location MAE is better than Width MAE by {(mean_width / mean_location - 1)*100:0.2f}%")
print()

mean_width, std_width, mean_location, std_location = get_stats(results, lambda r: r.actual_width < 30)
print(f"[width<30] Width MAE: {mean_width:0.2f} (std: {std_width:0.2f})")
print(f"[wodth<30] Location MAE: {mean_location:0.2f} (std: {std_location:0.2f})")
print()

mean_width, std_width, mean_location, std_location = get_stats(results, lambda r: r.actual_width > 80)
print(f"[width>80] Width MAE: {mean_width:0.2f} (std: {std_width:0.2f})")
print(f"[wodth>80] Location MAE: {mean_location:0.2f} (std: {std_location:0.2f})")
print()
