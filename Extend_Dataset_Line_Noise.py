# File: Extend_Dataset_Line_Noise.py
# Purpose: Extends the 2D TSP benchmark dataset with "line_noise" instances.
#
# This script imports all core logic (paths, solvers, helpers)
# from the main 2D_Benchmark_Generator.py file and adds a new
# distribution type.

import os
import sys
import numpy as np
import math
import gc
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import from Base Generator ---
import D2_Benchmark_Generator as base_gen

# --- New Distribution Function: Line Noise ---

def generate_line_noise(n_points, grid_size, rng, **kwargs):
    """
    Generates points in a line with perpendicular Gaussian noise.
    The line is guaranteed not to be parallel to the X or Y axis.
    """
    coords_list = []
    unique_set = set()
    
    # 1. Define the line y = mx + b
    
    # Choose a slope 'm' between 0.2 and 5.0 (or -0.2 and -5.0)
    # This avoids slopes of 0 (horizontal) or infinity (vertical).
    m = rng.uniform(0.2, 5.0) * rng.choice([-1, 1])
    
    # Choose a "center" point for the line in the middle 50% of the grid
    # to ensure the line crosses a significant portion of the area.
    cx = rng.uniform(grid_size * 0.25, grid_size * 0.75)
    cy = rng.uniform(grid_size * 0.25, grid_size * 0.75)
    
    # Calculate the y-intercept 'b' based on the center point
    # y = mx + b  =>  b = y - mx
    b = cy - m * cx
    
    # 2. Define noise level
    # We'll use a standard deviation of 2% of the grid size.
    noise_std_dev = grid_size * 0.02
    
    # 3. Define the perpendicular vector
    # The direction vector of the line is (1, m)
    # A perpendicular vector is (-m, 1)
    perp_vec = np.array([-m, 1])
    # Normalize it to create a unit vector for the noise direction
    perp_vec_unit = perp_vec / np.linalg.norm(perp_vec)

    while len(coords_list) < n_points:
        # 4. Get a base point on the line
        #    We sample 'x' uniformly across the grid's width.
        x_base = rng.uniform(0, grid_size)
        y_base = m * x_base + b
        
        # 5. Add perpendicular noise
        #    Get a noise magnitude from a normal distribution
        noise_mag = rng.normal(0, noise_std_dev)
        #    Calculate the offset vector
        offset_vec = perp_vec_unit * noise_mag
        
        x_final = x_base + offset_vec[0]
        y_final = y_base + offset_vec[1]
        
        # 6. Clip coordinates to be within the grid and add the unique point
        x_clip = np.clip(x_final, 0, grid_size)
        y_clip = np.clip(y_final, 0, grid_size)
        
        # Use the _add_unique_point helper imported from the base script
        base_gen._add_unique_point(coords_list, unique_set, x_clip, y_clip)
            
    return np.array(coords_list)

# --- Main Generation and Benchmarking Loop (for Line Noise) ---

def main():
    print("--- 2D TSP Benchmark Extender (Line Noise) ---")
    
    # 1. Patch the imported DIST_MAP with our new function
    base_gen.DIST_MAP['line_noise'] = generate_line_noise
    
    # 2. Define new configurations for 'line_noise'
    #    We will use all 'n_points' from the base config.
    NEW_LINE_CONFIGS = [
        {'n_points': n, 'dist_type': 'line_noise'} 
        for n in base_gen.n_points_list
    ]
    
    print(f"Adding {len(NEW_LINE_CONFIGS)} 'line_noise' configurations.")
    
    # 3. Build the parameter list (logic copied from base_gen.main)
    all_params = []
    print("Generating configurations...")
    # Start sequence from a high number to avoid filename collisions
    seq_j = 100000 
    
    for grid_size in base_gen.GRID_SIZE_LIST:
        for config in NEW_LINE_CONFIGS:
            for i in range(1, base_gen.SAMPLES_PER_CONFIG + 1):
                n = config['n_points']
                dist_type = config['dist_type']
                
                # Create a unique seed
                config_num = sum(ord(c) for c in dist_type)
                base_seed = config_num + seq_j * 1000 + n * 100 + grid_size + i
                
                all_params.append((config, grid_size, i, seq_j, base_seed))
                seq_j += 1
    
    num_workers = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    
    # 4. Instance Generation Pass
    print(f"\nPreparing to generate {len(all_params)} new instances using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use the imported generate_and_save_instance function
        futures = {executor.submit(base_gen.generate_and_save_instance, params): params for params in all_params}
        for future in tqdm(as_completed(futures), total=len(all_params), desc="Instance Generation (Line)"):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"ERROR: Generation failed for {futures[future]}: {e}")

    print("\n--- Line Instance Generation Complete ---")

    # 5. Solution Generation Pass
    print(f"\nPreparing to solve {len(all_params)} new instances using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use the imported solve_single_instance function
        futures = {executor.submit(base_gen.solve_single_instance, params): params for params in all_params}
        for future in tqdm(as_completed(futures), total=len(all_params), desc="Solution Generation (Line)"):
            try:
                future.result()
            except Exception as e:
                tqdm.write(f"ERROR: Solver failed for {futures[future]}: {e}")

    print("\n--- Line Solution Generation Complete ---")
    
    # 6. Visualization Pass
    #    This will (re)generate visualizations for ALL solved instances,
    #    including the new 'line_noise' ones.
    base_gen.visualize_solutions()
    print("\n--- Visualization Complete ---")

if __name__ == "__main__":
    main()