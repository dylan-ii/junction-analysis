import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import ListedColormap
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion

NUM_EROSIONS = 1  # keep to 1

# ellipsoid element params
RADIUS_X = 3 
RADIUS_Y = 3
RADIUS_Z = 0.85

# functionality for ellipsoid structuring element
def create_ellipsoid_strel(radius_x, radius_y, radius_z):
    x, y, z = np.ogrid[-radius_x:radius_x+1, -radius_y:radius_y+1, -radius_z:radius_z+1]
    strel = (x**2 / radius_x**2 + y**2 / radius_y**2 + z**2 / radius_z**2) <= 1
    return strel.astype(int)

STRUCTURING_ELEMENT = create_ellipsoid_strel(RADIUS_X, RADIUS_Y, RADIUS_Z)

# function for creating random colors for maximum spatial distinction
def generate_random_colors(n_colors, seed=42):
    """Generate random colors that are easily distinguishable for nearby objects."""
    np.random.seed(seed)  # For reproducibility across frames
    
    # Generate random RGB values with good saturation and brightness
    colors = []
    for i in range(n_colors):
        # Use label-based seed to ensure same label gets same color across frames
        np.random.seed(seed + i)
        
        # Generate bright, saturated colors by avoiding low values
        r = np.random.uniform(0.2, 1.0)
        g = np.random.uniform(0.2, 1.0) 
        b = np.random.uniform(0.2, 1.0)
        
        # Ensure at least one channel is high for brightness
        max_val = max(r, g, b)
        if max_val < 0.7:
            # Boost the maximum channel
            if r == max_val:
                r = np.random.uniform(0.7, 1.0)
            elif g == max_val:
                g = np.random.uniform(0.7, 1.0)
            else:
                b = np.random.uniform(0.7, 1.0)
        
        colors.append([r, g, b, 1.0])
    
    return np.array(colors)

# function for loading and removing small objects from volume
def load_and_preprocess_volume(file_path, min_size):
    volume = imread(file_path)
    
    for _ in range(NUM_EROSIONS):
        eroded_volume = binary_erosion(volume, structure=STRUCTURING_ELEMENT)
        volume[volume != eroded_volume] = 0
    
    cleaned_volume = remove_small_objects(volume, min_size=min_size)
    return cleaned_volume

# main function for segmentation
def segment_fibers(volume, min_size=0):
    labeled_volume = label(volume, connectivity=3)
    if min_size > 0:
        labeled_volume = remove_small_objects(labeled_volume, min_size=min_size)
    return labeled_volume

# function for calculations of fiber properties
def calculate_fiber_properties(labeled_volume):
    props = regionprops(labeled_volume)
    fiber_data = []
    for prop in props:
        voxel_count = prop.area
        coords = prop.coords
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        # Convert to (x, y, z) order for plotting
        endpoint1 = (min_coords[2], min_coords[1], min_coords[0])
        endpoint2 = (max_coords[2], max_coords[1], max_coords[0])
        
        end_to_end_length = np.linalg.norm(np.array(endpoint1) - np.array(endpoint2))
        fiber_data.append({
            'label': prop.label,
            'voxel_count': voxel_count,
            'end_to_end_length': end_to_end_length,
            'voxel_coords': coords
        })
    return fiber_data

# function for determining gel ratio
def identify_gel_point(fiber_data):
    voxel_counts = sorted([fiber['voxel_count'] for fiber in fiber_data], reverse=True)
    if len(voxel_counts) < 2:
        return False, 1
    gel_ratio = voxel_counts[1] / voxel_counts[0]
    return False, gel_ratio

# main function for processing and visualization
def analyze_time_series(folder_path, save_path, min_size):
    file_list = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path)
         if f.lower().endswith(('.tif', '.tiff'))]
    )
    if not file_list:
        print("No TIFF files found in directory.")
        return

    save_dir    = os.path.join(save_path, 'labeled_volumes')
    os.makedirs(save_dir, exist_ok=True)

    metrics = {
        'num_fibers': [],
        'avg_voxels': [],
        'avg_lengths': [],
        'gel_ratios': [],
        'labeled_voxels': []
    }

    # --- Determine volume dims & center for zoom ---
    sample_volume = imread(file_list[0])
    depth, height, width = sample_volume.shape
    center_x, center_y, center_z = width / 2, height / 2, depth / 2
    zoom_radius = 150

    # --- Set up full-volume figure & axis ---
    fig_full = plt.figure(figsize=(12, 10))
    ax_full = fig_full.add_subplot(111, projection='3d')
    ax_full.set_xlabel('X'); ax_full.set_ylabel('Y'); ax_full.set_zlabel('Z')
    ax_full.set_xlim(0, width); ax_full.set_ylim(0, height); ax_full.set_zlim(0, depth)

    # --- Set up zoomed figure & axis ---
    fig_zoom = plt.figure(figsize=(8, 8))
    ax_zoom = fig_zoom.add_subplot(111, projection='3d')
    ax_zoom.set_xlabel('X'); ax_zoom.set_ylabel('Y'); ax_zoom.set_zlabel('Z')
    zoom_min_x = center_x - zoom_radius
    zoom_max_x = center_x + zoom_radius
    zoom_min_y = center_y - zoom_radius
    zoom_max_y = center_y + zoom_radius
    zoom_min_z = center_z - zoom_radius
    zoom_max_z = center_z + zoom_radius

    # --- Prepare video writers ---
    writer_full = FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    writer_zoom = FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    full_video_path = os.path.join(save_path, 'labeled_fibers.mp4')
    zoom_video_path = os.path.join(save_path, 'labeled_fibers_zoom.mp4')

    # --- Render both videos in lockstep ---
    with writer_full.saving(fig_full, full_video_path, dpi=100), \
         writer_zoom.saving(fig_zoom, zoom_video_path, dpi=100):
        for i, file_path in enumerate(file_list):
            print(f"Processing timepoint {i+1}/{len(file_list)}...")
            volume = load_and_preprocess_volume(file_path, min_size)
            labeled_volume = segment_fibers(volume, min_size)
            imsave(
                os.path.join(save_dir, f'labeled_volume_{i:04d}.tif'),
                labeled_volume.astype(np.uint16)
            )

            fiber_data = calculate_fiber_properties(labeled_volume)
            # existing metrics
            metrics['num_fibers'].append(len(fiber_data))
            metrics['avg_voxels'].append(
                np.mean([f['voxel_count'] for f in fiber_data]) if fiber_data else 0
            )
            metrics['avg_lengths'].append(
                np.mean([f['end_to_end_length'] for f in fiber_data]) if fiber_data else 0
            )
            _, gel_ratio = identify_gel_point(fiber_data)
            metrics['gel_ratios'].append(gel_ratio)
            # new metric: total labeled voxels
            metrics['labeled_voxels'].append(
                sum(f['voxel_count'] for f in fiber_data)
            )

            colors = generate_random_colors(len(fiber_data))

            # full-volume plot
            ax_full.clear()
            ax_full.set_title(f'Full Volume — Time Step: {i}')
            ax_full.set_xlim(0, width); ax_full.set_ylim(0, height); ax_full.set_zlim(0, depth)
            for idx, fiber in enumerate(fiber_data):
                coords = fiber['voxel_coords']
                x, y, z = coords[:,2], coords[:,1], coords[:,0]
                if len(x) > 100_000_000:
                    sel = np.random.choice(len(x), 100_000_000, replace=False)
                    x, y, z = x[sel], y[sel], z[sel]
                ax_full.scatter(x, y, z,
                                c=[colors[idx]], s=3, alpha=0.8,
                                depthshade=False, edgecolors='none')
            writer_full.grab_frame()

            # zoomed-in plot
            ax_zoom.clear()
            ax_zoom.set_title(f'Zoomed (±{zoom_radius} voxels of center) — Time Step: {i}')
            ax_zoom.set_xlim(zoom_min_x, zoom_max_x)
            ax_zoom.set_ylim(zoom_min_y, zoom_max_y)
            ax_zoom.set_zlim(zoom_min_z, zoom_max_z)
            for idx, fiber in enumerate(fiber_data):
                coords = fiber['voxel_coords']
                x, y, z = coords[:,2], coords[:,1], coords[:,0]
                d2 = (x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2
                mask = d2 <= zoom_radius**2
                if not np.any(mask):
                    continue
                xz, yz, zz = x[mask], y[mask], z[mask]
                if len(xz) > 100_000_000:
                    sel = np.random.choice(len(xz), 100_000_000, replace=False)
                    xz, yz, zz = xz[sel], yz[sel], zz[sel]
                ax_zoom.scatter(xz, yz, zz,
                                c=[colors[idx]], s=3, alpha=0.8,
                                depthshade=False, edgecolors='none')
            writer_zoom.grab_frame()

            if gel_ratio <= 0.0001:
                print(f"Gelation detected at timepoint {i+1}! Stopping early.")
                break

    plt.close(fig_full)
    plt.close(fig_zoom)
    print(f"Full video saved to: {full_video_path}")
    print(f"Zoomed-in video saved to: {zoom_video_path}")

    # --- Metric plots (existing) ---
    timepoints = np.arange(1, len(metrics['num_fibers']) + 1)
    plt.figure()
    plt.plot(timepoints, metrics['num_fibers'], marker='o')
    plt.xlabel("Timepoint"); plt.ylabel("Number of Labels")
    plt.savefig(os.path.join(save_path, 'num_labels_over_time.png'))

    plt.figure()
    plt.plot(timepoints, metrics['avg_lengths'], marker='o')
    plt.xlabel("Timepoint"); plt.ylabel("Average End-to-End Length")
    plt.savefig(os.path.join(save_path, 'avg_lengths_over_time.png'))

    fig2, ax1 = plt.subplots()
    ax1.plot(timepoints, metrics['avg_voxels'], marker='o', label="Avg Voxel Count")
    ax1.set_xlabel("Timepoint"); ax1.set_ylabel("Avg Voxel Count")
    ax2 = ax1.twinx()
    ax2.plot(timepoints, metrics['gel_ratios'], marker='s', linestyle='--', label="Gel Ratio")
    ax2.set_ylabel("Gel Ratio")
    fig2.tight_layout()
    plt.savefig(os.path.join(save_path, 'voxel_count_gel_ratio.png'))

    # --- New metric plot: total labeled voxels over time ---
    plt.figure()
    plt.plot(timepoints, metrics['labeled_voxels'], marker='o')
    plt.xlabel("Timepoint"); plt.ylabel("Total Labeled Voxels")
    plt.savefig(os.path.join(save_path, 'labeled_voxels_over_time.png'))

    print("Analysis complete. Plots saved.")

# main script call
analyze_time_series('ex4Classified', "exp4LabeledNew", min_size=125)
