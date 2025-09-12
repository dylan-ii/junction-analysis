import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion

NUM_EROSIONS = 1  # keep to 1

# ellipsoid element params
RADIUS_X = 3 
RADIUS_Y = 3
RADIUS_Z = 0.85

BORDER_SENSITIVITY = 0

def create_ellipsoid_strel(radius_x, radius_y, radius_z):
    x, y, z = np.ogrid[
        -radius_x:radius_x+1,
        -radius_y:radius_y+1,
        -radius_z:radius_z+1
    ]
    strel = (x**2 / radius_x**2 +
             y**2 / radius_y**2 +
             z**2 / radius_z**2) <= 1
    return strel.astype(int)

STRUCTURING_ELEMENT = create_ellipsoid_strel(RADIUS_X, RADIUS_Y, RADIUS_Z)

def generate_random_colors(n_colors, seed=42):
    np.random.seed(seed)
    colors = []
    for i in range(n_colors):
        np.random.seed(seed + i)
        r = np.random.uniform(0.2, 1.0)
        g = np.random.uniform(0.2, 1.0)
        b = np.random.uniform(0.2, 1.0)
        # ensure brightness
        if max(r, g, b) < 0.7:
            idx = np.argmax([r, g, b])
            vals = [r, g, b]
            vals[idx] = np.random.uniform(0.7, 1.0)
            r, g, b = vals
        colors.append([r, g, b, 1.0])
    return np.array(colors)

def load_and_preprocess_volume(file_path, min_size):
    volume = imread(file_path)
    for _ in range(NUM_EROSIONS):
        eroded = binary_erosion(volume, structure=STRUCTURING_ELEMENT)
        volume[volume != eroded] = 0
    return remove_small_objects(volume, min_size=min_size)

def segment_fibers(volume, min_size=0):
    lbl = label(volume, connectivity=3)
    if min_size > 0:
        lbl = remove_small_objects(lbl, min_size=min_size)
    return lbl

def calculate_fiber_properties(labeled_volume):
    props = regionprops(labeled_volume)
    data = []
    for prop in props:
        coords = prop.coords  # (z, y, x)
        voxel_count = prop.area
        minc = coords.min(axis=0)
        maxc = coords.max(axis=0)
        e1 = (minc[2], minc[1], minc[0])
        e2 = (maxc[2], maxc[1], maxc[0])
        length = np.linalg.norm(np.array(e1) - np.array(e2))
        data.append({
            'label': prop.label,
            'voxel_count': voxel_count,
            'end_to_end_length': length,
            'voxel_coords': coords
        })
    return data

def identify_gel_point(fiber_data):
    """
    legacy ratio metric: returns (detected, ratio) where ratio =
    second-largest voxel count / largest voxel count.
    If fewer than 2 fibers, returns (False, 1).
    """
    counts = sorted([f['voxel_count'] for f in fiber_data], reverse=True)
    if len(counts) < 2:
        return False, 1.0
    return False, counts[1] / counts[0]

def is_spanning_volume(coords, volume_shape, tol=BORDER_SENSITIVITY):
    """
    Return True if the set of (z,y,x) coords touches BOTH:
      – the top & bottom faces (y=0 and y=height-1), OR
      – the left & right faces (x=0 and x=width-1),
    within a tolerance of `tol` voxels.
    """
    depth, height, width = volume_shape
    xs = coords[:, 2]
    ys = coords[:, 1]
    touch_left   = xs.min() <= tol
    touch_right  = xs.max() >= (width - 1 - tol)
    touch_top    = ys.min() <= tol
    touch_bottom = ys.max() >= (height - 1 - tol)
    return (touch_top and touch_bottom) or (touch_left and touch_right)

def analyze_time_series(folder_path, save_path, min_size, border_tol=BORDER_SENSITIVITY):
    file_list = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    if not file_list:
        print("No TIFF files found.")
        return

    os.makedirs(os.path.join(save_path, 'labeled_volumes'), exist_ok=True)

    metrics = {
        'num_fibers': [],
        'avg_voxels': [],
        'avg_lengths': [],
        'gel_ratios': [],
        'labeled_voxels': [],
        'gel_metric': []
    }

    gel_detected = False
    gel_frame = None

    sample = imread(file_list[0])
    depth, height, width = sample.shape
    center_x, center_y, center_z = width/2, height/2, depth/2
    zoom_radius = 150

    fig_full = plt.figure(figsize=(12,10))
    ax_full = fig_full.add_subplot(111, projection='3d')
    ax_full.set(xlabel='X', ylabel='Y', zlabel='Z',
                xlim=(0,width), ylim=(0,height), zlim=(0,depth))

    fig_zoom = plt.figure(figsize=(8,8))
    ax_zoom = fig_zoom.add_subplot(111, projection='3d')
    ax_zoom.set(xlabel='X', ylabel='Y', zlabel='Z')
    zx0, zx1 = center_x-zoom_radius, center_x+zoom_radius
    zy0, zy1 = center_y-zoom_radius, center_y+zoom_radius
    zz0, zz1 = center_z-zoom_radius, center_z+zoom_radius

    writer_full = FFMpegWriter(fps=5, codec="libx264",
                               extra_args=["-pix_fmt","yuv420p"])
    writer_zoom = FFMpegWriter(fps=5, codec="libx264",
                               extra_args=["-pix_fmt","yuv420p"])
    full_vid = os.path.join(save_path, 'labeled_fibers.mp4')
    zoom_vid = os.path.join(save_path, 'labeled_fibers_zoom.mp4')

    with writer_full.saving(fig_full, full_vid, dpi=100), \
         writer_zoom.saving(fig_zoom, zoom_vid, dpi=100):

        for i, path in enumerate(file_list):
            print(f"Frame {i+1}/{len(file_list)}")
            vol = load_and_preprocess_volume(path, min_size)
            lbl = segment_fibers(vol, min_size)
            imsave(os.path.join(save_path, 'labeled_volumes',
                                 f'labeled_volume_{i:04d}.tif'),
                   lbl.astype(np.uint16))

            fiber_data = calculate_fiber_properties(lbl)
            metrics['num_fibers'].append(len(fiber_data))
            metrics['avg_voxels'].append(
                np.mean([f['voxel_count'] for f in fiber_data]) if fiber_data else 0
            )
            metrics['avg_lengths'].append(
                np.mean([f['end_to_end_length'] for f in fiber_data]) if fiber_data else 0
            )
            _, ratio = identify_gel_point(fiber_data) if 'identify_gel_point' in globals() else (False, 0)
            metrics['gel_ratios'].append(ratio)

            # sum of voxels in all labels
            total_labeled = sum(f['voxel_count'] for f in fiber_data)
            metrics['labeled_voxels'].append(total_labeled)

            if fiber_data:
                largest = max(fiber_data, key=lambda f: f['voxel_count'])
                if not gel_detected and is_spanning_volume(
                        largest['voxel_coords'], vol.shape, tol=border_tol):
                    gel_detected = True
                    gel_frame = i + 1
                    print(f"Gelation spans faces at frame {gel_frame}")
                # use the sum of labeled voxels, not full volume size
                metrics['gel_metric'].append(
                    (largest['voxel_count'] / total_labeled) if gel_detected else 0
                )
            else:
                metrics['gel_metric'].append(0)

            colors = generate_random_colors(len(fiber_data))

            ax_full.clear()
            ax_full.set(title=f'Full Volume — Frame {i+1}',
                        xlim=(0,width), ylim=(0,height), zlim=(0,depth))
            for idx, f in enumerate(fiber_data):
                z,y,x = f['voxel_coords'].T
                if len(x)>1e8:
                    sel = np.random.choice(len(x), int(1e8), replace=False)
                    x,y,z = x[sel], y[sel], z[sel]
                ax_full.scatter(x, y, z, c=[colors[idx]],
                                s=3, alpha=0.8,
                                depthshade=False, edgecolors='none')
            writer_full.grab_frame()

            ax_zoom.clear()
            ax_zoom.set(title=f'Zoom (±{zoom_radius}) — Frame {i+1}',
                        xlim=(zx0,zx1), ylim=(zy0,zy1), zlim=(zz0,zz1))
            for idx, f in enumerate(fiber_data):
                z,y,x = f['voxel_coords'].T
                d2 = (x-center_x)**2 + (y-center_y)**2 + (z-center_z)**2
                mask = d2 <= zoom_radius**2
                if not mask.any(): continue
                x2,y2,z2 = x[mask], y[mask], z[mask]
                if len(x2)>1e8:
                    sel = np.random.choice(len(x2), int(1e8), replace=False)
                    x2,y2,z2 = x2[sel], y2[sel], z2[sel]
                ax_zoom.scatter(x2, y2, z2, c=[colors[idx]],
                                s=3, alpha=0.8,
                                depthshade=False, edgecolors='none')
            writer_zoom.grab_frame()

    plt.close(fig_full)
    plt.close(fig_zoom)

    t = np.arange(1, len(metrics['num_fibers'])+1)
    plt.figure()
    plt.plot(t, metrics['num_fibers'], marker='o')
    plt.xlabel("Frame"); plt.ylabel("Num Fibers")
    plt.savefig(os.path.join(save_path, 'num_labels_over_time.png'))

    plt.figure()
    plt.plot(t, metrics['avg_lengths'], marker='o')
    plt.xlabel("Frame"); plt.ylabel("Avg Length")
    plt.savefig(os.path.join(save_path, 'avg_lengths_over_time.png'))

    fig2, ax1 = plt.subplots()
    ax1.plot(t, metrics['avg_voxels'], marker='o', label="Avg Voxels")
    ax1.set_xlabel("Frame"); ax1.set_ylabel("Avg Voxels")
    ax2 = ax1.twinx()
    ax2.plot(t, metrics['gel_ratios'], marker='s', linestyle='--', label="Ratio")
    ax2.set_ylabel("Gel Ratio")
    fig2.tight_layout()
    plt.savefig(os.path.join(save_path, 'voxel_count_gel_ratio.png'))

    plt.figure()
    plt.plot(t, metrics['labeled_voxels'], marker='o')
    plt.xlabel("Frame"); plt.ylabel("Total Labeled Voxels")
    plt.savefig(os.path.join(save_path, 'labeled_voxels_over_time.png'))

    plt.figure()
    plt.plot(t, metrics['gel_metric'], marker='o')
    plt.xlabel("Frame"); plt.ylabel("Gel Fraction")
    plt.title("Fiber Gelation Over Time")
    plt.savefig(os.path.join(save_path, 'gelation_over_time.png'))

    print("Analysis complete. All plots and videos saved.")
    if gel_detected:
        print(f"Gel frame: {gel_frame}")
    else:
        print("No gelation detected.")

analyze_time_series('NewlyClassified\labeled_volumes', 'exp4/thresholdedData', min_size=125)
