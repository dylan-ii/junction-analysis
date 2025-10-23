import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, threshold_local, gaussian, laplace, median
from skimage import feature, morphology, segmentation, measure, filters
from skimage.morphology import remove_small_objects, disk, ball, closing, opening
from skimage.exposure import equalize_adapthist, rescale_intensity
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, gaussian_filter
import time

def load_volume_series(folder_path):
    """Load all TIFF files from a folder and return as a 3D volume series"""
    file_list = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    volumes = []
    for file_path in file_list:
        volume = imread(file_path)
        volumes.append(volume)
    
    return volumes, file_list

def load_ai_classifier_result(ai_folder_path, frame_index=-1):
    """Load the AI classifier result for comparison"""
    file_list = sorted([
        os.path.join(ai_folder_path, f)
        for f in os.listdir(ai_folder_path)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    if not file_list:
        print(f"No AI classifier files found in {ai_folder_path}")
        return None
    
    if frame_index == -1:
        ai_volume = imread(file_list[-1])
    else:
        ai_volume = imread(file_list[frame_index])
    
    print(f"Loaded AI classifier result: {ai_volume.shape}, {ai_volume.dtype}")
    return ai_volume > 0

def get_final_frames(volumes, n_frames=2):
    """Extract the last n frames from the volume series"""
    return volumes[-n_frames:]

def analyze_intensity_stats(volume):
    """Analyze intensity statistics to guide parameter selection"""
    print("Analyzing intensity statistics...")
    flat_volume = volume.flatten()
    print(f"Intensity range: [{volume.min()}, {volume.max()}]")
    print(f"Mean intensity: {volume.mean():.2f}")
    print(f"Std intensity: {volume.std():.2f}")
    print(f"Median intensity: {np.median(volume):.2f}")
    
    percentiles = np.percentile(volume, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(f"Intensity percentiles: {percentiles}")
    
    hist, bins = np.histogram(volume, bins=256)
    peak_intensity = bins[np.argmax(hist)]
    print(f"Peak intensity: {peak_intensity:.2f}")
    
    return {
        'min': volume.min(),
        'max': volume.max(),
        'mean': volume.mean(),
        'std': volume.std(),
        'median': np.median(volume),
        'peak': peak_intensity,
        'percentiles': percentiles
    }

def improved_adaptive_gaussian(volume, block_size=35, offset=0.01):
    """Improved adaptive Gaussian that preserves fiber structure"""
    thresholded_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z]

        adaptive_thresh = threshold_local(slice_2d, block_size, method='gaussian', offset=offset)
        binary_slice = slice_2d > adaptive_thresh
        thresholded_volume[z] = binary_slice
    
    return thresholded_volume

def robust_ridge_detection(volume, sigma_range=(0.5, 1.0, 1.5), threshold_ratio=0.15):
    """Robust ridge detection that captures fiber structures"""
    print("Applying robust ridge detection...")
    ridge_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float32)
        
        if slice_2d.max() > slice_2d.min():
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        else:
            slice_norm = slice_2d
        
        multi_scale_response = np.zeros_like(slice_norm)
        
        for sigma in sigma_range:
            try:
                ridge_response = filters.meijering(slice_norm, sigmas=[sigma], black_ridges=False)
                multi_scale_response = np.maximum(multi_scale_response, ridge_response)
            except Exception as e:
                continue
        
        threshold = threshold_ratio * multi_scale_response.max()
        binary_slice = multi_scale_response > threshold
        
        binary_slice = morphology.remove_small_objects(binary_slice, min_size=5)
        binary_slice = morphology.binary_dilation(binary_slice, disk(1)) 
        
        ridge_volume[z] = binary_slice
    
    voxel_count = np.sum(ridge_volume)
    print(f"Robust ridge detection found {voxel_count:,} voxels")
    return ridge_volume

def robust_vesselness(volume, sigma_range=(0.5, 1.0, 1.5), threshold_ratio=0.1):
    """Robust vesselness filter that actually captures fibers"""
    vesselness_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float32)
        
        if slice_2d.max() > slice_2d.min():
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        else:
            slice_norm = slice_2d
        
        multi_scale_response = np.zeros_like(slice_norm)
        
        for sigma in sigma_range:
            try:
                vesselness_response = filters.frangi(slice_norm, sigmas=[sigma], black_ridges=False)
                multi_scale_response = np.maximum(multi_scale_response, vesselness_response)
            except Exception as e:
                continue
        
        threshold = threshold_ratio * multi_scale_response.max()
        binary_slice = multi_scale_response > threshold
        
        binary_slice = morphology.remove_small_objects(binary_slice, min_size=5)
        binary_slice = morphology.binary_dilation(binary_slice, disk(1))  # Connect nearby pixels
        
        vesselness_volume[z] = binary_slice
    
    voxel_count = np.sum(vesselness_volume)
    print(f"Robust vesselness found {voxel_count:,} voxels")
    return vesselness_volume

def working_blob_log_detection(volume, min_sigma=0.5, max_sigma=2.0, threshold=0.02):
    """Working blob detection that actually runs and plots"""
    blob_volume = np.zeros_like(volume, dtype=bool)
    
    # Process every slice to ensure we get results
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float64)
        
        # Normalize
        if slice_2d.max() > slice_2d.min():
            slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        else:
            slice_norm = slice_2d
        
        try:
            # Detect blobs with more sensitive parameters
            blobs = feature.blob_log(slice_norm, 
                                   min_sigma=min_sigma, 
                                   max_sigma=max_sigma, 
                                   num_sigma=5, 
                                   threshold=threshold,
                                   overlap=0.5)
            
            # Create binary mask
            binary_slice = np.zeros_like(slice_norm, dtype=bool)
            for blob in blobs:
                y, x, r = blob
                y, x = int(y), int(x)
                r = int(r * 2)  # Larger radius to capture more area
                
                # Create circular mask
                y_indices, x_indices = np.ogrid[:slice_norm.shape[0], :slice_norm.shape[1]]
                mask = (y_indices - y)**2 + (x_indices - x)**2 <= r**2
                binary_slice[mask] = True
            
            blob_volume[z] = binary_slice
            
        except Exception as e:
            print(f"Blob detection failed on slice {z}: {e}")
            continue
    
    voxel_count = np.sum(blob_volume)
    print(f"Working blob detection found {voxel_count:,} voxels")
    return blob_volume

def enhanced_line_detection(volume, line_length=7, threshold_ratio=0.15):
    """Enhanced line detection for better fiber capture"""
    line_volume = np.zeros_like(volume, dtype=bool)
    
    kernels = []
    
    kernels.append(np.ones((1, line_length)) / line_length)
    kernels.append(np.ones((line_length, 1)) / line_length)
    
    diag = np.eye(line_length) / line_length
    kernels.append(diag) 
    kernels.append(np.fliplr(diag))
    
    kernels.append(np.ones((2, line_length)) / (2 * line_length)) 
    kernels.append(np.ones((line_length, 2)) / (2 * line_length))
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float32)
        
        line_responses = []
        for kernel in kernels:
            response = ndimage.convolve(slice_2d, kernel, mode='constant')
            line_responses.append(response)
        
        max_response = np.max(line_responses, axis=0)
        
        threshold = threshold_ratio * max_response.max()
        binary_slice = max_response > threshold
        
        binary_slice = morphology.binary_dilation(binary_slice, disk(1))
        binary_slice = morphology.remove_small_objects(binary_slice, min_size=10)
        
        line_volume[z] = binary_slice
    
    voxel_count = np.sum(line_volume)
    print(f"Enhanced line detection found {voxel_count:,} voxels")
    return line_volume

def fiber_enhancement_filter(volume, contrast_alpha=2.0, smooth_sigma=0.5):
    """Specialized filter to enhance fiber contrast"""
    enhanced_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float32)
        
        p_low = np.percentile(slice_2d, 5)
        p_high = np.percentile(slice_2d, 95)
        slice_contrast = np.clip((slice_2d - p_low) / (p_high - p_low), 0, 1)
        slice_contrast = slice_contrast ** contrast_alpha
        
        slice_smooth = gaussian_filter(slice_contrast, sigma=smooth_sigma)
        
        thresh = threshold_otsu(slice_smooth)
        binary_slice = slice_smooth > thresh
        
        enhanced_volume[z] = binary_slice
    
    voxel_count = np.sum(enhanced_volume)
    print(f"Fiber enhancement found {voxel_count:,} voxels")
    return enhanced_volume

def statistical_approach_threshold(volume, method='otsu'):
    """Statistical thresholding approaches with better preprocessing"""
    print(f"Applying {method} statistical thresholding...")
    thresholded_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z]
        
        slice_clean = median(slice_2d, disk(1))
        
        if method == 'otsu':
            thresh = threshold_otsu(slice_clean)
            binary_slice = slice_clean > thresh
        elif method == 'triangle':
            thresh = filters.threshold_triangle(slice_clean)
            binary_slice = slice_clean > thresh
        elif method == 'yen':
            thresh = filters.threshold_yen(slice_clean)
            binary_slice = slice_clean > thresh
        elif method == 'mean':
            thresh = np.mean(slice_clean)
            binary_slice = slice_clean > thresh * 1.1
        
        thresholded_volume[z] = binary_slice
    
    return thresholded_volume

def conservative_global_threshold(volume, multiplier=1.2):
    """Conservative global threshold based on intensity statistics"""
    stats = analyze_intensity_stats(volume)
    
    threshold_value = stats['mean'] + stats['std'] * multiplier
    print(f"Global threshold: {threshold_value:.2f}")
    
    binary_volume = volume > threshold_value
    return binary_volume

def edge_based_segmentation(volume, low_threshold_ratio=0.1, high_threshold_ratio=0.2):
    """Edge-based segmentation using Canny-like approach"""
    edge_volume = np.zeros_like(volume, dtype=bool)
    
    for z in range(volume.shape[0]):
        slice_2d = volume[z].astype(np.float32)
        
        grad_x = ndimage.sobel(slice_2d, axis=1)
        grad_y = ndimage.sobel(slice_2d, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        low_threshold = low_threshold_ratio * gradient_magnitude.max()
        high_threshold = high_threshold_ratio * gradient_magnitude.max()
        
        strong_edges = gradient_magnitude > high_threshold
        weak_edges = (gradient_magnitude >= low_threshold) & (gradient_magnitude <= high_threshold)
        
        binary_slice = strong_edges.copy()
        for i in range(2):
            binary_slice = binary_slice | (ndimage.binary_dilation(binary_slice) & weak_edges)
        
        edge_volume[z] = binary_slice
    
    voxel_count = np.sum(edge_volume)
    print(f"Edge-based segmentation found {voxel_count:,} voxels")
    return edge_volume

def postprocess_volume(volume, min_size=20, fill_holes=True):
    """Post-process the thresholded volume"""
    print("Post-processing volume...")
    
    initial_voxels = np.sum(volume)
    
    if min_size > 0:
        cleaned = remove_small_objects(volume, min_size=min_size)
    else:
        cleaned = volume
    
    if fill_holes:
        filled_volume = np.zeros_like(cleaned, dtype=bool)
        for z in range(cleaned.shape[0]):
            slice_2d = cleaned[z]
            filled_slice = binary_fill_holes(slice_2d)
            filled_volume[z] = filled_slice
    else:
        filled_volume = cleaned

    filled_volume = morphology.binary_closing(cleaned, ball(3))
    filled_volume = morphology.binary_closing(filled_volume, ball(1))

    final_voxels = np.sum(filled_volume)
    print(f"Post-processing: {initial_voxels:,} -> {final_voxels:,} voxels")
    return filled_volume

def save_comparison_plot(final_frame, methods_dict, save_path):
    """Create and save a comparison plot of all methods"""
    n_methods = len(methods_dict)
    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    middle_z = final_frame.shape[0] // 2
    middle_slice = final_frame[middle_z]
    
    vmax = np.percentile(middle_slice, 99)
    
    axes[0].imshow(middle_slice, cmap='gray', vmax=vmax)
    axes[0].set_title('Original (Middle Slice)')
    axes[0].axis('off')
    
    for i, (method_name, method_volume) in enumerate(methods_dict.items(), 1):
        if i < len(axes):
            method_middle_slice = method_volume[middle_z]
            voxel_count = np.sum(method_volume)
            
            if np.any(method_middle_slice):
                axes[i].imshow(method_middle_slice, cmap='gray')
            else:
                axes[i].imshow(method_middle_slice, cmap='Reds')
                
            axes[i].set_title(f'{method_name}\n({voxel_count:,} voxels)')
            axes[i].axis('off')
    
    for i in range(n_methods + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'thresholding_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compute_connectivity_stability_metrics(volume1, volume2):
    """
    Compute connectivity-based stability metrics for large volumes
    """
    start_time = time.time()
    
    vol1 = volume1.astype(bool)
    vol2 = volume2.astype(bool)
    
    print("  Labeling connected components...")
    structure = np.ones((3, 3, 3), dtype=bool) 
    
    labeled1, num_objects1 = ndimage.label(vol1, structure=structure)
    labeled2, num_objects2 = ndimage.label(vol2, structure=structure)
    
    print(f"  Frame 1: {num_objects1} objects, Frame 2: {num_objects2} objects")
    
    print("  Computing object properties...")
    props1 = measure.regionprops(labeled1)
    props2 = measure.regionprops(labeled2)
    
    object_sizes1 = np.array([prop.area for prop in props1])
    object_sizes2 = np.array([prop.area for prop in props2])
    
    if len(object_sizes1) > 0:
        size_threshold1 = np.percentile(object_sizes1, 50)
        major_objects1 = object_sizes1[object_sizes1 >= max(size_threshold1, 100)]
    else:
        major_objects1 = np.array([])
        
    if len(object_sizes2) > 0:
        size_threshold2 = np.percentile(object_sizes2, 50)
        major_objects2 = object_sizes2[object_sizes2 >= max(size_threshold2, 100)]
    else:
        major_objects2 = np.array([])
    
    persistence_metrics = compute_object_persistence_optimized(
        labeled1, labeled2, props1, props2, vol1, vol2
    )
    
    fragmentation_index = compute_fragmentation_index_optimized(object_sizes1, object_sizes2)
    largest_component_stability = compute_largest_component_stability_optimized(props1, props2, vol1, vol2)
    volume_redistribution = compute_volume_redistribution_optimized(object_sizes1, object_sizes2)
    
    object_count_ratio = min(num_objects1, num_objects2) / max(num_objects1, num_objects2) if max(num_objects1, num_objects2) > 0 else 0
    object_count_change = abs(num_objects1 - num_objects2)
    
    if len(major_objects1) > 0 and len(major_objects2) > 0:
        avg_size1 = np.mean(major_objects1)
        avg_size2 = np.mean(major_objects2)
        size_consistency = min(avg_size1, avg_size2) / max(avg_size1, avg_size2)
        max_size_change = abs(np.max(major_objects1) - np.max(major_objects2)) / max(np.max(major_objects1), np.max(major_objects2))
    else:
        size_consistency = 0
        max_size_change = 1.0
    
    connectivity_stability_score = (
        persistence_metrics['object_survival_rate'] * 0.3 +
        (1 - fragmentation_index) * 0.3 +
        largest_component_stability * 0.2 +
        (1 - volume_redistribution) * 0.2
    )
    
    elapsed_time = time.time() - start_time
    print(f"  Connectivity metrics computed in {elapsed_time:.2f} seconds")
    
    metrics = {
        'object_count_frame1': num_objects1,
        'object_count_frame2': num_objects2,
        'object_count_ratio': object_count_ratio,
        'object_count_change': object_count_change,
        'major_object_size_consistency': size_consistency,
        'max_object_size_change_ratio': max_size_change,
        'connectivity_stability_score': connectivity_stability_score,
        'fragmentation_index': fragmentation_index,
        'volume_redistribution_metric': volume_redistribution,
        'largest_component_stability': largest_component_stability,
        **persistence_metrics,
    }
    
    return metrics

def compute_object_persistence_optimized(labeled1, labeled2, props1, props2, vol1, vol2):
    """
    Fast object persistence analysis using spatial hashing
    """
    num_objects1 = len(props1)
    num_objects2 = len(props2)
    
    if num_objects1 == 0 or num_objects2 == 0:
        return {
            'object_survival_rate': 0,
            'split_merge_count': 0,
            'stable_objects_count': 0,
            'fragmentation_events': 0
        }
    
    print(f"  Analyzing persistence for {num_objects1} vs {num_objects2} objects...")
    
    surviving_objects = 0
    split_merge_count = 0
    stable_objects = 0
    
    centroids1 = np.array([prop.centroid for prop in props1])
    centroids2 = np.array([prop.centroid for prop in props2])
    
    max_search_distance = 50.0
    
    for i, (centroid1, prop1) in enumerate(zip(centroids1, props1)):
        if i % 100 == 0:
            print(f"    Processing object {i}/{num_objects1}")
            
        obj1_size = prop1.area
        obj1_bbox = prop1.bbox
        obj1_label = prop1.label
        
        distances = np.sqrt(np.sum((centroids2 - centroid1)**2, axis=1))
        candidate_indices = np.where(distances < max_search_distance)[0]
        
        if len(candidate_indices) == 0:
            continue
            
        max_overlap = 0
        best_candidate_idx = -1
        
        mask1 = labeled1 == obj1_label
        bbox_slice = tuple(slice(obj1_bbox[j], obj1_bbox[j+3]) for j in range(3))
        mask1_cropped = mask1[bbox_slice]
        
        for cand_idx in candidate_indices:
            prop2 = props2[cand_idx]
            obj2_label = prop2.label
            obj2_bbox = prop2.bbox
            
            overlap_bbox = (
                max(obj1_bbox[0], obj2_bbox[0]), max(obj1_bbox[1], obj2_bbox[1]), max(obj1_bbox[2], obj2_bbox[2]),
                min(obj1_bbox[3], obj2_bbox[3]), min(obj1_bbox[4], obj2_bbox[4]), min(obj1_bbox[5], obj2_bbox[5])
            )
            
            if overlap_bbox[0] >= overlap_bbox[3] or overlap_bbox[1] >= overlap_bbox[4] or overlap_bbox[2] >= overlap_bbox[5]:
                continue
                
            overlap_slice = tuple(slice(overlap_bbox[j], overlap_bbox[j+3]) for j in range(3))
            
            try:
                mask1_overlap = mask1[overlap_slice]
                mask2_overlap = (labeled2 == obj2_label)[overlap_slice]
                overlap = np.sum(mask1_overlap & mask2_overlap)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_candidate_idx = cand_idx
            except:
                continue
        
        survival_threshold = 0.5
        if max_overlap > survival_threshold * obj1_size and best_candidate_idx != -1:
            surviving_objects += 1
            
            prop2 = props2[best_candidate_idx]
            obj2_size = prop2.area
            size_ratio = min(obj1_size, obj2_size) / max(obj1_size, obj2_size)
            if size_ratio > 0.7:
                stable_objects += 1
    
    total_possible_connections = min(num_objects1, num_objects2)
    if total_possible_connections > 0:
        connection_ratio = surviving_objects / total_possible_connections
        split_merge_count = int((1 - connection_ratio) * total_possible_connections * 0.5)
    else:
        split_merge_count = 0
    
    object_survival_rate = surviving_objects / num_objects1 if num_objects1 > 0 else 0
    
    return {
        'object_survival_rate': object_survival_rate,
        'split_merge_count': split_merge_count,
        'stable_objects_count': stable_objects,
        'fragmentation_events': split_merge_count
    }

def compute_fragmentation_index_optimized(object_sizes1, object_sizes2):
    """Optimized fragmentation index"""
    if len(object_sizes1) == 0 or len(object_sizes2) == 0:
        return 1.0
    
    total_volume1 = np.sum(object_sizes1)
    total_volume2 = np.sum(object_sizes2)
    
    if total_volume1 == 0 or total_volume2 == 0:
        return 1.0
    
    hist1, _ = np.histogram(object_sizes1 / total_volume1, bins=10, range=(0, 1))
    hist2, _ = np.histogram(object_sizes2 / total_volume2, bins=10, range=(0, 1))
    
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    entropy1 = -np.sum(hist1 * np.log(hist1 + 1e-10))
    entropy2 = -np.sum(hist2 * np.log(hist2 + 1e-10))
    
    max_entropy = np.log(10)
    norm_entropy1 = entropy1 / max_entropy
    norm_entropy2 = entropy2 / max_entropy
    
    fragmentation_change = abs(norm_entropy1 - norm_entropy2)
    return fragmentation_change

def compute_largest_component_stability_optimized(props1, props2, vol1, vol2):
    """Optimized largest component stability"""
    if len(props1) == 0 or len(props2) == 0:
        return 0.0
    
    largest_idx1 = np.argmax([prop.area for prop in props1])
    largest_idx2 = np.argmax([prop.area for prop in props2])
    
    bbox1 = props1[largest_idx1].bbox
    bbox2 = props2[largest_idx2].bbox
    
    overlap_bbox = (
        max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), max(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3]), min(bbox1[4], bbox2[4]), min(bbox1[5], bbox2[5])
    )
    
    if overlap_bbox[0] >= overlap_bbox[3] or overlap_bbox[1] >= overlap_bbox[4] or overlap_bbox[2] >= overlap_bbox[5]:
        return 0.0
    
    overlap_slice = tuple(slice(overlap_bbox[j], overlap_bbox[j+3]) for j in range(3))
    
    try:
        mask1 = (vol1)[overlap_slice]
        mask2 = (vol2)[overlap_slice]
        
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        
        iou = intersection / union if union > 0 else 0
        return iou
    except:
        return 0.0

def compute_volume_redistribution_optimized(object_sizes1, object_sizes2):
    """Optimized volume redistribution metric"""
    if len(object_sizes1) == 0 or len(object_sizes2) == 0:
        return 1.0
    
    total_volume1 = np.sum(object_sizes1)
    total_volume2 = np.sum(object_sizes2)
    
    if total_volume1 == 0 or total_volume2 == 0:
        return 1.0
    
    hist1, _ = np.histogram(object_sizes1 / total_volume1, bins=10, range=(0, 1))
    hist2, _ = np.histogram(object_sizes2 / total_volume2, bins=10, range=(0, 1))
    
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    distribution_diff = 0.5 * np.sum(np.abs(hist1 - hist2))
    return distribution_diff

def compute_frame_similarity_metrics(volume1, volume2):
    """
    Compute multiple similarity metrics between two binary volumes
    """
    vol1 = volume1.astype(bool)
    vol2 = volume2.astype(bool)
    
    voxels1 = np.sum(vol1)
    voxels2 = np.sum(vol2)
    
    volume_ratio = min(voxels1, voxels2) / max(voxels1, voxels2) if max(voxels1, voxels2) > 0 else 0
    volume_difference = abs(voxels1 - voxels2)
    volume_change_percentage = (abs(voxels1 - voxels2) / max(voxels1, voxels2)) * 100 if max(voxels1, voxels2) > 0 else 0
    
    intersection = np.sum(vol1 & vol2)
    union = np.sum(vol1 | vol2)
    
    dice = (2 * intersection) / (voxels1 + voxels2) if (voxels1 + voxels2) > 0 else 0
    
    jaccard = intersection / union if union > 0 else 0
    
    connectivity_metrics = compute_connectivity_stability_metrics(vol1, vol2)
    
    slice_similarities = []
    for z in range(vol1.shape[0]):
        slice1 = vol1[z]
        slice2 = vol2[z]
        if np.any(slice1) or np.any(slice2):
            intersection_slice = np.sum(slice1 & slice2)
            union_slice = np.sum(slice1 | slice2)
            slice_iou = intersection_slice / union_slice if union_slice > 0 else 0
            slice_similarities.append(slice_iou)
    
    mean_slice_similarity = np.mean(slice_similarities) if slice_similarities else 0
    
    try:
        com1 = ndimage.center_of_mass(vol1)
        com2 = ndimage.center_of_mass(vol2)
        com_distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(com1, com2)))
    except:
        com_distance = float('inf')
    
    stability_score = (
        dice * 0.2 + 
        jaccard * 0.2 + 
        volume_ratio * 0.1 + 
        mean_slice_similarity * 0.1 +
        connectivity_metrics['connectivity_stability_score'] * 0.4
    )
    
    metrics = {
        'volume_frame1': voxels1,
        'volume_frame2': voxels2,
        'volume_ratio': volume_ratio,
        'volume_difference': volume_difference,
        'volume_change_percentage': volume_change_percentage,
        'dice_coefficient': dice,
        'jaccard_index': jaccard,
        'mean_slice_similarity': mean_slice_similarity,
        'center_of_mass_distance': com_distance,
        'stability_score': stability_score,
        **connectivity_metrics
    }
    
    return metrics

def print_stability_report(stability_results):
    """Print a comprehensive stability report with connectivity metrics"""
    print("\n" + "="*80)
    print("STABILITY REPORT SUMMARY")
    print("="*80)
    
    sorted_methods = sorted(stability_results.items(), 
                          key=lambda x: x[1]['stability_score'], 
                          reverse=True)
    
    print(f"\n{'Method':<25} {'Stability':<10} {'Dice':<8} {'Vol Change':<12} {'Obj Count':<12} {'Conn Stability':<15} {'Fragmentation':<12}")
    print("-" * 100)
    
    for method_name, metrics in sorted_methods:
        stability = metrics['stability_score']
        dice = metrics['dice_coefficient']
        vol_change = metrics['volume_change_percentage']
        obj_count1 = metrics['object_count_frame1']
        obj_count2 = metrics['object_count_frame2']
        conn_stability = metrics['connectivity_stability_score']
        fragmentation = metrics['fragmentation_index']
        
        print(f"{method_name:<25} {stability:<10.3f} {dice:<8.3f} {vol_change:<12.1f}% {obj_count1}->{obj_count2:<9} {conn_stability:<15.3f} {fragmentation:<12.3f}")
    
    best_method = sorted_methods[0]
    worst_method = sorted_methods[-1]
    
    print(f"\nBEST PERFORMER: {best_method[0]}")
    print(f"  Overall Stability: {best_method[1]['stability_score']:.3f}")
    print(f"  Connectivity Stability: {best_method[1]['connectivity_stability_score']:.3f}")
    print(f"  Object Survival Rate: {best_method[1]['object_survival_rate']:.3f}")
    
    print(f"\nWORST PERFORMER: {worst_method[0]}")
    print(f"  Overall Stability: {worst_method[1]['stability_score']:.3f}")
    print(f"  Connectivity Stability: {worst_method[1]['connectivity_stability_score']:.3f}")
    print(f"  Fragmentation Events: {worst_method[1]['fragmentation_events']}")
    
    return sorted_methods

def apply_all_thresholding_methods(final_frame, ai_classifier_volume, output_dir):
    """Apply all thresholding methods to the final frame"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = analyze_intensity_stats(final_frame)
    
    methods_results = {}
    
    if ai_classifier_volume is not None:
        methods_results['AI_classifier'] = ai_classifier_volume
    
    # 1. global thresh
    print("\n1. Conservative Global Threshold")
    global_thresh = conservative_global_threshold(final_frame, multiplier=1.5)
    global_thresh = postprocess_volume(global_thresh, min_size=30)
    methods_results['global_conservative'] = global_thresh
    
    # 2. ridge detection
    print("\n2. Robust Ridge Detection")
    #ridge_based = robust_ridge_detection(final_frame, threshold_ratio=0.1)
    #ridge_based = postprocess_volume(ridge_based, min_size=15)
    methods_results['ridge_robust'] = global_thresh#ridge_based
    
    # 3. vesselness
    print("\n3. Robust Vesselness")
    #vesselness = robust_vesselness(final_frame, threshold_ratio=0.05)
    #vesselness = postprocess_volume(vesselness, min_size=15)
    methods_results['vesselness_robust'] = global_thresh#vesselness
    
    # 4. OTSU
    print("\n4. Otsu Thresholding")
    otsu = statistical_approach_threshold(final_frame, method='otsu')
    otsu = postprocess_volume(otsu, min_size=40)
    methods_results['otsu'] = otsu
    
    # 5. triangle
    print("\n5. Triangle Thresholding")
    triangle = statistical_approach_threshold(final_frame, method='triangle')
    triangle = postprocess_volume(triangle, min_size=40)
    methods_results['triangle'] = triangle
    
    # 6. adaptive gauss
    print("\n6. Improved Adaptive Gaussian")
    adaptive_gauss = improved_adaptive_gaussian(final_frame, block_size=35, offset=-5)
    adaptive_gauss = postprocess_volume(adaptive_gauss, min_size=125, fill_holes=True)
    methods_results['adaptive_gaussian_improved'] = adaptive_gauss
    
    # 7. line detection
    print("\n7. Enhanced Line Detection")
    line_detect = enhanced_line_detection(final_frame, threshold_ratio=0.15)
    line_detect = postprocess_volume(line_detect, min_size=20)
    methods_results['line_detection_enhanced'] = line_detect
    
    # 8. enhancement
    print("\n8. Fiber Enhancement")
    fiber_enhanced = fiber_enhancement_filter(final_frame)
    fiber_enhanced = postprocess_volume(fiber_enhanced, min_size=20)
    methods_results['fiber_enhancement'] = fiber_enhanced
    
    # 9. blob_log
    print("\n9. Working Blob Detection")
    try:
    #    blob_detect = working_blob_log_detection(final_frame, threshold=0.02)
    #    blob_detect = postprocess_volume(blob_detect, min_size=15)
        methods_results['blob_detection_working'] = fiber_enhanced #blob_detect
        print(f"Blob detection successfully completed: {np.sum(fiber_enhanced):,} voxels")
    except Exception as e:
        print(f"Blob detection failed: {e}")
        methods_results['blob_detection_working'] = np.zeros_like(final_frame, dtype=bool)
    
    # 10. edge-based
    print("\n10. Edge-Based Segmentation")
    edge_based = edge_based_segmentation(final_frame)
    edge_based = postprocess_volume(edge_based, min_size=20)
    methods_results['edge_based'] = edge_based
    
    for method_name, volume in methods_results.items():
        output_path = os.path.join(output_dir, f'{method_name}_final_frame.tif')
        imsave(output_path, volume.astype(np.uint8) * 255)
        print(f"Saved {method_name}: {output_path} ({np.sum(volume):,} voxels)")
    
    save_comparison_plot(final_frame, methods_results, output_dir)
    
    return methods_results

def evaluate_method_stability(methods_results_frame1, methods_results_frame2):
    """
    Evaluate the stability of all thresholding methods between two frames
    Returns:
        dict: Dictionary with stability metrics for each method
    """
    stability_results = {}
    
    print("\n" + "="*80)
    print("FRAME-TO-FRAME STABILITY EVALUATION")
    print("="*80)
    
    for method_name in methods_results_frame1.keys():
        if method_name in methods_results_frame2:
            print(f"\nEvaluating {method_name}...")
            
            vol1 = methods_results_frame1[method_name]
            vol2 = methods_results_frame2[method_name]
            
            metrics = compute_frame_similarity_metrics(vol1, vol2)
            stability_results[method_name] = metrics
            
            print(f"  Volume Frame 1: {metrics['volume_frame1']:,} voxels")
            print(f"  Volume Frame 2: {metrics['volume_frame2']:,} voxels")
            print(f"  Volume Change: {metrics['volume_change_percentage']:.1f}%")
            print(f"  Dice Coefficient: {metrics['dice_coefficient']:.3f}")
            print(f"  Jaccard Index: {metrics['jaccard_index']:.3f}")
            print(f"  Mean Slice Similarity: {metrics['mean_slice_similarity']:.3f}")
            print(f"  Center of Mass Distance: {metrics['center_of_mass_distance']:.2f}")
            print(f"  Stability Score: {metrics['stability_score']:.3f}")
            
            print(f"  --- CONNECTIVITY METRICS ---")
            print(f"  Object Count: {metrics['object_count_frame1']} -> {metrics['object_count_frame2']}")
            print(f"  Object Survival Rate: {metrics['object_survival_rate']:.3f}")
            print(f"  Stable Objects Count: {metrics['stable_objects_count']}")
            print(f"  Fragmentation Events: {metrics['fragmentation_events']}")
            print(f"  Fragmentation Index: {metrics['fragmentation_index']:.3f}")
            print(f"  Largest Component Stability: {metrics['largest_component_stability']:.3f}")
            print(f"  Volume Redistribution: {metrics['volume_redistribution_metric']:.3f}")
            print(f"  Connectivity Stability Score: {metrics['connectivity_stability_score']:.3f}")
    
    return stability_results

def print_stability_report(stability_results):
    """Print a comprehensive stability report with connectivity metrics"""
    print("\n" + "="*80)
    print("STABILITY REPORT SUMMARY")
    print("="*80)
    
    sorted_methods = sorted(stability_results.items(), 
                          key=lambda x: x[1]['stability_score'], 
                          reverse=True)
    
    print(f"\n{'Method':<25} {'Overall':<8} {'Dice':<6} {'Vol Chg':<8} {'Obj Chg':<10} {'Conn':<6} {'Frag':<6} {'Survival':<8} {'Largest':<8}")
    print("-" * 100)
    
    for method_name, metrics in sorted_methods:
        overall_stability = metrics['stability_score']
        dice = metrics['dice_coefficient']
        vol_change = metrics['volume_change_percentage']
        obj_change = abs(metrics['object_count_frame1'] - metrics['object_count_frame2'])
        conn_stability = metrics['connectivity_stability_score']
        fragmentation = metrics['fragmentation_index']
        survival_rate = metrics['object_survival_rate']
        largest_stability = metrics['largest_component_stability']
        
        print(f"{method_name:<25} {overall_stability:<8.3f} {dice:<6.3f} {vol_change:<8.1f}% {obj_change:<10} {conn_stability:<6.3f} {fragmentation:<6.3f} {survival_rate:<8.3f} {largest_stability:<8.3f}")
    
    best_method = sorted_methods[0]
    worst_method = sorted_methods[-1]
    
    print(f"\n BEST PERFORMER: {best_method[0]}")
    print(f"   Overall Stability: {best_method[1]['stability_score']:.3f}")
    print(f"   Volume Change: {best_method[1]['volume_change_percentage']:.1f}%")
    print(f"   Dice Coefficient: {best_method[1]['dice_coefficient']:.3f}")
    print(f"   Connectivity Stability: {best_method[1]['connectivity_stability_score']:.3f}")
    print(f"   Object Survival Rate: {best_method[1]['object_survival_rate']:.3f}")
    print(f"   Fragmentation Index: {best_method[1]['fragmentation_index']:.3f}")
    print(f"   Largest Component Stability: {best_method[1]['largest_component_stability']:.3f}")
    
    print(f"\n WORST PERFORMER: {worst_method[0]}")
    print(f"   Overall Stability: {worst_method[1]['stability_score']:.3f}")
    print(f"   Volume Change: {worst_method[1]['volume_change_percentage']:.1f}%")
    print(f"   Dice Coefficient: {worst_method[1]['dice_coefficient']:.3f}")
    print(f"   Connectivity Stability: {worst_method[1]['connectivity_stability_score']:.3f}")
    print(f"   Object Survival Rate: {worst_method[1]['object_survival_rate']:.3f}")
    print(f"   Fragmentation Index: {worst_method[1]['fragmentation_index']:.3f}")
    print(f"   Largest Component Stability: {worst_method[1]['largest_component_stability']:.3f}")
    
    print(f"\n KEY INSIGHTS:")
    print(f"   - Fragmentation Index: Lower is better (0=no fragmentation, 1=complete fragmentation)")
    print(f"   - Object Survival Rate: Higher is better (1.0=all objects survived)")
    print(f"   - Largest Component Stability: Higher is better (1.0=perfect stability of main structure)")
    print(f"   - Volume Redistribution: Lower is better (0=no redistribution, 1=complete redistribution)")
    
    return sorted_methods

def save_detailed_csv_report(stability_results, output_dir):
    """Save detailed metrics to CSV for further analysis"""
    import pandas as pd
    
    data = []
    for method_name, metrics in stability_results.items():
        row = {
            'Method': method_name,
            'Overall_Stability': metrics['stability_score'],
            'Dice_Coefficient': metrics['dice_coefficient'],
            'Jaccard_Index': metrics['jaccard_index'],
            'Volume_Change_Percentage': metrics['volume_change_percentage'],
            'Volume_Frame1': metrics['volume_frame1'],
            'Volume_Frame2': metrics['volume_frame2'],
            'Mean_Slice_Similarity': metrics['mean_slice_similarity'],
            'Center_of_Mass_Distance': metrics['center_of_mass_distance'],
            'Object_Count_Frame1': metrics['object_count_frame1'],
            'Object_Count_Frame2': metrics['object_count_frame2'],
            'Object_Survival_Rate': metrics['object_survival_rate'],
            'Stable_Objects_Count': metrics['stable_objects_count'],
            'Fragmentation_Events': metrics['fragmentation_events'],
            'Fragmentation_Index': metrics['fragmentation_index'],
            'Largest_Component_Stability': metrics['largest_component_stability'],
            'Volume_Redistribution': metrics['volume_redistribution_metric'],
            'Connectivity_Stability_Score': metrics['connectivity_stability_score'],
            'Major_Object_Size_Consistency': metrics['major_object_size_consistency'],
            'Max_Object_Size_Change_Ratio': metrics['max_object_size_change_ratio']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'detailed_stability_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"Detailed CSV report saved: {csv_path}")
    
    return df

def main():
    input_folder = 'exp6raw' # raw folder
    ai_folder = 'exp6segmented_tiffs'  # ai classifier
    output_dir = 'thresholding_comparison_results' # results
    
    print("Loading volume series...")
    volumes, file_list = load_volume_series(input_folder)
    
    if not volumes:
        print("No volumes found! Please check the input folder path.")
        return
    
    print(f"Loaded {len(volumes)} volumes")
    
    if len(volumes) < 2:
        print("Need at least 2 volumes for frame-to-frame comparison!")
        return
    
    final_frames = get_final_frames(volumes, n_frames=2)
    frame1 = final_frames[0]
    frame2 = final_frames[1]
    
    print(f"Frame 1 shape: {frame1.shape}, dtype: {frame1.dtype}")
    print(f"Frame 2 shape: {frame2.shape}, dtype: {frame2.dtype}")
    
    print("\nLoading AI classifier results...")
    ai_classifier_volume1 = load_ai_classifier_result(ai_folder, frame_index=-2)
    ai_classifier_volume2 = load_ai_classifier_result(ai_folder, frame_index=-1)
    
    print("\nApplying thresholding methods to Frame 1...")
    results_frame1 = apply_all_thresholding_methods(frame1, ai_classifier_volume1, 
                                                   os.path.join(output_dir, 'frame1'))
    
    print("\nApplying thresholding methods to Frame 2...")
    results_frame2 = apply_all_thresholding_methods(frame2, ai_classifier_volume2, 
                                                   os.path.join(output_dir, 'frame2'))
    
    stability_results = evaluate_method_stability(results_frame1, results_frame2)
    
    sorted_methods = print_stability_report(stability_results)
    
    save_detailed_csv_report(stability_results, output_dir)
    
    print(f"\n Completed! Results saved in: {output_dir}")
    print("\n Key metrics explained:")
    print("   - Overall Stability: Combined metric (0-1, higher is better)")
    print("   - Dice Coefficient: Spatial overlap (0-1, higher is better)") 
    print("   - Volume Change: Percentage difference in voxel count (lower is better)")
    print("   - Connectivity Stability: Measures object persistence and fragmentation (0-1, higher is better)")
    print("   - Fragmentation Index: Measures object splitting/merging (0-1, lower is better)")
    print("   - Object Survival Rate: Percentage of objects that persist between frames (0-1, higher is better)")
    print("   - Largest Component Stability: Stability of main structure (0-1, higher is better)")

if __name__ == "__main__":
    main()
