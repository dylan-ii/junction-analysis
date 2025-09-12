import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.morphology import binary_closing, ball, skeletonize, thin
from scipy.ndimage import convolve, label as cc_label
import tifffile
from scipy.ndimage import distance_transform_edt
from collections import deque
import cv2
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

DATA_DIR = "CylShapeDecomposition-master/CSD/labeled_volumes_full"
G_RADIUS = 2
G_RES = 0.2
H_TH = 0.95  
SHIFT_IMPOSE = 1    
EULER_STEP_SIZE = 0.2
MIN_OBJ_SIZE = 500
MAX_WORKERS = 16

RECORD_ALL_OBJECTS_PLOT = True
ALL_OBJECTS_PLOTS_DIR = "all_objects_plots"
RECORD_INTERVAL = 252 

PLOT_FIGURES = True
RECORD_PLOTS = True
PLOTS_DIR = "decomposed_plots"

RECORD_NODE_DEGREE_CSV = True
NODE_DEGREE_CSV = "node_degree_stats3.csv"

if RECORD_PLOTS and not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if RECORD_ALL_OBJECTS_PLOT and not os.path.exists(ALL_OBJECTS_PLOTS_DIR):
    os.makedirs(ALL_OBJECTS_PLOTS_DIR)

OFFSETS = np.array([[i, j, k]
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                    for k in (-1, 0, 1)
                    if not (i == 0 and j == 0 and k == 0)])

distance_cache = {}

def point_to_line_distance(point, line_start, line_end):
    vector = line_end - line_start
    length_squared = np.dot(vector, vector)
    
    if length_squared == 0:
        return np.linalg.norm(point - line_start)
    
    t = max(0, min(1, np.dot(point - line_start, vector) / length_squared))
    projection = line_start + t * vector
    return np.linalg.norm(point - projection)

def should_merge_junctions(juncA, juncB, skel_vol, max_euclidean_dist=15, max_path_dist=10, max_deviation=1.5):
    euclidean_dist = np.linalg.norm(np.array(juncA) - np.array(juncB))
    if euclidean_dist > max_euclidean_dist:
        return False
    
    # shortest path
    path_dist, path = find_shortest_path(juncA, juncB, skel_vol)
    if path_dist > max_path_dist or path is None:
        return False
    
    juncA_np = np.array(juncA)
    juncB_np = np.array(juncB)
    
    # check for deviation
    for point in path[1:-1]:
        point_np = np.array(point)
        dist = point_to_line_distance(point_np, juncA_np, juncB_np)
        if dist > max_deviation:
            return False
    
    return True

def find_shortest_path(start, end, skel_vol):
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        pos, path = queue.popleft()
        if pos == end:
            return len(path) - 1, path
            
        for off in OFFSETS:
            neighbor = (pos[0] + off[0], pos[1] + off[1], pos[2] + off[2])
            
            if not (0 <= neighbor[0] < skel_vol.shape[0] and
                    0 <= neighbor[1] < skel_vol.shape[1] and
                    0 <= neighbor[2] < skel_vol.shape[2]):
                continue
                
            if skel_vol[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
                
    return float('inf'), None

def merge_junction_clusters(junction_centroids, cluster_map, skel_vol, max_distance=12.0):
    # cluster sets
    clusters = {}
    for voxel, label in cluster_map.items():
        clusters.setdefault(label, set()).add(voxel)
    
    # graph of possible merges
    merge_graph = {}
    centroids_dict = {label: np.array(centroid) for centroid, label in junction_centroids}
    labels = [label for _, label in junction_centroids]
    
    # KDTree for distance queries
    from scipy.spatial import cKDTree
    points = np.array([centroids_dict[label] for label in labels])
    kdtree = cKDTree(points)
    
    pairs = kdtree.query_pairs(max_distance)
    
    # check if pairs should be merged
    for i, j in pairs:
        labelA = labels[i]
        labelB = labels[j]
        
        voxelA = next(iter(clusters[labelA]))
        voxelB = next(iter(clusters[labelB]))
        
        if should_merge_junctions(voxelA, voxelB, skel_vol, 
                                max_euclidean_dist=max_distance, 
                                max_path_dist=max_distance, 
                                max_deviation=10):
            merge_graph.setdefault(labelA, set()).add(labelB)
            merge_graph.setdefault(labelB, set()).add(labelA)
    
    visited = set()
    merged_clusters = []
    label_mapping = {}
    
    for label in labels:
        if label in visited:
            continue
            
        component = set()
        stack = [label]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.add(current)
                stack.extend(merge_graph.get(current, set()) - visited)
        
        if len(component) > 1:
            all_voxels = set()
            for lbl in component:
                all_voxels |= clusters[lbl]
            
            center = np.mean(list(all_voxels), axis=0)
            
            candidate_centroids = [centroids_dict[lbl] for lbl in component]
            dists = [np.linalg.norm(c - center) for c in candidate_centroids]
            centroid = tuple(candidate_centroids[np.argmin(dists)].astype(int))
            
            new_label = min(component)
            merged_clusters.append((centroid, new_label))
            
            for lbl in component:
                label_mapping[lbl] = new_label
        else:
            centroid = tuple(centroids_dict[label])
            merged_clusters.append((centroid, label))
            label_mapping[label] = label
    
    new_cluster_map = {}
    for voxel, label in cluster_map.items():
        new_cluster_map[voxel] = label_mapping.get(label, label)
    
    return merged_clusters, new_cluster_map

def are_clusters_connected(voxelA, voxelB, skel_vol, max_path_length):
    visited = set()
    queue = deque([(voxelA, 0)])
    visited.add(voxelA)
    
    while queue:
        current, dist = queue.popleft()
        
        if current == voxelB:
            return True
            
        if dist >= max_path_length:
            continue
            
        for off in OFFSETS:
            nbr = (current[0] + off[0], current[1] + off[1], current[2] + off[2])
            
            if not (0 <= nbr[0] < skel_vol.shape[0] and
                    0 <= nbr[1] < skel_vol.shape[1] and
                    0 <= nbr[2] < skel_vol.shape[2]):
                continue
                
            if skel_vol[nbr] and nbr not in visited:
                visited.add(nbr)
                queue.append((nbr, dist + 1))
                
    return False

def analyze_branch_characteristics(skel_vol, closed_obj):
    cache_key = hash(closed_obj.tobytes())
    if cache_key in distance_cache:
        distance_map = distance_cache[cache_key]
    else:
        distance_map = distance_transform_edt(closed_obj)
        distance_cache[cache_key] = distance_map
    
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    
    neighbor_count = np.zeros_like(skel_vol, dtype=int)
    convolve(skel_vol.astype(int), kernel, output=neighbor_count, mode="constant", cval=0)
    
    junctions = {tuple(c) for c in np.argwhere(skel_vol & (neighbor_count > 2))}
    endpoints = {tuple(c) for c in np.argwhere(skel_vol & (neighbor_count == 1))}
    
    if not junctions:
        return skel_vol.copy()
    
    branches = find_all_branches_optimized(skel_vol, junctions, endpoints)
    
    pruned_vol = skel_vol.copy()
    
    for branch in branches:
        if len(branch) < 2:
            continue
            
        # check for short connection
        start, end = branch[0], branch[-1]
        if start in junctions and end in junctions and len(branch) <= 5: 
            if not is_legitimate_connection_optimized(branch, closed_obj, skel_vol, distance_map):
                for point in branch[1:-1]:
                    pruned_vol[point] = False
    
    return pruned_vol

def find_all_branches_optimized(skel_vol, junctions, endpoints):
    branches = []
    visited = set()
    shape = skel_vol.shape
    
    skel_points = set(zip(*np.where(skel_vol)))
    
    start_points = junctions | endpoints
    
    for point in start_points:
        if point in visited:
            continue
            
        # check all neighbors
        for dx, dy, dz in OFFSETS:
            neighbor = (point[0] + dx, point[1] + dy, point[2] + dz)
            
            if not (0 <= neighbor[0] < shape[0] and
                    0 <= neighbor[1] < shape[1] and
                    0 <= neighbor[2] < shape[2]):
                continue
                
            if neighbor in skel_points and neighbor not in start_points and neighbor not in visited:
                branch = [point]
                current = neighbor
                prev = point
                
                while current not in start_points:
                    if current in visited:
                        break
                        
                    branch.append(current)
                    visited.add(current)
                    
                    next_points = []
                    for dx2, dy2, dz2 in OFFSETS:
                        next_point = (current[0] + dx2, current[1] + dy2, current[2] + dz2)
                        
                        if not (0 <= next_point[0] < shape[0] and
                                0 <= next_point[1] < shape[1] and
                                0 <= next_point[2] < shape[2]):
                            continue
                            
                        if (next_point in skel_points and 
                            next_point != prev and 
                            next_point not in branch):
                            next_points.append(next_point)
                    
                    if len(next_points) != 1:
                        break
                        
                    prev = current
                    current = next_points[0]
                
                if current in start_points:
                    branch.append(current)
                    branches.append(branch)
                    visited.update(branch)
    
    return branches

def is_legitimate_connection_optimized(branch, closed_obj, skel_vol, distance_map):

    branch_points = np.array(branch)

    widths = distance_map[branch_points[:, 0], branch_points[:, 1], branch_points[:, 2]] * 2
    min_width = np.min(widths)
    
    # use min width threshold
    if min_width >= 3:
        return True
    
    # deviation threshold
    if not follows_natural_path_optimized(branch, closed_obj):
        return False
    
    # critical connection check
    if is_critical_connection_optimized(branch, skel_vol):
        return True
    
    return False

def follows_natural_path_optimized(branch, closed_obj):
    """
    Optimized version to check if the branch follows a natural path.
    """
    if len(branch) < 2:
        return False
        
    start, end = np.array(branch[0]), np.array(branch[-1])
    straight_distance = np.linalg.norm(end - start)
    
    path_length = len(branch) - 1 
    
    if path_length > straight_distance * 1.5:
        return False
    
    cache_key = hash(closed_obj.tobytes())
    if cache_key in distance_cache:
        distance_map = distance_cache[cache_key]
    else:
        distance_map = distance_transform_edt(closed_obj)
        distance_cache[cache_key] = distance_map
    
    branch_points = np.array(branch)
    distances = distance_map[branch_points[:, 0], branch_points[:, 1], branch_points[:, 2]]
    avg_distance = np.mean(distances)
    
    if avg_distance < 1.0:
        return False
    
    return True

def is_critical_connection_optimized(branch, skel_vol):
    temp_skel = skel_vol.copy()
    for point in branch[1:-1]:
        temp_skel[point] = False
    
    start, end = branch[0], branch[-1]
    
    visited = {start}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        if current == end:
            return False
        
        for dx, dy, dz in OFFSETS:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            
            if not (0 <= neighbor[0] < temp_skel.shape[0] and
                    0 <= neighbor[1] < temp_skel.shape[1] and
                    0 <= neighbor[2] < temp_skel.shape[2]):
                continue
                
            if temp_skel[neighbor] and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return True

def prune_radius_junction_branches(skel_vol, prune_dist_th):
    shape = skel_vol.shape
    
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    
    neighbor_count = np.zeros_like(skel_vol, dtype=int)
    convolve(skel_vol.astype(int), kernel, output=neighbor_count, mode="constant", cval=0)
    
    junctions = {tuple(c) for c in np.argwhere(skel_vol & (neighbor_count > 2))}
    endpoints = [tuple(c) for c in np.argwhere(skel_vol & (neighbor_count == 1))]
    if not junctions or not endpoints:
        return skel_vol.copy()
    
    pruned_vol = skel_vol.copy()

    for ep in endpoints:
        dists = [np.linalg.norm(np.array(ep) - np.array(j)) for j in junctions]
        if min(dists) < prune_dist_th:
            # prune branch from endpoint to nearest junction
            # find that junction
            junc = list(junctions)[int(np.argmin(dists))]
            # walk from endpoint to junction
            path = []
            cur = ep
            prev = None
            while True:
                path.append(cur)
                if cur == junc:
                    break
                # find neighbors WITH BOUNDS CHECKING
                neighs = []
                for off in OFFSETS:
                    nbr = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
                    if (0 <= nbr[0] < shape[0] and 
                        0 <= nbr[1] < shape[1] and 
                        0 <= nbr[2] < shape[2]):
                        if pruned_vol[nbr] and nbr != prev:
                            neighs.append(nbr)
                if not neighs:
                    break
                prev = cur
                cur = neighs[0]
            # remove path excluding junction
            for v in path[:-1]:
                pruned_vol[v] = False
    return pruned_vol

def plot_decomposed_objects(objs, skels, title="Object"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    for idx, (obj, skel) in enumerate(zip(objs, skels)):
        mask_color = cmap(idx % 20)
        skel_color = mask_color[:3]

        mask_coords = np.argwhere(obj)
        if mask_coords.size:
            if mask_coords.shape[1] == 2:
                x = mask_coords[:, 1]
                y = mask_coords[:, 0]
                z = np.zeros(len(mask_coords))
            else:
                x = mask_coords[:, 2]
                y = mask_coords[:, 1]
                z = mask_coords[:, 0]
            ax.scatter(x, y, z, s=1, color=[mask_color], alpha=0.25)

        skel = np.asarray(skel)
        if skel.dtype == object:
            skel = np.vstack(skel)
        if skel.size == 0:
            continue

        if skel.shape[1] == 2:
            x = skel[:, 1]
            y = skel[:, 0]
            z = np.zeros(len(skel))
        else:
            x = skel[:, 2]
            y = skel[:, 1]
            z = skel[:, 0]

        ax.plot(x, y, z, color=skel_color, linewidth=2)
        ax.scatter(x, y, z, s=4, color=skel_color, alpha=1.0)

    ax.set_title(title)
    plt.close(fig)

def compute_node_degree(skel_vol, junc, terminal_set, offsets, cluster_map, current_cluster_label):
    cluster_set = {voxel for voxel, lab in cluster_map.items() if lab == current_cluster_label}
    terminal_hits = set()
    shape = skel_vol.shape

    for v in cluster_set:
        # try every direction
        for off in offsets:
            nb = (v[0] + off[0], v[1] + off[1], v[2] + off[2])
            # skip out-of-bounds or non-skeleton voxels
            if not (0 <= nb[0] < shape[0] and
                    0 <= nb[1] < shape[1] and
                    0 <= nb[2] < shape[2] and
                    skel_vol[nb]):
                continue
            # start only when stepping out of cluster
            if nb in cluster_set:
                continue

            prev, cur = v, nb
            visited = {prev, cur}
            while True:
                if cur in terminal_set:
                    # endpoint
                    if cur not in cluster_map:
                        terminal_hits.add(cur)
                    # junction in a different cluster
                    elif cluster_map[cur] != current_cluster_label:
                        terminal_hits.add(cur)
                    break

                # find next skeleton neighbor
                next_steps = []
                for off2 in offsets:
                    nbr = (cur[0] + off2[0], cur[1] + off2[1], cur[2] + off2[2])
                    if (0 <= nbr[0] < shape[0] and
                        0 <= nbr[1] < shape[1] and
                        0 <= nbr[2] < shape[2] and
                        skel_vol[nbr] and
                        nbr not in visited):
                        next_steps.append(nbr)

                if not next_steps:
                    break

                prev, cur = cur, next_steps[0]
                visited.add(cur)

    return len(terminal_hits)

def plot_decomposed_objects_2d(original_obj, closed_obj, skel_coords, dec_objs, prune_dist_th, title="Object 2D", save_path=None):

    obj_2d = np.any(original_obj, axis=0)
    if np.any(obj_2d):
        coords = np.argwhere(obj_2d)
        min_row, min_col = coords.min(axis=0)
        max_row, max_col = coords.max(axis=0)
        pad = 5
        min_row = max(0, min_row - pad)
        min_col = max(0, min_col - pad)
        max_row = min(obj_2d.shape[0]-1, max_row + pad)
        max_col = min(obj_2d.shape[1]-1, max_col + pad)
    else:
        min_row, max_row = 0, obj_2d.shape[0]-1
        min_col, max_col = 0, obj_2d.shape[1]-1

    max_degree = 0
    fig = None

    degree_counts = {i: 0 for i in range(11)}
    five_plus_positions = []

    volume_voxels = np.count_nonzero(original_obj)

    original_max = np.max(original_obj, axis=0)
    closed_max = np.max(closed_obj, axis=0)
    skel_vol = np.zeros_like(original_obj, dtype=bool)

    if skel_coords:
        try:
            skel_arr = np.concatenate([arr for arr in skel_coords if arr.size > 0], axis=0).astype(int)
        except ValueError:
            skel_arr = np.zeros((0, 3), dtype=int)
    else:
        skel_arr = np.zeros((0, 3), dtype=int)  
    skel_vol[skel_arr[:, 0], skel_arr[:, 1], skel_arr[:, 2]] = True
    skel_vol = prune_radius_junction_branches(skel_vol, prune_dist_th)
    skel_vol = analyze_branch_characteristics(skel_vol, closed_obj)
    proj = np.max(skel_vol, axis=0)

    kernel = np.ones((3, 3, 3), dtype=int); kernel[1, 1, 1] = 0
    
    neighbor_count = np.zeros_like(skel_vol, dtype=int)
    convolve(skel_vol.astype(int), kernel, output=neighbor_count, mode="constant", cval=0)

    junction_mask = skel_vol & (neighbor_count > 2)
    structure = np.ones((3, 3, 3), dtype=int)
    junction_labels, n_junc = cc_label(junction_mask, structure=structure)
    junc_points = np.argwhere(junction_labels > 0)
    cluster_map = {}
    if junc_points.size > 0:
        labels = junction_labels[tuple(junc_points.T)]
        cluster_map = {tuple(c): lab for c, lab in zip(junc_points, labels)}

    # compute centroids
    junction_centroids = []
    if n_junc > 0:
        all_labels = junction_labels[junc_points[:,0], junc_points[:,1], junc_points[:,2]]
        for lab in range(1, n_junc+1):
            mask = (all_labels == lab)
            if not np.any(mask): continue
            coords = junc_points[mask]
            cent = tuple(np.round(coords.mean(axis=0)).astype(int))
            if cent in cluster_map and cluster_map[cent] == lab:
                junction_centroids.append((cent, lab))
            else:
                dists = np.sum((coords-cent)**2, axis=1)
                closest = tuple(coords[np.argmin(dists)])
                junction_centroids.append((closest, lab))
    
    # create cluster sets
    cluster_sets = {}
    for voxel, label in cluster_map.items():
        if label not in cluster_sets:
            cluster_sets[label] = set()
        cluster_sets[label].add(voxel)
    
    # merge nearby and connected junctions
    if junction_centroids:
        junction_centroids, cluster_map = merge_junction_clusters(
            junction_centroids,
            cluster_map,
            skel_vol,
            max_distance=12.0
        )

    endpoint_coords = [tuple(c) for c in np.argwhere(skel_vol & (neighbor_count == 1))]
    terminal_set = set(cluster_map.keys()) | set(endpoint_coords)
    offsets = [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)!=(0,0,0)]

    centroid = np.mean(np.argwhere(original_obj), axis=0)
    centroid_z, centroid_y, centroid_x = centroid
    centroid_text = f'Centroid: ({centroid_x:.1f}, {centroid_y:.1f}, {centroid_z:.1f})'

    if PLOT_FIGURES == True:
        fig, axes = plt.subplots(1,4,figsize=(16,4))
        axes[0].imshow(original_max, cmap="gray"); axes[0].set_title("original mask"); axes[0].axis("off")

        axes[0].imshow(original_max, cmap="gray")
        axes[0].text(0.02, 0.98, centroid_text, transform=axes[0].transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[0].set_title("original mask"); axes[0].axis("off")

        topmost_z = np.argmax(closed_obj, axis=0).astype(float)
        mask = np.any(closed_obj, axis=0)
        topmost_z[~mask] = np.nan
        
        depth_cmap = plt.cm.viridis.copy()
        depth_cmap.set_bad('black')
        
        axes[1].imshow(topmost_z, cmap=depth_cmap, vmin=0, vmax=closed_obj.shape[0]-1)
        axes[1].set_title("closed mask (depth)"); axes[1].axis("off")

        axes[2].imshow(proj, cmap="autumn", alpha=0.7)
        if endpoint_coords:
            pts = np.array([(c[2], c[1]) for c in endpoint_coords])
            axes[2].scatter(pts[:,0],pts[:,1],c="red",s=10,label="endpoint")
    if junction_centroids:
        pts_j = np.array([(c[2], c[1]) for c, _ in junction_centroids])
        junction_data = []
        if PLOT_FIGURES:
            axes[2].scatter(pts_j[:, 0], pts_j[:, 1], c="blue", s=25, label="junction")
        for junc_coord, cluster_label in junction_centroids:
            y2d, x2d = junc_coord[2], junc_coord[1]
            col = junc_coord[2]
            row = junc_coord[1]
            deg = compute_node_degree(
                skel_vol,
                junc_coord,
                terminal_set,
                offsets,
                cluster_map,
                cluster_label
            )
            bin_index = min(deg, 10)
            degree_counts[bin_index] += 1

            if deg > max_degree:
                max_degree = deg
            if deg >= 6:
                five_plus_positions.append((y2d, x2d))
                if PLOT_FIGURES == True: 
                    axes[2].scatter(y2d, x2d, c="green", s=25, label="junction")
            elif deg in degree_counts:
                degree_counts[deg] += 1
            if PLOT_FIGURES == True:
                axes[2].text(y2d, x2d, str(deg),color="yellow",fontsize=6,ha='center',va='center')

    if PLOT_FIGURES == True:
        axes[2].set_title("skeleton (pruned)"); axes[2].axis("off")
        for ax in axes:
            ax.set_xlim(min_col, max_col)
            ax.set_ylim(max_row, min_row)

    if RECORD_PLOTS and save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return None, five_plus_positions, degree_counts, max_degree
    
    plt.close(fig)
    
    return fig, five_plus_positions, degree_counts, max_degree

def crop_object_tight(obj, padding=3):
    """Optimized cropping with minimal padding"""
    coords = np.argwhere(obj)
    if len(coords) == 0:
        return obj, None
    
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    
    slices = tuple(slice(max(0, m-padding), min(s, M+padding+1)) 
                   for m, M, s in zip(mins, maxs, obj.shape))
    
    return obj[slices], slices

def optimized_binary_closing(obj, radius=3):
    """Faster binary closing using OpenCV when possible"""
    if np.count_nonzero(obj) < 500000:
        result = np.zeros_like(obj)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
        
        for z in range(obj.shape[0]):
            if np.any(obj[z]):
                result[z] = cv2.morphologyEx(obj[z].astype(np.uint8), 
                                           cv2.MORPH_CLOSE, kernel).astype(bool)
        return result
    else:
        return binary_closing(obj, ball(radius))

def optimized_skeletonize(binary_image):
    """Faster skeletonization for 3D objects using 2D thinning slice-by-slice"""
    skeleton = np.zeros_like(binary_image)
    for z in range(binary_image.shape[0]):
        if np.any(binary_image[z]):
            skeleton[z] = thin(binary_image[z])
    return skeleton

def plot_all_objects_in_volume(volume, filename, save_dir, frame_idx):
    """
    Create a comprehensive plot of all objects in a cropped region, including:
    - Original mask
    - Closed mask with depth coloring
    - Skeleton with junctions and endpoints
    - Node degree annotations
    - Print counts of each junction type
    """

    x_min, y_min = 310, 310
    square_size = 310
    x_max = x_min + square_size
    y_max = y_min + square_size
    
    cropped_volume = volume[:, y_min:y_max, x_min:x_max]
    
    binary_mask = cropped_volume > 0
    
    closed_mask = optimized_binary_closing(binary_mask, radius=3)
    
    skel_img = optimized_skeletonize(closed_mask)
    skel_coords = np.argwhere(skel_img)
    
    skel_vol = np.zeros_like(binary_mask, dtype=bool)
    if skel_coords.size > 0:
        skel_vol[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = True
    
    skel_vol = prune_radius_junction_branches(skel_vol, prune_dist_th=2)
    skel_vol = analyze_branch_characteristics(skel_vol, closed_mask)
    
    original_proj = np.max(binary_mask, axis=0)
    closed_proj = np.max(closed_mask, axis=0)
    skel_proj = np.max(skel_vol, axis=0)
    
    topmost_z = np.argmax(closed_mask, axis=0).astype(float)
    depth_mask = np.any(closed_mask, axis=0)
    topmost_z[~depth_mask] = np.nan
    
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0
    
    neighbor_count = np.zeros_like(skel_vol, dtype=int)
    convolve(skel_vol.astype(int), kernel, output=neighbor_count, mode="constant", cval=0)
    
    junction_mask = skel_vol & (neighbor_count > 2)
    structure = np.ones((3, 3, 3), dtype=int)
    junction_labels, n_junc = cc_label(junction_mask, structure=structure)
    
    cluster_map = {}
    if n_junc > 0:
        junc_points = np.argwhere(junction_labels > 0)
        labels = junction_labels[tuple(junc_points.T)]
        cluster_map = {tuple(c): lab for c, lab in zip(junc_points, labels)}
    
    junction_centroids = []
    if n_junc > 0:
        junc_points = np.argwhere(junction_labels > 0)
        all_labels = junction_labels[tuple(junc_points.T)]
        for lab in range(1, n_junc + 1):
            mask = (all_labels == lab)
            if not np.any(mask):
                continue
            coords = junc_points[mask]
            cent = tuple(np.round(coords.mean(axis=0)).astype(int))
            if cent in cluster_map and cluster_map[cent] == lab:
                junction_centroids.append((cent, lab))
            else:
                dists = np.sum((coords - cent) ** 2, axis=1)
                closest = tuple(coords[np.argmin(dists)])
                junction_centroids.append((closest, lab))
    
    if junction_centroids:
        junction_centroids, cluster_map = merge_junction_clusters(
            junction_centroids, cluster_map, skel_vol, max_distance=12.0
        )
    
    endpoint_coords = [tuple(c) for c in np.argwhere(skel_vol & (neighbor_count == 1))]
    terminal_set = set(cluster_map.keys()) | set(endpoint_coords)
    
    offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    degree_counts = {i: 0 for i in range(11)}
    five_plus_positions = []
    
    for junc_coord, cluster_label in junction_centroids:
        deg = compute_node_degree(
            skel_vol, junc_coord, terminal_set, offsets, cluster_map, cluster_label
        )
        bin_index = min(deg, 10)
        degree_counts[bin_index] += 1
        
        if deg >= 6:
            five_plus_positions.append((junc_coord[2], junc_coord[1]))
    
    print(f"\nJunction Type Counts for {filename} - Frame {frame_idx}:")
    print("Region: ({}, {}) to ({}, {})".format(x_min, y_min, x_max, y_max))
    print("-" * 40)
    total_junctions = sum(degree_counts.values())
    print(f"Total Junctions: {total_junctions}")
    
    for degree in range(11):
        count = degree_counts[degree]
        if degree == 10:
            label = "10+"
        else:
            label = str(degree)
        
        percentage = (count / total_junctions * 100) if total_junctions > 0 else 0
        print(f"Degree {label}: {count} ({percentage:.1f}%)")
    
    depth_cmap = plt.cm.viridis.copy()
    depth_cmap.set_bad('black')
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original_proj, cmap="gray")
    axes[0].set_title("Original Mask")
    axes[0].axis("off")
    
    axes[1].imshow(topmost_z, cmap=depth_cmap, vmin=0, vmax=closed_mask.shape[0] - 1)
    axes[1].set_title("Closed Mask (Depth)")
    axes[1].axis("off")
    
    axes[2].imshow(skel_proj, cmap="autumn", alpha=0.7)
    
    if endpoint_coords:
        pts = np.array([(c[2], c[1]) for c in endpoint_coords])
        axes[2].scatter(pts[:, 0], pts[:, 1], c="red", s=10, label="Endpoint")
    
    if junction_centroids:
        pts_j = np.array([(c[2], c[1]) for c, _ in junction_centroids])
        axes[2].scatter(pts_j[:, 0], pts_j[:, 1], c="blue", s=25, label="Junction")
        
        for junc_coord, cluster_label in junction_centroids:
            y2d, x2d = junc_coord[2], junc_coord[1]
            deg = compute_node_degree(
                skel_vol, junc_coord, terminal_set, offsets, cluster_map, cluster_label
            )
            
            if deg >= 6:
                axes[2].scatter(y2d, x2d, c="green", s=25)
            
            axes[2].text(y2d, x2d, str(deg), color="yellow", fontsize=6, 
                        ha='center', va='center')
    
    axes[2].set_title("Skeleton with Junctions")
    axes[2].axis("off")
    
    label_proj = np.max(cropped_volume, axis=0)
    
    stats_text = "Junction Statistics:\n"
    stats_text += f"Total: {total_junctions}\n"
    for degree in range(11):
        if degree_counts[degree] > 0:
            if degree == 10:
                label = "10+"
            else:
                label = str(degree)
            stats_text += f"Deg {label}: {degree_counts[degree]}\n"
    
    plt.suptitle(f"{filename} - Frame {frame_idx}\nRegion: ({x_min},{y_min}) to ({x_max},{y_max})", 
                 fontsize=14)
    
    plt.tight_layout()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(
        save_dir,
        f"{filename}_frame_{frame_idx}_all_objects_detailed.png"
    )
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"Saved detailed combined plot to {save_path}")
    return save_path

def label_worker(args):
    file_path, labels, frame_idx, record_all, volume = args
    try:
        results = []
        global_skeletons = {}
        
        for lbl in labels:
            obj = volume == lbl
            if np.count_nonzero(obj) < MIN_OBJ_SIZE:
                continue
            
            obj_cropped, slices = crop_object_tight(obj, padding=2)
            closed = optimized_binary_closing(obj_cropped, radius=3)
            skel_img = optimized_skeletonize(closed)
            skel_coords = [np.argwhere(skel_img)] if np.any(skel_img) else []
            
            save_path = None
            if RECORD_PLOTS and (frame_idx % RECORD_INTERVAL == 0):
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                save_path = os.path.join(PLOTS_DIR, f"{base_name}_frame_{frame_idx}_label_{lbl}.png")
            
            _, positions, degree_counts, max_degree = plot_decomposed_objects_2d(
                obj_cropped, closed, skel_coords, None, prune_dist_th=2,
                title=f"{os.path.basename(file_path)} – label {lbl}",
                save_path=save_path
            )
            
            results.append((positions, degree_counts, max_degree))
            
            if record_all:
                if skel_coords and skel_coords[0].size > 0:
                    global_coords = skel_coords[0].copy()
                    for i in range(3):
                        global_coords[:, i] += slices[i].start
                    global_skeletons[lbl] = global_coords
            
        return results, global_skeletons
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], {}

def process_file(file_path, frame_idx):
    try:
        volume = tifffile.memmap(file_path, mode='r')
        
        labels = np.unique(volume)
        labels = labels[labels != 0]
        
        valid_labels = []
        for lbl in labels:
            if np.count_nonzero(volume == lbl) >= MIN_OBJ_SIZE:
                valid_labels.append(lbl)
                
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return

    if not valid_labels:
        return

    record_all = RECORD_ALL_OBJECTS_PLOT and (frame_idx % RECORD_INTERVAL == 0)

    chunk_size = max(1, min(50, len(valid_labels) // (MAX_WORKERS * 2)))
    label_chunks = [valid_labels[i:i + chunk_size] 
                   for i in range(0, len(valid_labels), chunk_size)]
    
    tasks = [(file_path, chunk, frame_idx, record_all, volume) for chunk in label_chunks]
    
    frame_max_deg_counts = {i: 0 for i in range(11)}
    frame_total_deg_counts = {i: 0 for i in range(11)}
    all_global_skeletons = {}
    
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(label_chunks))) as executor:
        for chunk_results in tqdm(executor.map(label_worker, tasks),
                          total=len(tasks),
                          desc=f"Processing {os.path.basename(file_path)}"):
            chunk_res, global_skels = chunk_results
            all_global_skeletons.update(global_skels)
            
            for obj_result in chunk_res:
                if obj_result:
                    positions, deg_counts, max_degree = obj_result

                    for deg, count in deg_counts.items():
                        bin_index = min(deg, 10)
                        frame_total_deg_counts[bin_index] += count

                    max_bin = min(max_degree, 10) if max_degree > 0 else 0
                    frame_max_deg_counts[max_bin] += 1
    
    if RECORD_NODE_DEGREE_CSV:
        with open(NODE_DEGREE_CSV, 'a') as f:
            row = [os.path.basename(file_path)]
            row.extend([frame_max_deg_counts[i] for i in range(11)])
            row.extend([frame_total_deg_counts[i] for i in range(11)])
            f.write(",".join(map(str, row)) + "\n")

    if record_all and all_global_skeletons:
        try:
            plot_all_objects_in_volume(
                volume, 
                os.path.basename(file_path),
                ALL_OBJECTS_PLOTS_DIR,
                frame_idx
            )
        except Exception as e:
            print(f"Error creating combined plot for {file_path}: {str(e)}")
    
    print(f"Processed {len(valid_labels)} objects in {os.path.basename(file_path)}")

def main():
    tif_files = [f for f in sorted(os.listdir(DATA_DIR))
                 if f.lower().endswith(('.tif', '.tiff'))]
    file_paths = [os.path.join(DATA_DIR, f) for f in tif_files]

    for frame_idx, fp in enumerate(tqdm(file_paths, desc="Processing files", unit="file")):
        if frame_idx % RECORD_INTERVAL == 0:
            print(f"--- Processing frame {frame_idx}: {os.path.basename(fp)} ---")
            process_file(fp, frame_idx)

if RECORD_NODE_DEGREE_CSV and not os.path.exists(NODE_DEGREE_CSV):
    with open(NODE_DEGREE_CSV, 'w') as f:
        header = "File," + ",".join([f"MaxDeg_{i}" for i in range(11)]) + "," + \
                 ",".join([f"TotalDeg_{i}" for i in range(11)])
        f.write(header + "\n")

if __name__ == "__main__":
    main()
