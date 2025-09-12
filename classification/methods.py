from skimage.filters import apply_hysteresis_threshold, gaussian
from skimage.util import invert
from skimage.filters import threshold_multiotsu

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage.morphology import white_tophat as sk_white_tophat
from skimage.morphology import ball, binary_closing
from skimage.filters import frangi, threshold_otsu
import cv2

LABELED_DATA_DIR = "exp4/thresholdedData/labeled_volumes"
RAW_DATA_DIR = "exp4/lastFrame"
OUTPUT_DIR = "classification_comparison"

PLOT_DPI = 300

def rescale_to_uint8(volume, p_low=1.0, p_high=99.9):
    v = volume.astype(np.float32, copy=False)
    lo = np.percentile(v, p_low)
    hi = np.percentile(v, p_high)
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.uint8)
    v = np.clip((v - lo) / (hi - lo), 0, 1)
    return (v * 255).astype(np.uint8)

def anisotropic_filtered_uint8(raw_volume, alpha=0.05, K=0.05, niters=5):
    vol8 = rescale_to_uint8(raw_volume)
    filtered = np.empty_like(vol8, dtype=np.uint8)
    use_xproc = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "anisotropicDiffusion")

    for z in range(vol8.shape[0]):
        gray = vol8[z]
        if use_xproc:
            try:
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                diff = cv2.ximgproc.anisotropicDiffusion(bgr, alpha=alpha, K=K, niters=niters)
                out = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                out = cv2.bilateralFilter(gray, 5, 50, 50)
        else:
            out = cv2.bilateralFilter(gray, 5, 50, 50)
        filtered[z] = out

    return filtered

def tophat_filtered_uint8(raw_volume, radius=3):
    vol8 = rescale_to_uint8(raw_volume)
    th = sk_white_tophat(vol8, footprint=ball(radius))
    return th.astype(np.uint8)

def global_otsu_on_tophat(raw_volume, sigma=1.0, use_multi=False, multi_classes=3, bias=1.0):
    filtered = tophat_filtered_uint8(raw_volume)
    if sigma and (np.any(np.array(sigma) > 0)):
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)

    def _biased_thr(img):
        t = (threshold_multiotsu(img, classes=multi_classes)[0] if use_multi
             else threshold_otsu(img))
        t = t * bias
        return np.clip(t, 0, 255)

    t = _biased_thr(filtered)
    return (filtered >= t)


def adaptive_threshold_on_tophat(raw_volume, block_size=31, C=8, sigma=1.0):
    filtered = tophat_filtered_uint8(raw_volume)
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)

    binary_volume = np.zeros(filtered.shape, dtype=bool)
    for z in range(filtered.shape[0]):
        img = filtered[z]
        bs = block_size
        bs = min(bs, min(img.shape[:2]) - 1)
        bs = bs if bs % 2 == 1 else bs - 1
        if bs < 3:
            thr = threshold_otsu(img)
            binary = img > thr
        else:
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, bs, C
            ) > 0
        binary_volume[z] = binary
    return binary_volume


def hysteresis_threshold_on_tophat(raw_volume, low_factor=1.0, high_factor=1.6, sigma=1.0):
    from skimage.morphology import reconstruction

    filtered = tophat_filtered_uint8(raw_volume)
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)

    thr_otsu = threshold_otsu(filtered)
    thr_low  = low_factor  * thr_otsu
    thr_high = high_factor * thr_otsu

    low_mask  = filtered >= thr_low
    high_mask = filtered >= thr_high

    binary_volume = reconstruction(
        high_mask, low_mask, method='dilation', footprint=ball(1)
    )
    return binary_volume


def global_otsu_on_filtered(raw_volume, sigma=1.0, use_multi=False, multi_classes=3, bias=1.0):
    """
    Method: Global Otsu on Filtered Volume
    Parameters:
      sigma: Gaussian smoothing kernel (0 disables smoothing)
    """
    filtered = anisotropic_filtered_uint8(raw_volume)

    if sigma and (np.any(np.array(sigma) > 0)):
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)

    def _biased_thr(img):
        t = (threshold_multiotsu(img, classes=multi_classes)[0] if use_multi
             else threshold_otsu(img))
        t = t * bias
        return np.clip(t, 0, 255)

    t = _biased_thr(filtered)
    return (filtered >= t)

def adaptive_threshold_on_filtered(raw_volume, block_size=31, C=8, sigma=1.0):
    """
    Method: Adaptive Threshold on Filtered Volume
    Parameters:
      block_size: Local neighborhood size (must be odd)
      C: Constant subtracted from mean
      sigma: Gaussian smoothing kernel
    """
    filtered = anisotropic_filtered_uint8(raw_volume)
    
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    binary_volume = np.zeros(filtered.shape, dtype=bool)
    for z in range(filtered.shape[0]):
        img = filtered[z]
        bs = block_size

        bs = min(bs, min(img.shape[:2])-1) 
        bs = bs if bs % 2 == 1 else bs - 1
        if bs < 3:
            thr = threshold_otsu(img)
            binary = img > thr
        else:
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, bs, C
            ) > 0
        binary_volume[z] = binary
    return binary_volume

def hysteresis_threshold_on_filtered(raw_volume, low_factor=0.4, high_factor=0.8, sigma=1.0):
    """
    Method: Hysteresis Thresholding on Filtered Volume
    Parameters:
      low_factor: Fraction of Otsu for low threshold
      high_factor: Fraction of Otsu for high threshold
      sigma: Gaussian smoothing kernel
    """
    filtered = anisotropic_filtered_uint8(raw_volume, alpha=0.1, K=0.1, niters=10)
    
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    thr_high = threshold_otsu(filtered)
    thr_low = low_factor * thr_high
    thr_high = high_factor * thr_high

    low_mask = filtered >= thr_low
    high_mask = filtered >= thr_high
    
    from skimage.morphology import reconstruction
    binary_volume = reconstruction(
        high_mask, low_mask, 
        method='dilation', 
        footprint=ball(1)
    )
    return binary_volume

def background_subtract_uint8(raw_volume,
                              mode="rowmed",
                              sigma_xy=15,         # for mode="gaussian" (pixels)
                              rolling_radius=20,   # for mode="rolling_ball" (voxels)
                              profile_sigma=3,     # smooth the row/col median profile
                              clamp_min=0, clamp_max=255):
    """
    Background subtraction on a 3D stack (Z,Y,X), returning uint8.

    modes:
      - "rowmed": subtract per-row (along X) medians per slice (good for horizontal banding)
      - "colmed": subtract per-column (along Y) medians per slice (good for vertical banding)
      - "gaussian": subtract a large 2D Gaussian blur from each slice
      - "rolling_ball": subtract morphological opening via a 3D ball (like rolling-ball)

    Tips:
      - If stripes are HORIZONTAL lines (vary across columns, constant along a row),
        use mode="rowmed".
      - If stripes are VERTICAL lines (vary down rows, constant along a column),
        use mode="colmed".
    """
    vol8 = rescale_to_uint8(raw_volume).astype(np.float32)
    out = np.empty_like(vol8, dtype=np.float32)

    if mode in ("rowmed", "colmed"):
        for z in range(vol8.shape[0]):
            img = vol8[z]
            if mode == "rowmed":
                prof = np.median(img, axis=1).astype(np.float32)   # shape (H,)
                if profile_sigma and profile_sigma > 0:
                    prof = cv2.GaussianBlur(prof[:, None], (0, 0), profile_sigma,
                                            borderType=cv2.BORDER_REFLECT).squeeze()
                bg = np.tile(prof[:, None], (1, img.shape[1]))
            else:
                prof = np.median(img, axis=0).astype(np.float32)
                if profile_sigma and profile_sigma > 0:
                    prof = cv2.GaussianBlur(prof[None, :], (0, 0), profile_sigma,
                                            borderType=cv2.BORDER_REFLECT).squeeze()
                bg = np.tile(prof[None, :], (img.shape[0], 1))
            out[z] = img - bg
    elif mode == "gaussian":
        # 2D blur per slice to avoid leaking across Z
        blurred = gaussian(vol8, sigma=(0, sigma_xy, sigma_xy), preserve_range=True)
        out = vol8 - blurred
    elif mode == "rolling_ball":
        # opening is the background; subtracting it is equivalent to white tophat
        opened = sk_white_tophat(vol8.astype(np.uint8), footprint=ball(rolling_radius))
        # white_tophat already returns (img - opening), so just pass it through
        out = opened.astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    out = np.clip(out, clamp_min, clamp_max).astype(np.uint8)
    return out


def otsu_per_z_layer(raw_volume, sigma=0, bias=1.0, **kwargs):
    filtered = anisotropic_filtered_uint8(raw_volume)
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    binary_volume = np.zeros(filtered.shape, dtype=bool)
    for z in range(filtered.shape[0]):
        thr = threshold_otsu(filtered[z]) * bias
        thr = np.clip(thr, 0, 255)
        binary_volume[z] = filtered[z] >= thr
    return binary_volume

def otsu_binned_x(raw_volume, sigma=0, bias=1.0, n_bins=100, **kwargs):
    filtered = anisotropic_filtered_uint8(raw_volume)
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    binary_volume = np.zeros(filtered.shape, dtype=bool)
    Z, Y, X = filtered.shape
    bin_size = max(1, X // n_bins)
    for z in range(Z):
        for i in range(n_bins):
            x_start = i * bin_size
            x_end = min((i+1) * bin_size, X)
            strip = filtered[z, :, x_start:x_end]
            if strip.size > 0:
                thr = threshold_otsu(strip) * bias
                thr = np.clip(thr, 0, 255)
                binary_volume[z, :, x_start:x_end] = strip >= thr
    return binary_volume

def otsu_binned_y(raw_volume, sigma=0, bias=1.0, n_bins=100, **kwargs):
    filtered = anisotropic_filtered_uint8(raw_volume)
    if sigma > 0:
        filtered = gaussian(filtered, sigma=sigma, preserve_range=True).astype(np.uint8)
    
    binary_volume = np.zeros(filtered.shape, dtype=bool)
    Z, Y, X = filtered.shape
    bin_size = max(1, Y // n_bins)
    for z in range(Z):
        for i in range(n_bins):
            y_start = i * bin_size
            y_end = min((i+1) * bin_size, Y)
            strip = filtered[z, y_start:y_end, :]
            if strip.size > 0:
                thr = threshold_otsu(strip) * bias
                thr = np.clip(thr, 0, 255)
                binary_volume[z, y_start:y_end, :] = strip >= thr
    return binary_volume

def generate_classified_volumes(raw_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.tif', '.tiff'))
    ])
    
    print(f"Found {len(raw_files)} raw TIFF files in {raw_dir}")
    
    for i, filename in enumerate(raw_files):
        print(f"Processing frame {i+1}/{len(raw_files)}: {filename}")
        raw_volume = tifffile.imread(os.path.join(raw_dir, filename))
        
        pre = background_subtract_uint8(
            raw_volume, 
            mode="rowmed", 
            profile_sigma=5,
            clamp_min=0,
            clamp_max=255
        )
        
        # hand picked C values based on frame index
        if i < 25:
            C_val = -9
        elif i < 50:
            C_val = -5
        elif i < 100:
            C_val = -3
        else:
            C_val = -1
        
        bin_volume = adaptive_threshold_on_filtered(
            pre, 
            block_size=71, 
            C=C_val, 
            sigma=3
        )
        
        bin_volume_uint8 = bin_volume.astype(np.uint8)
        out_path = os.path.join(output_dir, filename)
        tifffile.imwrite(out_path, bin_volume_uint8, compression='zlib')
    
    print(f"Saved {len(raw_files)} binary masks to {output_dir}")

def plot_anisotropic_triplet(raw_volume, labeled_volume, filename, currentInt=0):
    raw_u8 = rescale_to_uint8(raw_volume)
    raw_mip = np.max(raw_u8, axis=0)

    existing_mask = (labeled_volume > 0)
    existing_mip = np.max(existing_mask, axis=0).astype(bool)

    pre = background_subtract_uint8(raw_volume, mode="rowmed", profile_sigma=5)

    filt_u8 = anisotropic_filtered_uint8(pre)
    filt_mip = np.max(filt_u8, axis=0)

    sigma = 2
    C = -1
    if currentInt <= 25: 
        C = -9
    elif currentInt <= 50: 
        C = -5
    elif currentInt <= 100:
        C = -3

    bin1 = global_otsu_on_filtered(pre, sigma=sigma, bias=.92, use_multi=False, multi_classes=2)
    bin2 = adaptive_threshold_on_filtered(pre, block_size=71, C=C, sigma=sigma)
    bin3 = hysteresis_threshold_on_filtered(pre, low_factor=1.0, high_factor=1.6, sigma=sigma)
    
    bin4 = otsu_per_z_layer(pre, sigma=sigma, bias=0.92)
    bin5 = otsu_binned_x(pre, sigma=sigma, n_bins=200, bias=0.92)
    bin6 = otsu_binned_y(pre, sigma=sigma, n_bins=200, bias=0.92)
    
    fig, axs = plt.subplots(2, 6, figsize=(48, 16))
    fig.suptitle(f"Anisotropic Debug (Extended): {filename}", fontsize=20)
    
    titles_row0 = [
        "Raw (MIP)",
        "Anisotropic Filtered (MIP)",
        f"Existing Classification\nFG: {100*existing_mask.mean():.1f}%",
        f"Global Otsu\nFG: {100*bin1.mean():.1f}%",
        f"Adaptive (bs=71, C={C})\nFG: {100*bin2.mean():.1f}%",
        f"Hysteresis\nFG: {100*bin3.mean():.1f}%"
    ]
    
    axs[0, 0].imshow(raw_mip, cmap='gray')
    axs[0, 0].set_title(titles_row0[0])
    
    axs[0, 1].imshow(filt_mip, cmap='gray')
    axs[0, 1].set_title(titles_row0[1])
    
    axs[0, 2].imshow(raw_mip, cmap='gray')
    overlay_existing = np.ma.masked_where(~existing_mip, existing_mip)
    axs[0, 2].imshow(overlay_existing, cmap='autumn', alpha=0.6)
    axs[0, 2].set_title(titles_row0[2])
    
    for col in range(3, 6):
        bin_vol = [bin1, bin2, bin3][col-3]
        bin_mip = np.max(bin_vol, axis=0).astype(bool)
        axs[0, col].imshow(filt_mip, cmap='gray')
        overlay = np.ma.masked_where(~bin_mip, bin_mip)
        axs[0, col].imshow(overlay, cmap='autumn', alpha=0.6)
        axs[0, col].set_title(titles_row0[col])
    
    titles_row1 = [
        f"Per-Z Otsu\nFG: {100*bin4.mean():.1f}%",
        f"Binned-X Otsu\nFG: {100*bin5.mean():.1f}%",
        f"Binned-Y Otsu\nFG: {100*bin6.mean():.1f}%",
        "", "", ""
    ]
    
    for col in range(3):
        bin_vol = [bin4, bin5, bin6][col]
        bin_mip = np.max(bin_vol, axis=0).astype(bool)
        axs[1, col].imshow(filt_mip, cmap='gray')
        overlay = np.ma.masked_where(~bin_mip, bin_mip)
        axs[1, col].imshow(overlay, cmap='autumn', alpha=0.6)
        axs[1, col].set_title(titles_row1[col])
    
    for col in range(3, 6):
        axs[1, col].axis('off')
    
    for row in range(2):
        for col in range(6):
            axs[row, col].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, f"{filename}_aniso_extended.png")
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved anisotropic extended plot to: {out_path}")
    return out_path

def main():
    labeled_files = sorted(
        f for f in os.listdir(LABELED_DATA_DIR)
        if f.lower().endswith(('.tif', '.tiff'))
    )
    raw_files = sorted(
        f for f in os.listdir(RAW_DATA_DIR)
        if f.lower().endswith(('.tif', '.tiff'))
    )

    print("Found labeled:  ", len(labeled_files), "files")
    print("Found raw:      ", len(raw_files),   "files")

    matched_pairs = list(zip(labeled_files, raw_files))
    print("Matched pairs:  ", len(matched_pairs))

    i=0

    for lfile, rfile in matched_pairs:
        i+=1
        #if i%10 != 0: continue
        print(f"Processing L:{lfile}  ↔  R:{rfile}")
        labeled_path = os.path.join(LABELED_DATA_DIR, lfile)    
        raw_path     = os.path.join(RAW_DATA_DIR,     rfile)

        labeled_volume = tifffile.imread(labeled_path)
        raw_volume     = tifffile.imread(raw_path)

        #methods = {
        #    "Existing Classification": load_existing_classification,
        #    "Anisotropic Diffusion":    anisotropic_diffusion_adaptive,
        #    "Frangi Filter":            frangi_filter_adaptive,
        #    "Top Hat":                  top_hat_adaptive
        #}

        #plot_comparison(
        #    raw_volume,
        #    labeled_volume,
        #    methods,
        #    os.path.splitext(lfile)[0]
        #)
        plot_anisotropic_triplet(
            raw_volume,
            labeled_volume,
            os.path.splitext(lfile)[0],
            currentInt=i,
        )

        generate_classified_volumes(
            raw_dir="exp4/tifFormat",          # path to raw TIFFs
            output_dir="exp4/classified"         # output directory
        )


if __name__ == "__main__":
    main()
