import cv2
import numpy as np
import matplotlib.pyplot as plt


#   IMAGE PATH

IMAGE_PATH = r"C:\Users\HP\Documents\Unmanned Devices and Processes\Task on Histogram\flower.jpg"

def _load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return image


#    CONVERT IMAGE TO GRAYSCALE

def to_grayscale(image):
    """
    Slide 19:  Y = 0.299*R + 0.587*G + 0.114*B
    """
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


#  STEP 1.   HISTOGRAM 

def histogram(image):
    """
    H(j) = count{x : I(x) = j}   for j = 0..255

    Returns
    
    hist : np.ndarray, int64, shape (256,)
    """
    hist = np.zeros(256, dtype=np.int64)
    for v in image.ravel():
        hist[v] += 1
    return hist


#  STEP 2. HISTOGRAM NORMALIZATION 

def histogram_normalization(hist, n):
    """
    p(i) = hist[i] / n      so that  sum_i p(i) = 1
    """
    return hist.astype(np.float64) / n


#  STEP 3. CUMULATIVE HISTOGRAM 

def cumulative_histogram_counts(hist):
    """
    Cumulative histogram in COUNTS:

        cdf_counts(i) = sum_{j=0}^{i} hist[j]
    """
    return np.cumsum(hist).astype(np.int64)


#  STEP 4.  EQUALIZATION 

def equalization(image):
    """
        pxl_new = ( cdf(pxl) - cdf_min ) / (n - cdf_min) * (L - 1)

    where:
        cdf(pxl) : cumulative COUNT up to that intensity
        cdf_min  : smallest non-zero value of the cdf
        n        : total number of pixels (H * W)
        L        : number of gray levels (256)

    Returns
    -------
    equalized   : np.ndarray  (uint8, same shape as input)
    hist        : np.ndarray  (256,) raw histogram
    pdf         : np.ndarray  (256,) normalized histogram
    cdf_counts  : np.ndarray  (256,) cumulative counts
    mapping     : np.ndarray  (256,) lookup table applied to the image
    """
    n = image.shape[0] * image.shape[1]
    L = 256

    # --- Step 1: Histogram ---
    hist = histogram(image)

    # --- Step 2: Normalized histogram ---
    pdf = histogram_normalization(hist, n)

    # --- Step 3: Cumulative histogram (COUNTS) ---
    cdf_counts = cumulative_histogram_counts(hist)

    # subtract cdf_min (first non-zero cdf value)
    nonzero = cdf_counts[cdf_counts > 0]
    cdf_min = nonzero[0] if nonzero.size else 0

    # Build the mapping for every intensity 0..255
    #   denominator n - cdf_min  (avoid division by zero on flat images)
    denom = max(n - cdf_min, 1)
    mapping = np.round(
        (cdf_counts - cdf_min) / denom * (L - 1)
    ).astype(np.int64)

    # Clip safety (should already be in [0, 255])
    mapping = np.clip(mapping, 0, L - 1).astype(np.uint8)

    # Apply the mapping to every pixel
    equalized = mapping[image]

    return equalized, hist, pdf, cdf_counts, mapping


#    MATPLOTLIB PLOTS

def plot_histogram(hist, title, color="tab:blue"):
    plt.figure()
    plt.plot(range(len(hist)), hist, color=color)
    plt.xlim(0, 255)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all(gray, equalized, hist, pdf, cdf_counts, hist_after):
    """
    2 x 3 grid of plots in the style of the lecture notes:
        (0,0) original grayscale
        (0,1) raw histogram          (counts)
        (0,2) normalized histogram   (probabilities)
        (1,0) cumulative histogram   (counts)
        (1,1) equalized image
        (1,2) equalized histogram
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Grayscale (Step 0)")
    axes[0, 0].axis("off")

    axes[0, 1].plot(range(256), hist, color="tab:blue")
    axes[0, 1].set_title("Histogram (Step 1)")
    axes[0, 1].set_xlim(0, 255)
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(range(256), pdf, color="tab:green")
    axes[0, 2].set_title("Normalized histogram (Step 2)")
    axes[0, 2].set_xlim(0, 255)
    axes[0, 2].set_xlabel("X")
    axes[0, 2].set_ylabel("probability")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(range(256), cdf_counts, color="tab:orange")
    axes[1, 0].set_title("Cumulative histogram (Step 3) — counts")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].imshow(equalized, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Equalized image (Step 4)")
    axes[1, 1].axis("off")

    axes[1, 2].plot(range(256), hist_after, color="tab:red")
    axes[1, 2].set_title("Histogram after equalization (Step 4)")
    axes[1, 2].set_xlim(0, 255)
    axes[1, 2].set_xlabel("X")
    axes[1, 2].set_ylabel("Y")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


#  MAIN 

def improve_image_quality(image_path, save_path=None):
    # Step 0 — grayscale
    color = _load_image(image_path)
    gray = to_grayscale(color)

    # Steps 1–4
    equalized, hist, pdf, cdf_counts, mapping = equalization(gray)
    hist_after = histogram(equalized)

    n = gray.shape[0] * gray.shape[1]
    nonzero = cdf_counts[cdf_counts > 0]
    cdf_min = nonzero[0] if nonzero.size else 0

    # ---- console report ----
    print("=== STEP 0 — Grayscale ===")
    print(f"  Color shape : {color.shape}")
    print(f"  Gray shape  : {gray.shape}")

    print("\n=== STEP 1 — Histogram (counts) ===")
    print(f"  total pixels : {n}")
    print(f"  min intensity: {np.min(np.where(hist > 0)[0])}")
    print(f"  max intensity: {np.max(np.where(hist > 0)[0])}")

    print("\n=== STEP 2 — Normalized histogram (pdf) ===")
    print(f"  sum(pdf) = {pdf.sum():.6f}")

    print("\n=== STEP 3 — Cumulative histogram (counts) ===")
    print(f"  cdf_min  = {cdf_min}   (first non-zero cdf value)")
    print(f"  cdf[128] = {cdf_counts[128]}")
    print(f"  cdf[255] = {cdf_counts[255]}  (== n)")

    print("\n=== STEP 4 — Equalization (slide-33 formula) ===")
    print("  Selected mapping values:")
    for v in (0, 32, 64, 128, 192, 255):
        print(f"    mapping[{v:3d}] = {mapping[v]}")
    print(f"  Output range : {np.min(np.where(hist_after>0)[0])} – "
          f"{np.max(np.where(hist_after>0)[0])}")

    # Save 
    if save_path:
        cv2.imwrite(save_path, equalized)
        print(f"\nSaved: {save_path}")

    # Plots
    plot_all(gray, equalized, hist, pdf, cdf_counts, hist_after)


#      RUN

if __name__ == "__main__":
    improve_image_quality(
        IMAGE_PATH,
        save_path=r"C:\Users\HP\Documents\Unmanned Devices and Processes\Task on Histogram\flower_equalized.jpg",
    )