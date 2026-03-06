import cv2
import numpy as np
import os
import csv
from skimage.morphology import medial_axis
import tqdm


# ======================== Functional Parts (Processing) ========================

def thinning_fallback(img):
    """Iterative morphological skeletonization (fallback)."""
    img = img.copy()
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def find_lines(image, method='medial', kernel_size=(3, 3), morph_iterations=1, preprocessed=False):
    """
    Compute the skeleton of the image using the specified method.
    (Medial axis is used by default.)
    """
    if not preprocessed:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = image.copy()
    kernel = np.ones(kernel_size, np.uint8)
    binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    if method == 'medial':
        skeleton = medial_axis(binary_clean.astype(bool))
        skeleton = (skeleton * 255).astype(np.uint8)
    elif method == 'thinning':
        try:
            skeleton = cv2.ximgproc.thinning(binary_clean)
        except AttributeError:
            skeleton = thinning_fallback(binary_clean)
    else:
        raise ValueError("Unknown method: {}. Choose 'medial' or 'thinning'.".format(method))
    return skeleton


def split_lines_from_skeleton(skel, min_length_threshold=30, branch_proximity=5):
    """
    Split the skeleton into segments by removing branch points and discard segments
    with arc length below min_length_threshold.
    """
    skel_bin = (skel > 0).astype(np.uint8)
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_bin, -1, neighbor_kernel)
    branch_mask = ((skel_bin == 1) & (neighbor_count >= 3)).astype(np.uint8) * 255
    skel_split = skel.copy()
    skel_split[branch_mask == 255] = 0
    segments, _ = cv2.findContours(skel_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filtered_segments = [cnt for cnt in segments if cv2.arcLength(cnt, False) >= min_length_threshold]
    return filtered_segments, branch_mask


def generate_holes_grid(image_shape, fragile_mask, hole_min_distance, hole_gridsearch_distance,
                        fragile_clearance, noise_std, safety_factor=3):
    """
    Generate hole positions via grid sampling.
    """
    h, w = image_shape
    grid_spacing = hole_gridsearch_distance
    x_coords = np.arange(grid_spacing / 2, w, grid_spacing)
    y_coords = np.arange(grid_spacing / 2, h, grid_spacing)
    
    candidate_holes = []
    inverted_mask = 255 - fragile_mask
    dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    clearance_mask = (dist_transform < fragile_clearance).astype(np.uint8) * 255
    
    for y in y_coords:
        for x in x_coords:
            perturbed_x = x + np.random.normal(0, noise_std)
            perturbed_y = y + np.random.normal(0, noise_std)
            candidate = (int(round(perturbed_x)), int(round(perturbed_y)))
            if candidate[0] < 0 or candidate[0] >= w or candidate[1] < 0 or candidate[1] >= h:
                continue
            if clearance_mask[candidate[1], candidate[0]] != 0:
                continue
            if any(np.linalg.norm(np.array(candidate) - np.array(existing)) < hole_min_distance
                   for existing in candidate_holes):
                continue
            candidate_holes.append(candidate)
    return candidate_holes

#취약영역 고려하지 않은 것 
def generate_holes_grid_2(image_shape, hole_min_distance, hole_gridsearch_distance,
                        noise_std):
    """
    Generate hole positions via grid sampling.
    """
    h, w = image_shape
    grid_spacing = hole_gridsearch_distance
    x_coords = np.arange(grid_spacing / 2, w, grid_spacing)
    y_coords = np.arange(grid_spacing / 2, h, grid_spacing)
    
    just_holes = []
    
    for y in y_coords:
        for x in x_coords:
            perturbed_x = x + np.random.normal(0, noise_std)
            perturbed_y = y + np.random.normal(0, noise_std)
            candidate = (int(round(perturbed_x)), int(round(perturbed_y)))
            if candidate[0] < 0 or candidate[0] >= w or candidate[1] < 0 or candidate[1] >= h:
                continue
            if any(np.linalg.norm(np.array(candidate) - np.array(existing)) < hole_min_distance
                   for existing in just_holes):
                continue
            just_holes.append(candidate)
    return just_holes

#노이즈 없는 격자 생성
def generate_holes_grid_3(image_shape, fragile_mask, hole_min_distance, hole_gridsearch_distance,
                        fragile_clearance, noise_std, safety_factor=3):
    """
    Generate hole positions via grid sampling.
    """
    h, w = image_shape
    grid_spacing = hole_gridsearch_distance
    x_coords = np.arange(grid_spacing / 2, w, grid_spacing)
    y_coords = np.arange(grid_spacing / 2, h, grid_spacing)
    
    candidate_holes = []
    inverted_mask = 255 - fragile_mask
    dist_transform = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, 5)
    clearance_mask = (dist_transform < fragile_clearance).astype(np.uint8) * 255
    
    for y in y_coords:
        for x in x_coords:
            perturbed_x = x 
            perturbed_y = y 
            candidate = (int(round(perturbed_x)), int(round(perturbed_y)))
            if candidate[0] < 0 or candidate[0] >= w or candidate[1] < 0 or candidate[1] >= h:
                continue
            if clearance_mask[candidate[1], candidate[0]] != 0:
                continue
            if any(np.linalg.norm(np.array(candidate) - np.array(existing)) < hole_min_distance
                   for existing in candidate_holes):
                continue
            candidate_holes.append(candidate)
    return candidate_holes

def draw_dashed_circle(img, center, radius, color, thickness=1, dash_length=5, gap_length=3):
    """
    Draw a dashed circle around a given point.
    """
    step_angle = (dash_length + gap_length) / float(radius)
    num_steps = int(2 * np.pi / step_angle)
    for i in range(num_steps):
        theta_start = i * step_angle
        theta_end = theta_start + (dash_length / float(radius))
        pt1 = (int(center[0] + radius * np.cos(theta_start)),
               int(center[1] + radius * np.sin(theta_start)))
        pt2 = (int(center[0] + radius * np.cos(theta_end)),
               int(center[1] + radius * np.sin(theta_end)))
        cv2.line(img, pt1, pt2, color, thickness)

# 취약영역 반영한 천공 위치 후보군군
def save_csv(holes, filename):
    """
    Save the hole coordinates to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for (x, y) in holes:
            writer.writerow([x, y])

# 취약영역 반영하지 않은 천고 위치 후보군
def save_csv_2(holes, filename):
    """
    Save the hole coordinates to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for (x, y) in holes:
            writer.writerow([x, y])

# 노이즈 제거거
def save_csv_3(holes, filename):
    """
    Save the hole coordinates to a CSV file.
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])
        for (x, y) in holes:
            writer.writerow([x, y])

# ======================== Step Functions & Their Visualizations ========================

def step1_raw_annotation(ann_img):
    """Return the raw annotation (without any blur)."""
    return ann_img


def visualize_step1(raw_annotation, filename):
    cv2.imwrite(filename, raw_annotation)
    return raw_annotation


def step2_binarize(ann_img, closing_level, opening_level, median_blur_ksize, morph_ksize):
    """Binarize the annotation image."""
    _, binary = cv2.threshold(ann_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filled = binary.copy()
    if closing_level > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        for _ in range(closing_level):
            filled = cv2.dilate(filled, kernel)
        for _ in range(closing_level):
            filled = cv2.erode(filled, kernel)
    if opening_level > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        for _ in range(opening_level):
            filled = cv2.erode(filled, kernel)
        for _ in range(opening_level):
            filled = cv2.dilate(filled, kernel)
    if median_blur_ksize > 1:
        filled = cv2.medianBlur(filled, median_blur_ksize)
    return filled


def visualize_step2(binarized, filename):
    cv2.imwrite(filename, binarized)
    return binarized


def step3_skeletonize(binarized, method='medial'):
    """Generate the skeleton from the binarized image."""
    skeleton = find_lines(binarized, method=method, preprocessed=True)
    return skeleton


def visualize_step3(skeleton, filename):
    cv2.imwrite(filename, skeleton)
    return skeleton


def step4_skeleton_postprocess(skeleton, ann_img, min_length_threshold, branch_proximity):
    """Postprocess the skeleton (split segments and draw contours)."""
    segments, _ = split_lines_from_skeleton(skeleton, min_length_threshold, branch_proximity)
    postprocessed = np.zeros_like(ann_img, dtype=np.uint8)
    for cnt in segments:
        cv2.drawContours(postprocessed, [cnt], -1, 255, thickness=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    
    postprocessed = cv2.dilate(postprocessed, kernel, iterations=1)

    return postprocessed


def visualize_step4(postprocessed, filename):
    cv2.imwrite(filename, postprocessed)
    return postprocessed


def step5_clearance_mask(postprocessed, fragile_clearance):
    """Compute the clearance mask from the postprocessed skeleton."""
    inverted = 255 - postprocessed
    dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    clearance_mask = (dist_transform < fragile_clearance).astype(np.uint8) * 255
    return clearance_mask


def visualize_step5(clearance_mask, filename):
    # clearance_mask = cv2.GaussianBlur(clearance_mask, (15, 15), 0)
    cv2.imwrite(filename, clearance_mask)
    return clearance_mask


def step6_generate_holes(ann_img, postprocessed, hole_min_distance, hole_gridsearch_distance,
                         fragile_clearance, noise_std, safety_factor, thresh_roi):
    """Generate candidate hole centers via grid sampling."""
    holes = generate_holes_grid(ann_img.shape, postprocessed, hole_min_distance,
                                hole_gridsearch_distance, fragile_clearance, noise_std, safety_factor)
    roi_mask = np.where(ann_img >= thresh_roi, 255, 0).astype(np.uint8)
    holes = [hole for hole in holes if roi_mask[hole[1], hole[0]] == 255]
    return holes

# 취약영역 천공 생성
def step6_generate_holes_2(ann_img, hole_min_distance, hole_gridsearch_distance,
                         noise_std):
    """Generate candidate hole centers via grid sampling."""
    holes = generate_holes_grid_2(ann_img.shape, hole_min_distance,
                                hole_gridsearch_distance, noise_std)
    
    holes = [hole for hole in holes ]
    return holes


# 노이즈 없는 것 
def step6_generate_holes_3(ann_img, postprocessed, hole_min_distance, hole_gridsearch_distance,
                         fragile_clearance, noise_std, safety_factor,thresh_roi):
    """Generate candidate hole centers via grid sampling."""
    holes = generate_holes_grid_3(ann_img.shape, postprocessed, hole_min_distance,
                                hole_gridsearch_distance, fragile_clearance, noise_std, safety_factor)
    roi_mask = np.where(ann_img >= thresh_roi, 255, 0).astype(np.uint8)
    holes = [hole for hole in holes if roi_mask[hole[1], hole[0]] == 255]
    return holes

def visualize_step6(ann_img, holes, hole_min_distance, filename):
    """
    Draw circles and dashed circles on the annotation.
    The background remains as the original black-white image while the circles are drawn in red.
    """
    vis = cv2.cvtColor(ann_img, cv2.COLOR_GRAY2BGR)
    for (x, y) in holes:
        cv2.circle(vis, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        step_angle = (5 + 3) / float(hole_min_distance)
        num_steps = int(2 * np.pi / step_angle)
        for i in range(num_steps):
            theta_start = i * step_angle
            theta_end = theta_start + (5 / float(hole_min_distance))
            pt1 = (int(x + hole_min_distance * np.cos(theta_start)),
                   int(y + hole_min_distance * np.sin(theta_start)))
            pt2 = (int(x + hole_min_distance * np.cos(theta_end)),
                   int(y + hole_min_distance * np.sin(theta_end)))
            cv2.line(vis, pt1, pt2, (0, 0, 255), 1)
    cv2.imwrite(filename, vis)
    return vis


def MasterVisualization(raw_img, overlay_img, alpha_raw=0.3, alpha_overlay=0.7):
    """Blend the raw image with the overlay image."""
    final_result = cv2.addWeighted(raw_img, alpha_raw, overlay_img, alpha_overlay, 0)
    return final_result


def visualize_MasterVisualization(final_result, filename):
    cv2.imwrite(filename, final_result)
    return final_result


# ======================== Run Function for Import ========================

def run(ann_img, raw_img, output_dir, basename):
    """
    Run the full processing pipeline on two input ndarrays.

    Parameters:
      ann_img    - grayscale annotation image (ndarray)
      raw_img    - raw color image (BGR, ndarray)
      output_dir - directory in which to save the results
      basename   - base filename used for naming output files
    """
    # Hardcoded parameters
    method = 'medial'
    closing_level = 1
    opening_level = 1
    median_blur_ksize = 3
    morph_ksize = 3
    min_length_threshold = 20  # Remove segments shorter than this.
    branch_proximity = 5  # For branch detection.
    hole_min_distance = 75  # Also used as the hole radius.
    hole_gridsearch_distance = 15  # Grid search spacing.
    fragile_clearance = 20
    thresh_roi = 0  # All pixels are included.
    noise_std = 3
    safety_factor = 3
    
    # Create output subdirectory if it does not exist.
    out_subdir = os.path.join(output_dir, basename)
    if not os.path.exists(out_subdir):
        os.makedirs(out_subdir)
    
    # ---- Step 1: Raw Annotation ----
    raw_annotation = step1_raw_annotation(ann_img)
    visualize_step1(raw_annotation, os.path.join(out_subdir, f"{basename}_Step1_RawAnnotation.png"))
    
    # ---- Step 2: Binarized Image ----
    binarized = step2_binarize(ann_img, closing_level, opening_level, median_blur_ksize, morph_ksize)
    visualize_step2(binarized, os.path.join(out_subdir, f"{basename}_Step2_Binarized.png"))
    
    # ---- Step 3: Skeletonization ----
    skeleton = step3_skeletonize(binarized, method=method)
    visualize_step3(skeleton, os.path.join(out_subdir, f"{basename}_Step3_Skeleton.png"))
    
    # ---- Step 4: Skeleton Postprocess ----
    skeleton_post = step4_skeleton_postprocess(skeleton, ann_img, min_length_threshold, branch_proximity)
    visualize_step4(skeleton_post, os.path.join(out_subdir, f"{basename}_Step4_SkeletonPostprocess.png"))
    
    # ---- Step 5: Clearance Mask ----
    clearance_mask = step5_clearance_mask(skeleton_post, fragile_clearance)
    clearance_mask_blur = visualize_step5(clearance_mask, os.path.join(out_subdir, f"{basename}_Step5_ClearanceMask.png"))
    
    # ---- Step 6: Generate Holes ----
    holes = step6_generate_holes(ann_img, skeleton_post, hole_min_distance, hole_gridsearch_distance,
                                 fragile_clearance, noise_std, safety_factor, thresh_roi)
    holes_vis = visualize_step6(skeleton_post, holes, hole_min_distance, os.path.join(out_subdir, f"{basename}_Step6_Holes.png"))
    
    # Save CSV of holes coordinates.
    csv_filename = os.path.join(out_subdir, f"{basename}_Step6_Holes.csv")
    
    # ----- Verification: Read CSV and visualize the holes -----
    verification_img = cv2.cvtColor(skeleton_post, cv2.COLOR_GRAY2BGR)
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header.
        for row in reader:
            x, y = int(row[0]), int(row[1])
            cv2.circle(verification_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imwrite(os.path.join(out_subdir, f"{basename}_verification_holes.png"), verification_img)
    
    # ---- MasterVisualization (Final Alpha Blended Result) ----
    master_vis = MasterVisualization(raw_img, holes_vis, 0.3, 0.7)
    visualize_MasterVisualization(master_vis, os.path.join(out_subdir, f"{basename}_MasterVisualization.png"))
    
    # Optional: Concatenation of master visualization and raw image
    raw_img_resized = cv2.resize(raw_img, (128, 128))
    master_vis_resized = cv2.resize(master_vis, (128, 128))
    concat_h = np.hstack((master_vis_resized, raw_img_resized))
    cv2.imwrite(os.path.join(out_subdir, "Concatenated_Result.png"), concat_h)


# ======================== Main Debugging Block ========================

if __name__ == '__main__':
    # Hardcoded directories for debugging
    root_path = './dataset/sam_final/pred/'
    raw_root = './dataset/sam_final/img/'
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ann_files = sorted([f for f in os.listdir(root_path) if f.lower().endswith(('.jpg', '.png'))])
    for fname in tqdm.tqdm(ann_files):
        ann_path = os.path.join(root_path, fname)
        raw_path = os.path.join(raw_root, fname)
        basename = os.path.splitext(fname)[0]
        # Read images from disk for debugging.
        ann_img = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)  # (H x W)
        raw_img = cv2.imread(raw_path, cv2.IMREAD_COLOR)  # (H x W x 3)
        if ann_img is None or raw_img is None:
            # continue
            raise FileNotFoundError(f"Could not read image files: {ann_path}, {raw_path}")
        
        # Run the processing pipeline.
        run(ann_img, raw_img, output_dir, basename)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
