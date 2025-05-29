import concurrent.futures

import cv2
import numpy as np
from tqdm.auto import tqdm
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import peak_widths
from cellpose import models


def segment_images(
    model: models.CellposeModel, 
    frames: list[np.ndarray],
    *,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0,
    tile_norm_blocksize: int = 0,
    batch_size: int = 32,
    niter: int | None = None,
    diameter: int | None = None,
    ) -> list[np.ndarray]:
    """
    Segment the frames using the Cellpose model.
    """
    masks, _, _ = model.eval(
        x=frames, 
        batch_size=batch_size, 
        flow_threshold=flow_threshold, 
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize}, # type: ignore
        niter=niter,
        diameter=diameter,
        channel_axis=2,
    )
    return masks


def parse_cells_mask(mask: np.ndarray) -> dict[int, np.ndarray]:
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]
    return {cell_id: mask == cell_id for cell_id in cell_ids}


def select_region_by_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    frame_copy = frame.copy()
    frame_copy[~mask] = 0
    return frame_copy


def calculate_center_of_mass(image: np.ndarray) -> tuple[float, float]:
    """
    Calculate the center of mass of an image represented as a NumPy array.
    
    Args:
        image (np.ndarray): Input image array of shape (h, w, 3)
        
    Returns:
        tuple[float, float]: (x, y) coordinates of the center of mass
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input array must have shape (h, w, 3)")
    
    # Convert to grayscale using cv2
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create coordinate grids
    h, w = grayscale.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    # Calculate total mass (sum of pixel intensities)
    total_mass = np.sum(grayscale)
    
    if total_mass == 0:
        return (w/2, h/2)  # Return center of image if no mass
    
    # Calculate center of mass using weighted average
    x_com = np.sum(x_coords * grayscale) / total_mass
    y_com = np.sum(y_coords * grayscale) / total_mass
    
    return (x_com, y_com)


def get_center_of_mass_indices(image: np.ndarray) -> tuple[int, int]:
    """
    Get the integer indices (pixel coordinates) of the center of mass.
    
    Args:
        image (np.ndarray): Input image array of shape (h, w, 3)
        
    Returns:
        tuple[int, int]: (x, y) integer indices of the center of mass
    """
    x_com, y_com = calculate_center_of_mass(image)
    return (int(round(x_com)), int(round(y_com)))


def get_cells_center_of_masses(frames: list[np.ndarray], mask: np.ndarray, max_workers: int = 10) -> list[dict[int, tuple[int, int]]]:
    assert frames[0].shape[:-1] == mask.shape, f"Frame and mask shapes do not match: {frames[0].shape[:-1]} != {mask.shape}"

    cell_masks = parse_cells_mask(mask)
    
    def process_frame(frame_idx):
        frame = frames[frame_idx]
        frame_coms = {}
        for cell_id, cell_mask in cell_masks.items():
            cell_frame = frame.copy()
            cell_frame[~cell_mask] = 0
            frame_coms[cell_id] = get_center_of_mass_indices(cell_frame)
        return frame_idx, frame_coms
    
    # Use ThreadPoolExecutor to process frames in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, i) for i in range(len(frames))]
        
        # Track progress with tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Calculating center of masses"):
            idx, result = future.result()
            results.append(result)
    
    return results


def calculate_flow_field(frames: list[np.ndarray], mask: np.ndarray, threshold: int = 25) -> list[dict[int, tuple[int, int]]]:
    """
    Calculate the center of moving clouds (flowing liquid) in each frame.
    
    Args:
        frames (list[np.ndarray]): List of image frames
        mask (np.ndarray): Mask to identify regions of interest
        
    Returns:
        list[dict[int, tuple[int, int]]]: List of dictionaries mapping region IDs to their flow centers
    """
    # Parse the mask to identify different regions
    regions = parse_cells_mask(mask)
    
    # Initialize results list to store flow centers for each frame
    results = []
    
    # Dictionary to track the last known flow center for each region
    last_flow_centers = {}
    
    # Process each frame
    for i in tqdm(range(len(frames) - 1), desc="Calculating flow field"):
        current_frame = frames[i]
        next_frame = frames[i + 1]
        
        # Dictionary to store flow centers for this frame
        flow_centers = {}
        
        # Process each region in the mask
        for region_id, region_mask in regions.items():
            # Apply mask to current and next frame
            current_region = current_frame.copy()
            current_region[~region_mask] = 0
            
            next_region = next_frame.copy()
            next_region[~region_mask] = 0
            
            # Calculate difference to identify movement
            diff = cv2.absdiff(current_region, next_region)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.shape[-1] == 3 else diff
            
            # Apply threshold to highlight significant movement
            _, thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
            
            # Find center of the moving cloud
            if np.sum(thresh) > 0:  # Check if there's any movement # type: ignore
                # Convert thresh to 3-channel image for get_center_of_mass_indices
                thresh_3ch = np.stack([thresh, thresh, thresh], axis=2)
                flow_center = get_center_of_mass_indices(thresh_3ch)
                flow_centers[region_id] = flow_center
                # Update last known position
                last_flow_centers[region_id] = flow_center
            else:
                # If no movement detected, use the last known position if available
                if region_id in last_flow_centers:
                    flow_centers[region_id] = last_flow_centers[region_id]
                else:
                    # If no previous position, use the center of the region
                    flow_centers[region_id] = get_center_of_mass_indices(current_region)
        
        results.append(flow_centers)
    
    # For the last frame, use the same flow centers as the previous frame
    if frames:
        results.append(results[-1] if results else {})
    
    return results


def find_moving_wavefront(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Detect moving wavefronts in video frames using optical flow and create masks for each frame.
    
    Args:
        frames (list[np.ndarray]): List of image frames
        
    Returns:
        list[np.ndarray]: List of binary masks where 255 indicates moving wavefront regions
    """
    if not frames:
        return []
    
    # Initialize list to store masks
    wavefront_masks = []
    
    # Preprocess first frame
    def preprocess(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=10
        )
        return gray, corners
    
    # Initialize with first frame
    prev_gray, prev_pts = preprocess(frames[0])
    
    # Create initial mask (will be empty for first frame)
    initial_mask = np.zeros_like(frames[0])
    if len(initial_mask.shape) == 3:
        initial_mask = cv2.cvtColor(initial_mask, cv2.COLOR_RGB2GRAY)
    wavefront_masks.append(initial_mask)
    
    # Process each subsequent frame
    for i in range(1, len(frames)):
        # Create mask for current frame
        mask = np.zeros_like(frames[i])
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        curr_gray, curr_pts = preprocess(frames[i])
        
        # Skip if no features found in previous frame
        if prev_pts is None:
            wavefront_masks.append(mask)
            prev_gray = curr_gray.copy()
            prev_pts = curr_pts
            continue
        
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, winSize=(15, 15), maxLevel=2 # type: ignore
        ) # type: ignore
        
        # Skip if no flow vectors found
        if curr_pts is None or status is None:
            wavefront_masks.append(mask)
            prev_gray = curr_gray.copy()
            prev_pts = curr_pts
            continue
        
        # Get valid points
        good_new = curr_pts[status == 1]
        good_old = prev_pts[status == 1]
        
        # Draw flow vectors on mask
        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            # Draw line between old and new position
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 255, 2) # type: ignore
            # Draw circle at new position
            cv2.circle(mask, (int(a), int(b)), 5, 255, -1) # type: ignore
        
        # Apply morphological operations to enhance the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        wavefront_masks.append(mask)
        
        # Update for next iteration
        prev_gray = curr_gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)
    
    return wavefront_masks


def find_peaks(data, height_threshold=None, distance=None):
    """
    Find peaks in a 1D array of data.
    
    Args:
        data (list or np.ndarray): The data to analyze
        height_threshold (float, optional): Minimum height for a peak to be considered
        distance (int, optional): Minimum distance between peaks
        
    Returns:
        np.ndarray: Indices of the peaks in the data
    """
    
    if height_threshold is None:
        # Default to 60% of the data range
        height_threshold = np.min(data) + 0.6 * (np.max(data) - np.min(data))
    
    if distance is None:
        # Default to 5% of the data length
        distance = max(int(len(data) * 0.05), 1)
    
    peaks, _ = scipy_find_peaks(data, height=height_threshold, distance=distance)
    return peaks


def measure_peak_widths(data, peaks, height_threshold=None, rel_height=0.5):
    """
    Measure the width of each peak in the data.
    
    Parameters:
    -----------
    data : array-like
        The intensity data containing peaks
    peaks : array-like
        Indices of the peaks in the data
    height_threshold : float, optional
        Minimum height for a peak to be considered. If None, all peaks are considered.
    rel_height : float, optional
        The relative height at which the peak width is measured.
        Default is 0.5 (half height), which gives the full width at half maximum (FWHM).
    
    Returns:
    --------
    widths : array
        The width of each peak
    width_heights : array
        The height at which the width is measured for each peak
    left_ips : array
        The interpolated left edge positions of each peak
    right_ips : array
        The interpolated right edge positions of each peak
    """
    
    # Filter peaks by height if threshold is provided
    if height_threshold is not None:
        valid_peaks = [p for p in peaks if data[p] >= height_threshold]
    else:
        valid_peaks = peaks
    
    if len(valid_peaks) == 0:
        raise ValueError("No valid peaks found")
    
    # Convert valid_peaks to numpy array if it's not already
    valid_peaks = np.array(valid_peaks)
    
    # Calculate peak widths
    # The rel_height parameter determines at what fraction of the peak height the width is measured
    # 0.5 means half the height (FWHM), 0.9 means 90% of the height (narrower width)
    widths, width_heights, left_ips, right_ips = peak_widths(data, valid_peaks, rel_height=rel_height)
    
    return widths, width_heights, left_ips, right_ips