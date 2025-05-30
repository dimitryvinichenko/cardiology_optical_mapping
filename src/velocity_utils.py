import concurrent.futures

import cv2
import numpy as np
from cellpose import models
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import peak_widths
from tqdm.auto import tqdm


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
        image (np.ndarray): Input image array of shape (h, w) for grayscale
                           or (h, w, 3) for RGB
        
    Returns:
        tuple[float, float]: (x, y) coordinates of the center of mass
    """
    # Handle grayscale or RGB images
    if image.ndim == 3 and image.shape[2] == 3:
        # Convert RGB to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        # Already grayscale
        grayscale = image
    else:
        raise ValueError("Input array must have shape (h, w) for grayscale or (h, w, 3) for RGB")
    
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


def find_contours(frames, threshold=20):
    """
    Find contours in grayscale frames.
    
    Args:
        frames (list[np.ndarray]): List of grayscale frames
        threshold (int, optional): Threshold value for binary conversion. Defaults to 20.
        
    Returns:
        list: List of contours for each frame
    """
    contours_per_frame = []
    
    for frame in frames:
        # Threshold the image to create a binary image
        _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contours_per_frame.append(contours)
    
    return contours_per_frame


def calculate_contour_speed(contours_per_frame, time_interval=1.0, min_contour_area=10):
    """
    Calculate the speed of moving contours across frames.
    
    Args:
        contours_per_frame (list): List of contours for each frame
        time_interval (float, optional): Time between frames in seconds. Defaults to 1.0.
        min_contour_area (float, optional): Minimum contour area to consider. Defaults to 10.
        
    Returns:
        dict: Dictionary containing:
            - 'speeds': List of speeds between consecutive frames (pixels/time_interval)
            - 'avg_speed': Average speed across all frames
            - 'centroid_positions': List of centroid positions for each frame
    """
    centroids = []
    speeds = []
    
    for frame_contours in contours_per_frame:
        frame_centroids = []
        
        for contour in frame_contours:
            # Filter out small contours
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
                
            # Calculate centroid of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                frame_centroids.append((cx, cy))
        
        # Use the largest contour's centroid if multiple are found
        if frame_centroids:
            centroids.append(frame_centroids[0])  # Simplified - could be improved
    
    # Calculate speeds between consecutive frames
    for i in range(1, len(centroids)):
        prev_x, prev_y = centroids[i-1]
        curr_x, curr_y = centroids[i]
        
        # Calculate Euclidean distance
        distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        
        # Speed = distance / time
        speed = distance / time_interval
        speeds.append(speed)
    
    # Calculate average speed
    avg_speed = np.mean(speeds) if speeds else 0
    
    return {
        'speeds': speeds,
        'avg_speed': avg_speed,
        'centroid_positions': centroids
    }
