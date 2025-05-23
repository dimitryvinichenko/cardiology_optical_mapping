import concurrent.futures

import cv2
import numpy as np
from tqdm.auto import tqdm


def parse_cells_mask(mask: np.ndarray) -> dict[int, np.ndarray]:
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]
    return {cell_id: mask == cell_id for cell_id in cell_ids}


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
