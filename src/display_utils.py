import os

import cv2
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm


def create_video_from_frames(
    frames: list[np.ndarray], 
    output_path: str,
    fps: int = 30,
    is_color: bool = True
) -> None:
    """
    Create a video file from a list of numpy arrays (frames).
    
    Args:
        frames (List[np.ndarray]): List of frames as numpy arrays
        output_path (str): Path where the video will be saved
        fps (int, optional): Frames per second. Defaults to 30.
        is_color (bool, optional): Whether the frames are in color. Defaults to True.
    
    Raises:
        ValueError: If frames list is empty
        ValueError: If frames have inconsistent shapes
    """
    if not frames:
        raise ValueError("Frames list cannot be empty")
    
    # Get frame dimensions from the first frame
    height, width = frames[0].shape[:2]
    
    # Ensure all frames have the same dimensions
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("All frames must have the same dimensions")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Determine the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)
    
    try:
        # Write each frame to the video
        for frame in frames:
            # Ensure frame is in the correct format (uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            # Handle grayscale frames if is_color is True
            if is_color and len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Handle color frames if is_color is False
            elif not is_color and len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            out.write(frame)
    finally:
        # Release the video writer
        out.release()


def apply_mask(frames: list[np.ndarray], mask: np.ndarray | list[np.ndarray], alpha: float = 0.5) -> list[np.ndarray]:
    """
    Apply colorful masks to frames for visualization.
    
    Args:
        frames (list[np.ndarray]): List of image frames
        mask (np.ndarray | list[np.ndarray]): Either a single mask for all frames or a list of masks
        alpha (float, optional): Transparency of the mask. Defaults to 0.5.
        
    Returns:
        list[np.ndarray]: List of frames with colorful masks applied
    """
    if isinstance(mask, list):
        assert len(mask) == len(frames), "Number of masks must match number of frames"
        masks = mask
    else:
        # Use the same mask for all frames
        masks = [mask] * len(frames)
    
    result_frames = []
    
    for i, frame in enumerate(frames):
        # Create a copy of the frame to avoid modifying the original
        result_frame = frame.copy()
        
        # Get unique cell IDs (excluding background which is 0)
        unique_ids = np.unique(masks[i])
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        if len(unique_ids) > 0:
            # Create a normalized colormap for the mask
            norm = Normalize(vmin=1, vmax=len(unique_ids))
            cmap = cm.viridis # type: ignore
            
            # Create a colored mask with transparency
            colored_mask = np.zeros((*masks[i].shape, 3))  # RGB only, no alpha channel
            for cell_id in unique_ids:
                cell_mask = masks[i] == cell_id
                color = list(cmap(norm(cell_id)))[:3]  # Get only RGB components
                colored_mask[cell_mask] = color
            
            # Convert frame to float for proper blending
            result_frame = result_frame.astype(float) / 255.0
            
            # Blend the original frame with the colored mask
            mask_overlay = np.zeros_like(result_frame)
            mask_overlay[masks[i] > 0] = colored_mask[masks[i] > 0]
            
            # Apply the mask only where mask > 0
            result_frame = np.where(masks[i][..., None] > 0,
                                  result_frame * (1 - alpha) + mask_overlay * alpha,
                                  result_frame)
            
            # Convert back to uint8
            result_frame = (result_frame * 255).astype(np.uint8)
        
        result_frames.append(result_frame)
    
    return result_frames


def add_dot_on_frame(frame: np.ndarray, dot_position: tuple[int, int], color: tuple[int, int, int] = (255, 0, 0), radius: int = 5) -> np.ndarray:
    """
    Add a dot on a frame.
    """
    frame = frame.copy()
    cv2.circle(frame, dot_position, radius, color, -1)
    return frame


def visualize_contours(frames, contours_per_frame, color=(0, 255, 0), thickness=2):
    """
    Draw contours on frames.
    
    Args:
        frames (list[np.ndarray]): List of original frames
        contours_per_frame (list): List of contours for each frame
        color (tuple, optional): Color of contours. Defaults to (0, 255, 0).
        thickness (int, optional): Thickness of contour lines. Defaults to 2.
        
    Returns:
        list[np.ndarray]: Frames with contours drawn
    """
    frames_with_contours = []
    
    for i, frame in enumerate(frames):
        # Create a copy of the original frame
        frame_copy = frame.copy()
        
        # If we have contours for this frame, draw them
        if i < len(contours_per_frame):
            # Draw all contours on the frame
            cv2.drawContours(frame_copy, contours_per_frame[i], -1, color, thickness)
        
        frames_with_contours.append(frame_copy)
    
    return frames_with_contours
