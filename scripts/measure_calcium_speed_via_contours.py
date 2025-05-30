import sys
from pathlib import Path

import click
import cv2
import numpy as np
from cellpose import core, io, models
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.velocity_utils import (
    calculate_contour_speed,
    find_contours,
    parse_cells_mask,
    segment_images,
    select_region_by_mask,
)


def load_video(video_path: Path) -> list[np.ndarray]:
    """
    Load the video from the given path.
    """
    frames = io.imread(video_path)
    
    assert frames is not None, "Image data is None"
    assert frames.ndim == 4, "Image data must be 4D"
    
    # Unstack into list of (h, w, c) arrays
    frames = [frames[i] for i in range(frames.shape[0]-1)]
    return frames


def load_model() -> models.CellposeModel:
    io.logger_setup() # run this to get printing of progress

    if not core.use_gpu():
        raise RuntimeError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)
    return model


def find_best_mask(masks: list[np.ndarray]) -> np.ndarray:
    """
    Find the best mask by comparing the number of cells in each mask.
    """
    return masks[np.argmax([np.unique(mask).size for mask in masks])]


@click.command()
@click.option("--video_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def main(video_path: Path):
    video_path = Path(video_path)
    assert video_path.suffix in [".tif", ".tiff"], f"Video file {video_path} is not a TIFF file"

    frames = load_video(video_path)
    
    model = load_model()
    masks = segment_images(model, frames) #type: ignore
    best_mask = find_best_mask(masks)
    cell_masks = parse_cells_mask(best_mask)
    
    avg_calcium_speeds = []
    
    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    avg_gray_frame = np.mean(gray_frames, axis=0).astype(np.uint8) # type: ignore
    normalized_gray_frames = [np.clip((frame.astype(np.float32) - avg_gray_frame), 0, 255).astype(np.uint8) for frame in gray_frames]
    
    for cell_id, cell_mask in tqdm(cell_masks.items(), desc="Processing cells", total=len(cell_masks)):
        cell_regions = [select_region_by_mask(frame, cell_mask) for frame in normalized_gray_frames]
        contours_per_frame = find_contours(cell_regions)
        speed_data = calculate_contour_speed(contours_per_frame)
        avg_calcium_speeds.append(speed_data["avg_speed"])
        print(f"Cell {cell_id} has average calcium speed {speed_data['avg_speed']:.2f} pixels/frame")
    
    overall_mean = np.mean(avg_calcium_speeds)
    overall_std = np.std(avg_calcium_speeds)
    print(f"\nAverage calcium speed across all cells: {overall_mean:.2f} ± {overall_std:.2f} pixels/frame")    


if __name__ == "__main__":
    main()