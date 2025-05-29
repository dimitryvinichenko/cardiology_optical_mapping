import sys
from pathlib import Path

import click
from cellpose import io, models, core
import numpy as np
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.velocity_utils import (
    measure_peak_widths, 
    segment_images, 
    parse_cells_mask,
    select_region_by_mask,
    find_peaks,
)

REL_HEIGHT = 0.85


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
    
    for cell_id, cell_mask in tqdm(cell_masks.items(), desc="Processing cells", total=len(cell_masks)):
        cell_regions = [select_region_by_mask(frame, cell_mask) for frame in frames]
        cell_intensities = [cell_region.sum() for cell_region in cell_regions]
        peaks = find_peaks(cell_intensities)
        widths, *_ = measure_peak_widths(cell_intensities, peaks, rel_height=REL_HEIGHT)
        width_mean = widths.mean()
        width_std = widths.std()
        avg_calcium_speeds.append(width_mean)
        print(f"Cell {cell_id} has average calcium speed {width_mean:.2f} ± {width_std:.2f} pixels/frame")
    
    overall_mean = np.mean(avg_calcium_speeds)
    overall_std = np.std(avg_calcium_speeds)
    print(f"\nAverage calcium speed across all cells: {overall_mean:.2f} ± {overall_std:.2f} pixels/frame")    


if __name__ == "__main__":
    main()