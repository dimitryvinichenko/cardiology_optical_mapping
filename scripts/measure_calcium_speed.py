import sys
from pathlib import Path
from copy import deepcopy

import numpy as np
from cellpose import core, io, models
from natsort import natsorted

sys.path.append(str(Path(__file__).parent.parent))

from src.velocity_utils import calculate_flow_field
from src.display_utils import add_dot_on_frame, create_video_from_frames, apply_mask


DATA_DIR = Path("./data")


def load_data(dir_path: Path, image_ext: str = ".tif") -> dict:
    """
    Load the data from the given path.
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"The directory {dir_path} does not exist.")
    
    files = natsorted([
        f for f in dir_path.glob("*"+image_ext) 
        if "_masks" not in f.name 
        and "_flows" not in f.name
    ])

    if len(files) == 0:
        raise FileNotFoundError("No image files found, did you specify the correct folder and extension?")

    files_data = {}
    for file_path in files:
        # Load the image data which returns (num_frames, h, w, c)
        frames = io.imread(file_path)
        
        assert frames is not None, "Image data is None"
        assert frames.ndim == 4, "Image data must be 4D"
        
        # Unstack into list of (h, w, c) arrays
        frames = [frames[i] for i in range(frames.shape[0]-1)]
        files_data[file_path.as_posix()] = frames
        
    return files_data


def load_model() -> models.CellposeModel:
    io.logger_setup() # run this to get printing of progress

    if not core.use_gpu():
        raise RuntimeError("No GPU access, change your runtime")

    model = models.CellposeModel(gpu=True)
    return model


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


def find_best_mask(masks: list[np.ndarray]) -> np.ndarray:
    """
    Find the best mask by comparing the number of cells in each mask.
    """
    return masks[np.argmax([np.unique(mask).size for mask in masks])]


def main():
    files_data = load_data(DATA_DIR)
    print(f"Found videos: {list(files_data.keys())}")
    
    model = load_model()
    
    masks = {}
    for file_path, frames in files_data.items():
        masks[file_path] = segment_images(model, frames)
    
    for file_path, frames in files_data.items():
        mask = find_best_mask(masks[file_path])
        coms = calculate_flow_field(frames, mask)
        
        frames = deepcopy(files_data[file_path])

        assert len(frames) == len(coms), f"Frames and coms have different lengths: {len(frames)} != {len(coms)}"
        for i in range(len(frames)):
            for com in coms[i].values():
                frames[i] = add_dot_on_frame(frames[i], com)
        
        create_video_from_frames(apply_mask(frames, mask, alpha=0.3), f"./output/{Path(file_path).stem}.mp4", fps=10)
    

if __name__ == "__main__":
    main()
