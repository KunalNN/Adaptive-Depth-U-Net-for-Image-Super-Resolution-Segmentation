"""
Adaptive-depth U-Net training script for image segmentation.

This script consolidates the adaptive-depth notebook workflow into a reusable
entry point that can be launched on a cluster. It parameterises the scale
factor, produces low-resolution inputs on-the-fly, and keeps the encoder depth
in line with the design table from the project summary.
"""

def load_image_stack(directory: Path, size: int, limit: int | None = None) -> np.ndarray:
    """Load and normalise images from a directory into an array of shape (N, H, W, 3)."""
    paths = sorted_alphanumeric([p.name for p in directory.iterdir() if p.is_file()])
    if limit is not None:
        paths = paths[:limit]

    images: List[np.ndarray] = []
    for filename in paths:
        img = cv2.imread(str(directory / filename), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {directory / filename}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        images.append(img)

    if not images:
        raise ValueError(f"No images found in {directory}")

    return np.stack(images, axis=0)
