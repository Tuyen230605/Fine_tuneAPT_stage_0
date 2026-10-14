import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class PlateConfig:
    """Configuration for a plate object."""
    usd_path: str
    scale: np.ndarray
    mass: float
    description: str
    name: str
    is_isaac_sim: bool = False


# Define available plates
PLATES: Dict[str, PlateConfig] = {
    # Standard plates
    "plate": PlateConfig(
        usd_path="assets/mesh/wooden_plate/model.usd",
        scale=np.array([2, 2, 2]),
        mass=1.0,
        description="A standard wooden plate",
        name="plate"
    ),
    "bowl": PlateConfig(
        usd_path="assets/mesh/Axis_Aligned/024_bowl.usd",
        scale=np.array([1, 1, 1]),
        mass=1.0,
        description="A red bowl",
        name="red bowl"
    ),
    "box": PlateConfig(
        usd_path="assets/mesh/box/box.usd",
        scale=np.array([1, 1, 1]),
        mass=1.0,
        description="A white box",
        name="white box"
    ),

    ########################################################
    # Novel plates
    ########################################################
    "mug": PlateConfig(
        usd_path="assets/mesh/Axis_Aligned/025_mug.usd",
        scale=np.array([1, 1, 1]),
        mass=1.0,
        description="A mug",
        name="red mug"
    ),
    "dish": PlateConfig(
        usd_path="assets/mesh/GraspNet_models/simplified_objects/046/textured_simplified.usd",
        scale=np.array([1, 1, 1]),
        mass=1.0,
        description="A dish",
        name="dish"
    ),
    "soapbox": PlateConfig(
        usd_path="assets/mesh/GraspNet_models/simplified_objects/039/textured_simplified.usd",
        scale=np.array([1, 1, 1]),
        mass=1.0,
        description="A soap box",
        name="soap box"
    ),
    
    
}


def get_plate_config(plate_key: str) -> PlateConfig:
    """Get the configuration for a specific plate."""
    if plate_key not in PLATES:
        raise ValueError(f"Plate '{plate_key}' not found. Available plates: {list(PLATES.keys())}")
    return PLATES[plate_key]


def get_random_plate() -> str:
    """Get a random plate key."""
    return np.random.choice(list(PLATES.keys()))


def get_novel_plate() -> str:
    """Get a novel plate key."""
    return np.random.choice(list(PLATES.keys())[2:])


def list_available_plates() -> List[Tuple[str, str]]:
    """List all available plates with their descriptions."""
    return [(key, config.description) for key, config in PLATES.items()] 