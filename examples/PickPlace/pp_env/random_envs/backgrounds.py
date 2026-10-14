import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class BackgroundConfig:
    """Configuration for a background environment."""
    name: str
    description: str
    hdr_file: Optional[str] = None
    intensity: float = 1000.0


# Define available backgrounds
BACKGROUNDS: Dict[str, BackgroundConfig] = {
    # Standard backgrounds with HDR lighting
    "default": BackgroundConfig(
        name="Default Environment",
        description="Default environment with symmetrical garden",
        hdr_file="./assets/hdr/symmetrical_garden_02_2k.hdr",
        intensity=1000.0
    ),
    "garden": BackgroundConfig(
        name="Garden",
        description="Outdoor garden environment",
        hdr_file="./assets/hdr/symmetrical_garden_02_2k.hdr",
        intensity=1200.0
    ),
    "studio": BackgroundConfig(
        name="Studio",
        description="Professional photography studio",
        hdr_file="./assets/hdr/studio_small_04_4k.hdr",
        intensity=800.0
    ),
    "warehouse": BackgroundConfig(
        name="Warehouse",
        description="Large warehouse interior environment",
        hdr_file="./assets/hdr/ZetoCG_com_WarehouseInterior2b.hdr",
        intensity=900.0
    ),
    "exhibition": BackgroundConfig(
        name="Exhibition Hall",
        description="Spacious exhibition hall interior",
        hdr_file="./assets/hdr/ZetoCGcom_ExhibitionHall_Interior1.hdr",
        intensity=1000.0
    ),
    "hospital": BackgroundConfig(
        name="Hospital",
        description="Clean hospital room environment",
        hdr_file="./assets/hdr/hospital_room_4k.hdr",
        intensity=900.0
    ),
    "hotel": BackgroundConfig(
        name="Hotel Room",
        description="Cozy hotel room environment",
        hdr_file="./assets/hdr/hotel_room_4k.hdr",
        intensity=800.0
    ),
    "bathroom": BackgroundConfig(
        name="Bathroom",
        description="Modern bathroom environment",
        hdr_file="./assets/hdr/bathroom_4k.hdr",
        intensity=850.0
    ),
    "carpentry": BackgroundConfig(
        name="Carpentry Shop",
        description="Detailed carpentry workshop",
        hdr_file="./assets/hdr/carpentry_shop_01_4k.hdr",
        intensity=1100.0
    ),
    "lounge": BackgroundConfig(
        name="Wooden Lounge",
        description="Comfortable wooden lounge area",
        hdr_file="./assets/hdr/wooden_lounge_4k.hdr",
        intensity=900.0
    ),
    "empty_house": BackgroundConfig(
        name="Empty House",
        description="Small empty house interior",
        hdr_file="./assets/hdr/small_empty_house_4k.hdr",
        intensity=850.0
    )
}


def get_background_config(bg_key: str) -> BackgroundConfig:
    """Get the configuration for a specific background."""
    if bg_key not in BACKGROUNDS:
        raise ValueError(f"Background '{bg_key}' not found. Available backgrounds: {list(BACKGROUNDS.keys())}")
    return BACKGROUNDS[bg_key]


def get_random_background() -> str:
    """Get a random background key."""
    return np.random.choice(list(BACKGROUNDS.keys()))


def get_novel_background() -> str:
    """Get a novel background key."""
    return np.random.choice(list(BACKGROUNDS.keys())[1:])


def list_available_backgrounds() -> List[Tuple[str, str]]:
    """List all available backgrounds with their descriptions."""
    return [(key, config.description) for key, config in BACKGROUNDS.items()]