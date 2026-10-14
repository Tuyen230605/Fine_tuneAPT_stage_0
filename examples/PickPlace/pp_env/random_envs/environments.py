import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class EnvironmentConfig:
    """Configuration for a background environment."""
    name: str
    description: str
    usd_path: Optional[str] = None


# Define available environments
ENVIRONMENTS: Dict[str, EnvironmentConfig] = {
    "default": EnvironmentConfig(
        name="Default Environment",
        description="Balanced lighting for general purpose use",
        usd_path="assets/mesh/grid/default_environment.usd"
    ),
    "gridroom_black": EnvironmentConfig(
        name="Gridroom Black",
        description="Gridroom with black background",
        usd_path="assets/mesh/grid/gridroom_black.usd"
    ),
    "gridroom_curved": EnvironmentConfig(
        name="Gridroom Curved",
        description="Gridroom with curved walls",
        usd_path="assets/mesh/grid/gridroom_curved.usd"
    ),
    "rough_plane": EnvironmentConfig(
        name="Rough Plane",
        description="Rough plane",
        usd_path="assets/mesh/Terrains/rough_plane.usd"
    ),
}


def get_environment_config(env_key: str) -> EnvironmentConfig:
    """Get the configuration for a specific environment."""
    if env_key not in ENVIRONMENTS:
        raise ValueError(f"Environment '{env_key}' not found. Available environments: {list(ENVIRONMENTS.keys())}")
    return ENVIRONMENTS[env_key]


def get_random_environment(seed: int = None) -> str:
    """Get a random environment key."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(list(ENVIRONMENTS.keys()))


def get_novel_environment(seed: int = None) -> str:
    """Get a novel environment key."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.choice(list(ENVIRONMENTS.keys())[1:])


def list_available_environments() -> List[Tuple[str, str]]:
    """List all available environments with their descriptions."""
    return [(key, config.description) for key, config in ENVIRONMENTS.items()] 