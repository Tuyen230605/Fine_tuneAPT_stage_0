"""2D sampling helpers for tabletop scene generation.

`sample_xy` draws points from a quarter-annulus in the workspace (x in
[0.3, 0.5], y in [0, 0.5]). `sample_obj_xys2` then resolves overlaps using a
simple iterative repulsion step (`update_pos`).
"""

import numpy as np
from isaac_env.non_isaac import sampling


def sample_xy(shape=None):
    """Uniform sample in the +x, +y quarter of an annulus with r in [0.3, 0.5].

    Args:
        shape: scalar or tuple controlling the output shape; None returns one (2,) point.
    Returns:
        Array of shape `(*shape, 2)` (or `(2,)` if `shape is None`).
    """
    rmin, rmax = 0.3, 0.5
    if shape is None:
        xy = sampling.uniform_in_sphere(N=1, d=2, r_min=rmin/rmax)[0]
    else:
        if isinstance(shape, int):
            shape = (shape,)
        xy = sampling.uniform_in_sphere(N=np.prod(shape), d=2, r_min=rmin/rmax)
        xy = np.reshape(xy, list(shape) + [2])
    xy = np.abs(xy) * rmax
    return xy


def update_pos(
    points: np.ndarray, 
    radius: np.ndarray, 
    fixed: np.ndarray, 
    step_size: float,
    bbox: np.ndarray = None
):
    """
    - points: (N, ndim), float
    - raidus: (N,), float,
    - fixed: (N,), bool
    - step_size: float, update time step
    - bbox: (ndim, 2), 2 means [min, max]
    """
    deltas = points[:, None, :] - points[None, :, :]  # (N, N, dim)
    dist = np.linalg.norm(deltas, axis=-1)  # (N, N)
    thresh = radius[:, None] + radius[None, :]  # (N, N)
    collision_mask = dist < thresh  # (N, N)
    np.fill_diagonal(collision_mask, False)  # avoid self collision
    collision_mask[fixed[:, None] & fixed[None, :]] = False
    # disable collisions between fixed objects

    # simple constant force to resolve collision
    force = (dist < thresh).astype(np.float32)  # (N, N)
    np.fill_diagonal(force, 0.)  # avoid self collision

    dirs = deltas / (dist[:, :, None] + 1e-15)
    acc = (force[:, :, None] * dirs).sum(axis=1)  # (N, ndim)
    acc[fixed] = 0.  # disable position update of fixed points

    points = points + acc * step_size
    if bbox is not None:
        for d in range(len(bbox)):
            eps = 1e-7
            mask = points[:, d] < bbox[d][0]
            if np.any(mask):
                points[mask, d] = bbox[d][0] + np.random.uniform(-eps, eps, size=mask.sum())
            
            mask = points[:, d] > bbox[d][1]
            if np.any(mask):
                points[mask, d] = bbox[d][1] + np.random.uniform(-eps, eps, size=mask.sum())
    return points, collision_mask


def sample_obj_xys(N: int, avoid_dist: float = 0.2):
    xys = sample_xy(N)
    radius = np.ones(N) * avoid_dist / 2.0
    fixed = np.zeros(N, dtype=bool)
    bbox = [[0, 0.6], [-0.6, 0.6]]

    for _ in range(100):
        xys, collision_mask = update_pos(xys, radius, fixed, 
                                         step_size=0.02, bbox=bbox)
        if not np.any(collision_mask):
            break
    return xys


def sample_obj_xys2(
    N_pick: int,
    N_place: int,
    pick_raidus: float = 0.07,
    place_raidus: float = 0.1,
):
    """Sample non-overlapping 2D positions for pickable + placeable objects.

    Args:
        N_pick:       number of pickable objects.
        N_place:      number of placeable (target container) objects.
        pick_raidus:  min-separation radius for pickable objects.
        place_raidus: min-separation radius for placeable objects.
    Returns:
        (pick_xys, place_xys) — each shape (N_*, 2), inside the [0, 0.5]² bbox.
    """
    xys = sample_xy(N_pick + N_place)
    raidus = np.concatenate([np.ones(N_pick) * pick_raidus,
                             np.ones(N_place) * place_raidus])
    fixed = np.zeros(N_pick + N_place, dtype=bool)
    bbox = [[0.0, 0.5], [0.0, 0.5]]

    for n in range(100):
        xys, collision_mask = update_pos(xys, raidus, fixed,
                                         step_size=0.02, bbox=bbox)
        if not np.any(collision_mask):
            break
    print("[INFO] take {} steps to avoid collision".format(n))

    pick_xys  = xys[:N_pick]
    place_xys = xys[N_pick:]
    return pick_xys, place_xys


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    pick_xys, place_xys = sample_obj_xys2(6, 2)

    plt.figure()
    plt.plot(pick_xys[:, 0], pick_xys[:, 1], "o")
    plt.plot(place_xys[:, 0], place_xys[:, 1], "o")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()



