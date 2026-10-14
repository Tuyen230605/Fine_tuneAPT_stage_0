import numbers
import numpy as np
from scipy.spatial.transform import Rotation as R


def uniform_on_sphere(N: int, d: int) -> np.ndarray:
    """Uniform sample N points on a d-dimensional sphere

    Returns: (N, d)
    """
    u = np.random.randn(N, d)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return u


def uniform_in_sphere(N: int, d: int, r_min: float = 0) -> np.ndarray:
    """Uniform sample N points inside a d-dimensional sphere with 
    with radius from [r_min, 1]

    Returns: (N, d)
    """
    assert r_min < 1

    u1 = uniform_on_sphere(N, d)
    if r_min == 0:
        u2 = np.random.uniform(0, 1, (N, 1))
    else:
        u2 = np.random.uniform(r_min ** d, 1, (N, 1))
    r = u2 ** (1.0/d)
    u = u1 * r
    return u


def fibonacci_disk(N: int, offset: float = 0.5) -> np.ndarray:
    """Evenly distribute N points on a 2D disk. 
    Ref: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Arguments:
    - N: int, number of points
    - offset: float, value from (0, 1), provide randomness, i.e., offset ~ Uniform(0, 1)

    Returns:
    - points: (N, 2)
    """
    if isinstance(offset, numbers.Number):
        assert offset >= 0 and offset <= 1
    else:
        assert np.all(offset >= 0) and np.all(offset <= 1)
    
    i = np.arange(N, dtype=float) + offset
    r = np.sqrt(i / N)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    points = np.stack([x, y], axis=-1)
    return points


def fibonacci_sphere(N: int, offset: float = 0.5) -> np.ndarray:
    """Evenly distribute N points on a 3D sphere. 
    Ref: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    Arguments:
    - N: int, number of points
    - offset: float, value from (0, 1), provide randomness, i.e., offset ~ Uniform(0, 1)

    Returns:
    - points: (N, 3)
    """
    if isinstance(offset, numbers.Number):
        assert offset >= 0 and offset <= 1
    else:
        assert np.all(offset >= 0) and np.all(offset <= 1)
    
    i = np.arange(N, dtype=float) + offset
    cos_phi = 1 - 2*i/N
    sin_phi = np.sqrt(1 - cos_phi**2)
    theta = np.pi * (1 + np.sqrt(5)) * i

    x = np.cos(theta) * sin_phi
    y = np.sin(theta) * sin_phi
    z = cos_phi

    points = np.stack([x, y, z], axis=-1)
    return points


def fibonacci_SO3(N: int, offset: float = 0.5):
    """Samples n rotations equivolumetrically using a Super-Fibonacci Spiral. 
    Ref: Marc Alexa, Super-Fibonacci Spirals. CVPR 22.

    Arguments:
    - N: int, number of points
    - offset: float, value from (0, 1), provide randomness, i.e., offset ~ Uniform(0, 1)

    Returns:
    - quat: (N, 4), quaterions
    """
    if isinstance(offset, numbers.Number):
        assert offset >= 0 and offset <= 1
    else:
        assert np.all(offset >= 0) and np.all(offset <= 1)
    
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041  # solution for: psi^4 = psi + 4

    i = np.arange(N, dtype=float) + offset
    r = np.sqrt(i / N)
    R = np.sqrt(1.0 - i / N)
    alpha = 2 * np.pi * i / phi
    beta = 2.0 * np.pi * i / psi
    quat = np.stack(
        [
            r * np.sin(alpha),
            r * np.cos(alpha),
            R * np.sin(beta),
            R * np.cos(beta),
        ],
        axis=-1,
    )
    return quat


def sample_pose_disturb(drz_max, dry_max, drx_max, dt_max) -> np.ndarray:
    """Apply right hand pose disturbance.

    Arguments:
    - drz_max, dry_max, drx_max: pose disturbance along self z-y-x axis, unit: degrees
    - dt_max: position disturbance

    Returns:
    - dT: (4, 4), right hand transformation disturbance
    """
    dRz = R.from_rotvec(np.array([0, 0, 1]) * np.random.uniform(-drz_max, drz_max) / 180*np.pi)
    dRy = R.from_rotvec(np.array([0, 1, 0]) * np.random.uniform(-dry_max, dry_max) / 180*np.pi)
    dRx = R.from_rotvec(np.array([1, 0, 0]) * np.random.uniform(-drx_max, drx_max) / 180*np.pi)
    dR = dRz.as_matrix() @ dRy.as_matrix() @ dRx.as_matrix()
    dt = uniform_in_sphere(1, 3)[0] * dt_max
    dT = np.eye(4)
    dT[:3, :3] = dR
    dT[:3, 3] = dt
    return dT


def sample_camera_pose(r_min, r_max, phi_min, phi_max, drz_max, dry_max, drx_max) -> np.ndarray:
    """
    Arguments:
    - r_min, r_max: raidus range of pose sampling space
    - phi_min, phi_max: angle between OP and ground plane, P is current position.
        * if phi == 0, OP is parallel to ground plane 
        * if phi == 90, will looking down to O
    - drz_max: rotation disturbance along self z-axis, unit: degrees
    - dry_max: rotation disturbance along self y-axis, unit: degrees
    - drx_max: rotation disturbance along self x-axis, unit: degrees

    Returns:
    - wcT: (4, 4), ^{world}_{cam} T, camera extrinsic
    """
    r = np.random.uniform(r_min, r_max)
    theta = np.random.uniform(-np.pi, np.pi)
    z = np.random.uniform(np.sin(phi_min/180*np.pi), np.sin(phi_max/180*np.pi))
    phi = np.arcsin(z)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    wct = np.array([x, y, z]) * r

    z_vec = -np.array([x, y, z])
    assert z_vec[-1] < 0
    y_vec = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        -np.cos(phi)
    ])
    x_vec = np.cross(y_vec, z_vec)
    wcR = np.stack([x_vec, y_vec, z_vec], axis=1)
    dRz = R.from_rotvec(np.array([0, 0, 1]) * np.random.uniform(-drz_max, drz_max) / 180*np.pi)
    dRy = R.from_rotvec(np.array([0, 1, 0]) * np.random.uniform(-dry_max, dry_max) / 180*np.pi)
    dRx = R.from_rotvec(np.array([1, 0, 0]) * np.random.uniform(-drx_max, drx_max) / 180*np.pi)
    wcR = wcR @ dRz.as_matrix() @ dRy.as_matrix() @ dRx.as_matrix()

    wcT = np.eye(4)
    wcT[:3, :3] = wcR
    wcT[:3, 3] = wct
    return wcT

