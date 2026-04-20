import warnings
import numpy as np


def regular_noisy_placement(ndim: int, 
                            n_required_positions: int, 
                            ROI_size: np.ndarray, 
                            noise_var: float, 
                            rng: np.random.Generator,
    ) -> np.ndarray:
    """
    Regular placement within a region of interest with noise on top to observe different positions.

    Args:
        n_required_positions: Number of required positions per OP.
        ndim: Number of dimensions (not tested for more than 2).
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        noise_var: Variance of Gaussian perturbation.
        rng: Random number generator.

    Returns:
        rand_pos (n_required_positions, ndim): Array of generated positions.
    """
    dim_div = np.ceil(n_required_positions**(1 / ndim))  # number of positions per dimension
    dim_dist = (ROI_size[1] - ROI_size[0]) / dim_div  # distance between regular positions
    reg_grid = np.linspace(ROI_size[0] + dim_dist / 2, ROI_size[1] - dim_dist / 2, dim_div)  # grid of positions
    n_possible_positions = dim_div**ndim  # total number of possible positions
    possible_pos = np.array(np.meshgrid(reg_grid[:, 0], reg_grid[:, 1])).T.reshape(-1, 2)
    pos_choice = rng.choice(n_possible_positions, n_required_positions, replace=False)
    chosen_pos = possible_pos[pos_choice, :]
    rand_pos = chosen_pos + rng.normal(0, np.sqrt(noise_var), chosen_pos.shape)
    return rand_pos


def gen_POS(N_OP: int, 
            N_BS: int, 
            N_UE: int, 
            N_RIS: int, 
            ROI_size: np.ndarray, 
            rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positions of UEs, BSs, and RISs. UE positions are uniformly random, 
    BS and RIS positions are regular with noise.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        N_RIS: Number of RISs.
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        rng: Random number generator.
        
    Returns:
        UE_pos (N_OP, N_UE, 2): Array of user positions for each operator.
        BS_pos (N_OP, N_BS, 2): Array of base station positions for each operator.
        RIS_pos (N_RIS, 2): Array of RIS positions.
    """
    UE_pos = rng.uniform(ROI_size[0], ROI_size[1], size=(N_OP, N_UE, 2))

    # Generate random base station positions for each operator.
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 5)**2
    BS_pos = []
    for _ in range(N_OP):
        BS_pos.append(regular_noisy_placement(2, N_BS, ROI_size, pos_noise_var, rng))

    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    RIS_pos = regular_noisy_placement(2, N_RIS, ROI_size, pos_noise_var, rng)
    
    return UE_pos, BS_pos, RIS_pos

def place_on_line( 
                            n_required_positions: int, 
                            end_points: np.ndarray,
    ) -> np.ndarray:
    
    positions_x = np.linspace(end_points[0,0],end_points[0,1],n_required_positions)
    positions_y = np.linspace(end_points[1,0],end_points[1,1],n_required_positions)
    positions = np.array([positions_x,positions_y])
    positions = positions.transpose()
    return positions

def gen_POS_cell_edge(N_OP: int, 
                      N_BS: int, 
                      N_UE: int, 
                      N_RIS: int, 
                      ROI_size: np.ndarray, 
                      rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positions of UEs, BSs, and RISs. UE and RIS positions are 
    correlated random, BS positions are at the corners of the ROI.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        N_RIS: Number of RISs.
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        rng: Random number generator.
        
    Returns:
        UE_pos (N_OP, N_UE, 2): Array of user positions for each operator.
        BS_pos (N_OP, N_BS, 2): Array of base station positions for each operator.
        RIS_pos (N_RIS, 2): Array of RIS positions.
    """
    if N_BS != 2:
        # The problem is with the BS positions. 
        # ROI_size is 2x2 array, so we can only place two BSs at the corners.
        raise ValueError(f"Only implemented for two base stations.")
    if N_OP != 1:
        warnings.warn(f"BSs for different operators will be at the same position.")
    
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    cross_corr = -0.75
    covariance = np.array([[1.0, cross_corr], [cross_corr, 1]]) * pos_noise_var
    UE_pos = rng.multivariate_normal(np.array([0,0]), covariance, size=(N_OP, N_UE))

    # Put BSs at the corners of the ROI.
    BS_pos = np.array([ROI_size for _ in range(N_OP)])

    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    cross_corr = -0.9
    covariance = np.array([[1.0, cross_corr],[cross_corr, 1]]) * pos_noise_var
    RIS_pos = rng.multivariate_normal(np.array([0,0]), covariance, size=N_RIS)

    return UE_pos, BS_pos, RIS_pos


def gen_POS_around_BS(N_OP: int, 
                      N_BS: int, 
                      N_UE: int, 
                      N_RIS: int, 
                      ROI_size: np.ndarray, 
                      rng: np.random.Generator,
                      beta_ue: float = 0.25,
                      beta_ris: float = 0.5,
                      one_sided: bool = False,
                      bs_probs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate positions of UEs, BSs, and RISs. UEs are placed radially 
    around BSs located at the ROI corners. RISs are on the cell edge.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        N_RIS: Number of RISs.
        ROI_size (2, ndim): Region of interest array with min and max coordinates.
        rng: Random number generator.
        beta_ue: Radial exponent for UE distances (smaller means more near cell edge).
        beta_ris: Radial exponent for RIS distances (smaller means more near cell edge).
        one_sided: If True, UEs are more likely to be near one BS corner, and 
            the distribution won't be symmetric.
        bs_probs: Optional array of probabilities for each BS when sampling UE positions.
        
    Returns:
        UE_pos (N_OP, N_UE, 2): Array of user positions for each operator.
        BS_pos (N_OP, N_BS, 2): Array of base station positions for each operator.
        RIS_pos (N_RIS, 2): Array of RIS positions.
    """
    if N_BS != 2:
        # The problem is with the BS positions. 
        # ROI_size is 2x2 array, so we can only place two BSs at the corners.
        raise ValueError(f"Only implemented for two base stations.")
    if N_OP != 1:
        warnings.warn(f"BSs for different operators will be at the same position.")

    # Put BSs at the corners of the ROI.
    BS_pos = np.array([ROI_size for _ in range(N_OP)])

    # Bounds and cell radius.
    xmin, ymin = ROI_size[0]
    xmax, ymax = ROI_size[1]
    cell_radius = min(ROI_size[1] - ROI_size[0])

    def angle_range_for_corner(xc, yc):
        if np.isclose(xc, xmin) and np.isclose(yc, ymin):
            return 0.0, 0.5 * np.pi
        if np.isclose(xc, xmax) and np.isclose(yc, ymax):
            return np.pi, 1.5 * np.pi
        # Fallback.
        raise ValueError("The BSs location was changed. Please redefine this function.")

    # Sampler around the given corners.
    def sample_radial_around_corners(count: int, bs_corners: np.ndarray, beta: float,
                                     cell_radius: float, bs_probs: np.ndarray | None = None,
        ) -> np.ndarray:
        # Choose which corner each point belongs to (0 or 1).
        bs_idx = rng.integers(0, N_BS, size=count)

        # Choose which BS each point belongs to.
        bs_idx = rng.choice(np.arange(N_BS), size=count, p=bs_probs)

        u = rng.random(count)
        distances = cell_radius * (u**beta)

        # Angles.
        theta = np.empty(count, dtype=float)
        for nb in range(N_BS):
            mask = (bs_idx == nb)
            if not np.any(mask):
                continue
            a_min, a_max = angle_range_for_corner(*bs_corners[nb])
            theta[mask] = rng.uniform(a_min, a_max, size=mask.sum())

        # Offsets and positions.
        dx, dy = distances * np.cos(theta), distances * np.sin(theta)
        pts = bs_corners[bs_idx] + np.stack([dx, dy], axis=1)
        return pts

    # UE positions (per operator).
    UE_pos = np.empty((N_OP, N_UE, 2), dtype=float)
    if bs_probs is None:
        bs_probs = np.ones(N_BS) / N_BS
    if bs_probs.shape[0] != N_BS:
        raise ValueError("bs_probs must have length equal to number of BSs.")
    bs_probs = bs_probs / bs_probs.sum()
    for no in range(N_OP):
        if not one_sided:
            bs_probs = rng.choice([bs_probs, bs_probs[::-1]])
        UE_pos[no] = sample_radial_around_corners(N_UE, BS_pos[no], beta_ue, 
                                                  cell_radius / 2**(1/2), bs_probs=bs_probs)

    # RIS positions.
    """
    no = 0
    RIS_pos = sample_radial_around_corners(N_RIS, BS_pos[no], beta_ris, 
                                        cell_radius, bs_probs=np.array([0.5, 0.5]))
    """
    """
    pos_noise_var = ((np.amax(ROI_size) - np.amin(ROI_size)) / 4)**2
    cross_corr = -0.9
    covariance = np.array([[1.0, cross_corr],[cross_corr, 1]]) * pos_noise_var
    RIS_pos = rng.multivariate_normal(np.array([0,0]), covariance, size=N_RIS)
    """
    end_points = np.array([[ROI_size[0,0],ROI_size[1,1]],[ROI_size[1,0],ROI_size[0,1]]])
    RIS_pos = place_on_line(N_RIS, end_points)

    return UE_pos, BS_pos, RIS_pos
