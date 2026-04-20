from typing import Literal
import numpy as np


def uma_pathloss(distances: np.ndarray, 
                 rng: np.random.Generator, 
                 wavelength: float, 
                 gamma_los: float,
                 gamma_nlos: float,
                 fix_los: Literal[0, 1, 2] = 0, 
                 shadow_fading_var: float = 0.0,
                 d0: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    UMA path-loss with distance-dependent LOS/NLOS and shadow fading.
    
    Args:
        distances: Arbitrary-shape array of TX–RX distances in meters.
        rng: NumPy random number generator.
        wavelength: Wavelength.
        gamma_los: Path-loss exponent for LOS.
        gamma_nlos: Path-loss exponent for NLOS.
        fix_los: 0 = sample LOS by probability; 1 = force NLOS; 2 = force LOS.
        shadow_fading_var: Shadow-fading variance. Use 0.0 to disable.
        d0: Reference distance in meters.


    Returns:
        PL: Path-loss in dB, same shape as `distances`, dtype float64.
        LOS_mask: Boolean mask (True=LOS, False=NLOS), same shape as `distances`.

    """
    # Distance-dependent LOS probability.
    LOS_prob = np.exp(-distances / 25)
    LOS_prob[distances < d0] = 1.0 # To make sure LOS PL <= NLOS PL always.

    # Optional overrides: force LOS or NLOS.
    if fix_los == 1:
        LOS_prob = 0
    elif fix_los == 2:
        LOS_prob = 1

    LOS_mask = rng.uniform(0.0, 1.0, size=distances.shape) < LOS_prob

    # Reference-path-loss term at d0.
    PL0 = 10 * np.log10((4 * np.pi * d0 / wavelength)**2)

    gamma = np.where(LOS_mask, float(gamma_los), float(gamma_nlos))
    PL = PL0 + 10 * gamma * np.log10(distances / d0) + rng.normal(
        0, np.sqrt(shadow_fading_var), LOS_mask.shape)
    return PL, LOS_mask


def _pairwise_pathloss(A_pos: np.ndarray,
                       B_pos: np.ndarray,
                       rng: np.random.Generator,
                       wavelength: float,
                       shadow_fading_var: float,
                       fix_los: Literal[0, 1, 2],
                       gamma_los: float = 2.0,
                       gamma_nlos: float = 4.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Common core: distances, path loss, LOS mask, and angles between two sets A and B.
    Supports per-operator inputs or a global set for either side.
    Returns shapes (N_A, N_B, N_OP).
    """
    # Normalize shapes to (N_OP, N_*, 2).
    def ensure_op_dim(X: np.ndarray) -> np.ndarray:
        if X.ndim == 2 and X.shape[1] == 2:   # (N_*, 2)
            return X[None, ...]               # -> (1, N_*, 2)
        return X                              # assumed (N_OP, N_*, 2)

    A = ensure_op_dim(A_pos)
    B = ensure_op_dim(B_pos)

    # Broadcast operator dimension if one side is global.
    if A.shape[0] != B.shape[0]:
        if A.shape[0] == 1:
            A = np.repeat(A, B.shape[0], axis=0)
        elif B.shape[0] == 1:
            B = np.repeat(B, A.shape[0], axis=0)
        else:
            raise ValueError("Operator dimension mismatch for A_pos and B_pos.")

    n_op = A.shape[0]
    n_a = A.shape[1]
    n_b = B.shape[1]

    PL_db   = np.empty((n_a, n_b, n_op), dtype=np.float64)
    LOS_int = np.empty((n_a, n_b, n_op), dtype=int)
    angles_A_B = np.empty((n_a, n_b, n_op), dtype=np.float64)
    angles_B_A = np.empty((n_a, n_b, n_op), dtype=np.float64)

    for no in range(n_op):
        A_no = A[no]                     # (N_A, 2)
        B_no = B[no]                     # (N_B, 2)

        # Pairwise differences: diff = A - B (shape: N_A x N_B x 2).
        diff = A_no[:, None, :] - B_no[None, :, :]

        distances = np.linalg.norm(diff, axis=2)
        angles_A_B[:, :, no] = np.arctan2(diff[..., 1], diff[..., 0])
        angles_B_A[:, :, no] = np.arctan2(-diff[..., 1], -diff[..., 0])

        # Path loss + LOS.
        PL, LOS = uma_pathloss(distances, rng, wavelength, gamma_los, 
            gamma_nlos, fix_los, shadow_fading_var)
        PL_db[:, :, no]   = PL
        LOS_int[:, :, no] = LOS.astype(int)

    return PL_db, LOS_int, angles_A_B, angles_B_A


def direct_PL(UE_pos: np.ndarray,
              BS_pos: np.ndarray,
              rng: np.random.Generator,
              wavelength: float,
              shadow_fading_var: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    UE–BS direct path-loss, LOS mask, and angles.

    Args:
        UE_pos (N_OP, N_UE, 2): UE positions per operator.
        BS_pos (N_OP, N_BS, 2): BS positions per operator.
        rng: NumPy random number generator.
        wavelength: Wavelength.
        shadow_fading_var: Shadow-fading variance.

    Returns:
        PL_UE_BS (N_UE, N_BS, N_OP): Path-loss in dB.
        LOS_UE_BS (N_UE, N_BS, N_OP): LOS mask (0/1).
        angles_UE_BS (N_UE, N_BS, N_OP): Angles between UE and BS.
        angles_BS_UE (N_UE, N_BS, N_OP): Angles between BS and UE.
    """
    PL, LOS, angles_UE_BS, angles_BS_UE = _pairwise_pathloss(
        A_pos=UE_pos, B_pos=BS_pos, rng=rng, wavelength=wavelength,
        shadow_fading_var=shadow_fading_var, fix_los=1)
    return PL, LOS, angles_UE_BS, angles_BS_UE


def RIS_PL(UE_pos: np.ndarray,
           BS_pos: np.ndarray,
           RIS_pos: np.ndarray,
           rng: np.random.Generator,
           wavelength: float,
           shadow_fading_var: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RIS-related path-loss, LOS mask, and angles.
    Computes path-loss and angles for both RIS–BS and RIS–UE links.

    Args:
        UE_pos (N_OP, N_UE, 2): UE positions per operator.
        BS_pos (N_OP, N_BS, 2): BS positions per operator.
        RIS_pos (N_RIS, 2): RIS positions. Global across operators.
        rng: NumPy random number generator.
        wavelength: Wavelength.
        shadow_fading_var: Shadow-fading variance.

    Returns:
        PL_RIS_BS (N_RIS, N_BS, N_OP): Path-loss between RIS and BS.
        LOS_RIS_BS (N_RIS, N_BS, N_OP): LOS mask (0/1).
        angles_RIS_BS (N_RIS, N_BS, N_OP): Angles between RIS and BS.
        angles_BS_RIS (N_RIS, N_BS, N_OP): Angles between BS and RIS.
        PL_RIS_UE (N_RIS, N_UE, N_OP): Path-loss between RIS and UE.
        LOS_RIS_UE (N_RIS, N_UE, N_OP): LOS mask (0/1).
        angles_RIS_UE (N_RIS, N_UE, N_OP): Angles between RIS and UE.
    """
    PL_RIS_BS, LOS_RIS_BS, angles_RIS_BS, angles_BS_RIS = _pairwise_pathloss(
        A_pos=RIS_pos, B_pos=BS_pos, rng=rng, wavelength=wavelength,
        shadow_fading_var=shadow_fading_var, fix_los=2)

    PL_RIS_UE, LOS_RIS_UE, angles_RIS_UE, _ = _pairwise_pathloss(
        A_pos=RIS_pos, B_pos=UE_pos, rng=rng, wavelength=wavelength,
        shadow_fading_var=shadow_fading_var, fix_los=0)

    return (
        PL_RIS_BS, LOS_RIS_BS, angles_RIS_BS, angles_BS_RIS,
        PL_RIS_UE, LOS_RIS_UE, angles_RIS_UE
    )


def uca_response(M: int, 
                 R: float, 
                 wavelength: float, 
                 phi: np.ndarray,
    ) -> np.ndarray:
    """
    Array steering vector for a Uniform Circular Array (UCA).

    Args:
        M: Number of antenna elements (uniformly spaced on a circle).
        R: Array radius.
        wavelength: Wavelength.
        phi: Angles.

    Returns:
        array_response: Complex steering vector for the UCA.
    """
    m_vec = np.reshape(np.arange(M), (M, 1, 1, 1))
    array_response = np.exp(-1j * 2 * np.pi / wavelength * R *
                            np.cos(phi - m_vec * 2 * np.pi / M))
    return array_response


def Gauss_channel(array_size: tuple[int, ...], rng: np.random.Generator,
    ) -> np.ndarray:
    """
    Generate a complex, normalized Gaussian channel.

    Args:
        array_size: Number of antennas.
        rng: NumPy random number generator.

    Returns:
        channels: array_size complex array.
    """
    channels = 1 / np.sqrt(2) * (np.array(rng.normal(0, 1, array_size)) +
                                 1j * np.array(rng.normal(0, 1, array_size)))
    return channels


def direct_channel_vectors(M_BS: int, 
                           M_UE: int, 
                           N_UE: int, 
                           N_OP: int, 
                           N_BS: int, 
                           NN: int, 
                           rng: np.random.Generator, 
                           wavelength: float, 
                           angles_UE_BS: np.ndarray, 
                           angles_BS_UE: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Direct-link channel vectors.

    Args:
        M_BS: Number of BS antenna elements.
        M_UE: Number of UE antenna elements.
        N_UE: Number of UEs per operator.
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        NN: Number of microscopic fading realizations.
        rng: NumPy random number generator.
        wavelength: Wavelength..
        angles_UE_BS (N_UE, N_BS, N_OP): Angles between UE and BS.
        angles_BS_UE (N_UE, N_BS, N_OP): Angles between BS and UE.

    Returns:
        UE_BS_channel (M_UE, N_UE, N_BS, N_OP): UCA response at UE side.
        BS_UE_channel (M_BS, N_UE, N_BS, N_OP): UCA response at BS side.
        channels_BS_UE (M_BS, 1, N_UE, N_BS, N_OP, NN): Gaussian part.
    """
    # Directional parts.
    UE_BS_channel = uca_response(M_UE, (M_UE * wavelength / 2) / (2 * np.pi), wavelength, angles_UE_BS)
    BS_UE_channel = uca_response(M_BS, (M_BS * wavelength / 2) / (2 * np.pi), wavelength, angles_BS_UE)

    # Gaussian part.
    channels_BS_UE = Gauss_channel((M_BS, 1, N_UE, N_BS, N_OP, NN), rng)

    return UE_BS_channel, BS_UE_channel, channels_BS_UE


def RIS_channel_vectors(M_RIS: int, 
                        M_BS: int, 
                        M_UE: int, 
                        N_UE: int, 
                        N_OP: int, 
                        N_BS: int, 
                        N_RIS: int, 
                        NN: int, 
                        rng: np.random.Generator, 
                        wavelength: float, 
                        angles_RIS_BS: np.ndarray, 
                        angles_BS_RIS: np.ndarray, 
                        angles_RIS_UE: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RIS-link channel vectors.

    Args:
        M_RIS: Number of RIS elements.
        M_BS: Number of BS antenna elements.
        M_UE: Number of UE antenna elements.
        N_UE: Number of UEs per operator.
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_RIS: Number of RISs.
        NN: Number of microscopic fading realizations.
        rng: NumPy random number generator.
        wavelength: Wavelength.
        angles_RIS_BS (N_RIS, N_BS, N_OP): Angles between RIS and BS.
        angles_BS_RIS (N_RIS, N_BS, N_OP): Angles between BS and RIS.
        angles_RIS_UE (N_RIS, N_UE, N_OP): Angles between RIS and UE.

    Returns:
        channels_BS_RIS (M_RIS, M_BS, N_BS, N_RIS, N_OP, NN): Gaussian part.
        channels_RIS_UE (1, M_RIS, N_RIS, N_UE, N_OP, NN): Gaussian part (transposed).
        RIS_BS_channel (M_RIS, N_RIS, N_BS, N_OP): UCA response at RIS side.
        BS_RIS_channel (M_BS, N_RIS, N_BS, N_OP): UCA response at BS side.
        RIS_UE_channel (M_RIS, N_RIS, N_UE, N_OP): UCA response at RIS side.
    """
    # Gaussian parts.
    channels_BS_RIS = Gauss_channel((M_RIS, M_BS, N_BS, N_RIS, N_OP, NN), rng)
    channels_RIS_UE = Gauss_channel((1, M_RIS, N_RIS, N_UE, N_OP, NN), rng)  # Transposed.

    # Directional parts.
    RIS_BS_channel = uca_response(M_RIS, (M_RIS * wavelength / 2) / (2 * np.pi), wavelength, angles_RIS_BS)
    BS_RIS_channel = uca_response(M_BS, (M_BS * wavelength / 2) / (2 * np.pi), wavelength, angles_BS_RIS)
    RIS_UE_channel = uca_response(M_RIS, (M_RIS * wavelength / 2) / (2 * np.pi), wavelength, angles_RIS_UE)

    return channels_BS_RIS, channels_RIS_UE, RIS_BS_channel, BS_RIS_channel, RIS_UE_channel


def RIS_alloc_response(RIS_UE_channel: np.ndarray,
                       RIS_BS_channel: np.ndarray,
                       RIS_alloc: np.ndarray,
                       BS_RIS_alloc: np.ndarray,
                       BS_UE_assoc: np.ndarray,
                       rng: np.random.Generator,
    ) -> np.ndarray:
    """
    Build the optimized RIS response under RIS allocation.

    Args:
        RIS_UE_channel (M_RIS, N_RIS, N_UE, N_OP): UCA response at RIS side.
        RIS_BS_channel (M_RIS, N_RIS, N_BS, N_OP): UCA response at RIS side.
        RIS_alloc: Allocation of each RIS to operators.
        BS_RIS_alloc: Allocation of each RIS to BSs.
        BS_UE_assoc: Allocation of each UEs to BSs.
        rng: NumPy random number generator.

    Returns:
        RIS_resp (M_RIS, N_RIS, N_UE, N_OP): RIS response.
    """
    RIS_BS_phase = np.angle(RIS_BS_channel)
    RIS_UE_phase = np.angle(RIS_UE_channel)
    M_RIS, N_RIS, N_BS, N_OP = RIS_BS_channel.shape
    N_UE = RIS_UE_phase.shape[2]
    
    RIS_resp = np.exp(-1j * 2 * np.pi * rng.uniform(0, 1, size=(M_RIS, N_RIS, N_UE, N_OP)))
    for no in range(N_OP):
        for nu in range(N_UE):
            for nr in range(N_RIS):
                # RIS "nr" belongs to operator "no" AND RIS's BS matches UE "nu"'s BS 
                if (RIS_alloc[nr] == no) and (BS_RIS_alloc[nr] == BS_UE_assoc[nu, no]):
                    RIS_resp[:, nr, nu, no] = np.exp(
                        -1j * (RIS_BS_phase[:, nr, BS_UE_assoc[nu,no], no] + RIS_UE_phase[:, nr, nu, no]))
    return RIS_resp


def direct_channel(N_OP: int,
                   N_BS: int,
                   N_UE: int,
                   M_BS: int,
                   M_UE: int,
                   NN: int,
                   K: np.ndarray,
                   LOS_UE_BS: np.ndarray,
                   pow_ue_bs: np.ndarray,
                   UE_BS_channel: np.ndarray,
                   BS_UE_channel: np.ndarray,
                   channels_BS_UE: np.ndarray,
                   BS_UE_assoc: np.ndarray,
    ) -> np.ndarray:
    """
    Construct the fading channel for direct BS–UE links using 
    distance/geometry-dependent LOS conditions and Rician fading.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        M_BS: Number of BS antenna elements.
        M_UE: Number of UE antenna elements.
        NN: Number of microscopic fading realizations.
        K: Rician K-factor for LOS and NLOS.
        LOS_UE_BS (N_UE, N_BS, N_OP): LOS mask (0/1).
        pow_ue_bs (N_UE, N_BS, N_OP): Path gain.
        UE_BS_channel (M_UE, N_UE, N_BS, N_OP): UCA response at UE side.
        BS_UE_channel (M_BS, N_UE, N_BS, N_OP): UCA response at BS side.
        channels_BS_UE (M_BS, 1, N_UE, N_BS, N_OP, NN): Gaussian part.
        BS_UE_assoc: Allocation of each UEs to BSs.

    Returns:
        direct_channel_BS_UE (M_BS, 1, N_UE, N_BS, N_OP, NN): Direct (BS–UE) channel.
    """
    # We assume that K[1] = 0.
    KK = (K[0] * LOS_UE_BS + 0 * (1 - LOS_UE_BS))[None, None, :, :, :, None]

    # Deterministic (directional) part.
    ue_bs = np.zeros((M_BS, 1, N_UE, N_BS, N_OP, NN), dtype=np.complex128)
    ue_bs[...] = BS_UE_channel[:, None, :, :, :, None]

    direct_channel_BS_UE = np.sqrt(KK / (1 + KK)) * ue_bs

    # Serving BSs are phase-synchronized.
    for no in range(N_OP):
        for nu in range(N_UE):
            temp_chan = direct_channel_BS_UE[:, :, nu, BS_UE_assoc[nu, no], no, :]
            direct_channel_BS_UE[:, :, nu, BS_UE_assoc[nu, no], no, :] = temp_chan * \
                np.exp(-1j*np.angle(temp_chan))
    
    # Add scattering.
    direct_channel_BS_UE = direct_channel_BS_UE + np.sqrt(1 / (1 + KK)) * channels_BS_UE

    # Apply path loss.
    direct_channel_BS_UE = direct_channel_BS_UE * np.sqrt(pow_ue_bs)[:, :, :, None]

    return direct_channel_BS_UE


def RIS_channel(N_OP: int,
                N_BS: int,
                N_RIS: int,
                N_UE: int,
                M_BS: int,
                M_RIS: int,
                M_UE: int,
                NN: int,
                K: np.ndarray, 
                LOS_RIS_UE: np.ndarray, 
                LOS_RIS_BS: np.ndarray, 
                RIS_BS_channel: np.ndarray, 
                BS_RIS_channel: np.ndarray, 
                channels_BS_RIS: np.ndarray, 
                RIS_UE_channel: np.ndarray, 
                channels_RIS_UE: np.ndarray, 
                BS_UE_assoc: np.ndarray, 
                RIS_resp: np.ndarray, 
                pow_ris_bs: np.ndarray, 
                pow_ris_ue: np.ndarray, 
                rng: np.random.Generator,
    ) -> np.ndarray:
    """
    Construct the fading channel for BS–RIS–UE links using 
    distance/geometry-dependent LOS conditions and Rician fading on the BS–RIS 
    and UE–RIS sublinks, combined through a diagonal RIS phase.

    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_RIS: Number of RISs.
        N_UE: Number of UEs per operator.
        M_BS: Number of BS antenna elements.
        M_RIS: Number of RIS elements.
        M_UE: Number of UE antenna elements.
        NN: Number of microscopic fading realizations.
        K: Rician K-factor for LOS and NLOS.
        LOS_RIS_UE (N_RIS, N_UE, N_OP): LOS mask (0/1).
        LOS_RIS_BS (N_RIS, N_BS, N_OP): LOS mask (0/1).
        RIS_BS_channel (M_RIS, N_RIS, N_BS, N_OP): UCA response at RIS side.
        BS_RIS_channel (M_BS, N_RIS, N_BS, N_OP): UCA response at BS side.
        channels_BS_RIS (M_RIS, M_BS, N_BS, N_RIS, N_OP, NN): Gaussian part.
        RIS_UE_channel (M_RIS, N_RIS, N_UE, N_OP): UCA response at RIS side.
        channels_RIS_UE (1, M_RIS, N_RIS, N_UE, N_OP, NN): Gaussian part (transposed).
        BS_UE_assoc: Allocation of each UEs to BSs.
        RIS_resp (M_RIS, N_RIS, N_UE, N_OP): RIS response.
        pow_ris_bs (N_RIS, N_BS, N_OP): Path gain.
        pow_ue_bs (N_UE, N_RIS, N_OP): Path gain.
        rng: NumPy random number generator.

    Returns:
        RIS_channel_BS_UE (M_BS, 1, N_UE, N_BS, N_OP, NN): Indirect (BS–RIS–UE) channel.
    """
    KK_UE = (K[0] * LOS_RIS_UE + K[1] * (1 - LOS_RIS_UE))[None, None, :, :, :, None]

    RIS_channel_BS_UE = np.zeros((1, M_BS, N_UE, N_BS, N_OP, NN), dtype=np.complex128)

    # Directional channel matrix between RIS and BS.
    ris_bs_all = (RIS_BS_channel[:, None, :, :, :] * BS_RIS_channel[None, :, :, :, :])[None, ...]

    for no in range(N_OP):
        for nb in range(N_BS):
            for nr in range(N_RIS):
                ris_bs = ris_bs_all[:, :, :, nr, nb, no]

                for nu in range(N_UE):
                    # Directional channel between RIS and UE.
                    ris_ue_direct = RIS_UE_channel[None, :, nr, nu, no, None]

                    # Apply a random phase shift to unintended signals.
                    if BS_UE_assoc[nu, no] != nb:
                        ris_ue_direct = ris_ue_direct * np.exp(1j * rng.uniform(0, np.pi*2, (1, 1, NN)))

                    # Scattering channel between RIS and UE.
                    ris_ue_scatter = channels_RIS_UE[:, :, nr, nu, no, :]

                    # Channel between UE and RIS.
                    kk_ue = KK_UE[:, :, nr, nu, no, :]
                    ris_ue = np.sqrt(kk_ue / (1 + kk_ue)) * ris_ue_direct + np.sqrt(1 / (1 + kk_ue)) * ris_ue_scatter
                    
                    ris_ue = np.moveaxis(ris_ue, 2, 0) # -> (NN, 1, M_RIS)
                    ris_chan = np.matmul(ris_ue, np.diag(RIS_resp[:, nr, nu, no]))
                    ris_chan = np.moveaxis(np.matmul(ris_chan, ris_bs), 0, 2)

                    # Total path loss BS-to-RIS and RIS-to-UE.
                    pl_lin = np.sqrt(pow_ris_bs[nr, nb, no] * pow_ris_ue[nu, nr, no])
                    RIS_channel_BS_UE[:, :, nu, nb, no, :] += ris_chan * pl_lin

    RIS_channel_BS_UE = np.moveaxis(RIS_channel_BS_UE, 1, 0)
    return RIS_channel_BS_UE
