import numpy as np

from src.envs.ris_env import RISAuctionEnv
from src.sim.channels import Gauss_channel


def compute_beamforming_vector(env: RISAuctionEnv, 
                               BS_RIS_alloc: np.ndarray,
                               RIS_channel_BS_UE: np.ndarray = np.array([])
    ) -> np.ndarray:
    """
    Compute the beamforming vectors for a given RIS allocation.

    Args:
        env: The environment object with BS_RIS_channel and other necessary data, 
            including: BS_RIS_channel (M_BS, N_RIS, N_BS, N_OP).
        BS_RIS_alloc (N_RIS,): array assigning each RIS to a base station.
        RIS_channel_BS_UE (M_BS, 1, N_UE, N_BS, N_OP, NN): Indirect (BS–RIS–UE) channel.

    Returns:
        beamforming_vector (M_BS, N_UE, N_BS, N_OP, NN): complex array.
    """
    beamforming_vector = np.zeros((env.M_BS, env.N_UE, env.N_BS, env.N_OP, env.NN), dtype=np.complex128)
    no = 0 # Only one operator right now.

    for nb in range(env.N_BS):
        # Indices of RISs and UEs assigned to this BS.
        ris_idxs = np.where(BS_RIS_alloc == nb)[0]
        ue_idxs = np.where(env.BS_UE_assoc == nb)[0]

        if ris_idxs.size == 0:
            # No RIS: random Gaussian beamforming.
            single_beamformer = Gauss_channel((env.M_BS, env.N_UE), env.rng)

            # Mask out non-assigned UEs.
            mask = np.zeros(env.N_UE, dtype=bool)
            mask[ue_idxs] = True
            single_beamformer[:, ~mask] = 0.0

            bf_norm = np.sum(np.abs(single_beamformer)**2, axis=0) # (N_UE)
            bf_norm[bf_norm == 0] = 1
            single_beamformer = np.sqrt(env.ps_lin / bf_norm) * single_beamformer
            single_beamformer = np.repeat(single_beamformer[:, :, np.newaxis, np.newaxis], env.NN, axis=3)
            beamforming_vector[:, :, nb, :, :] = single_beamformer
        else:
            # With RIS: beamforming towards the assigned RISs.
            single_beamformer = np.sqrt(1/env.M_BS)*np.conj(env.BS_RIS_channel[:, ris_idxs, nb, no]) # (M_BS, n_riss)
            current_power_alloc = env.power_alloc[nb, ris_idxs, :] # (n_riss, N_UE) (0.0 for not assigned RISs or not assigned users)
            single_beamformer = single_beamformer @ np.sqrt(current_power_alloc) # (M_BS, N_UE) (0.0 for not assigned users)
            bf_norm = np.sum(np.abs(single_beamformer)**2, axis=0) # (N_UE)
            bf_norm[bf_norm == 0] = 1
            single_beamformer = np.sqrt(env.ps_lin / bf_norm) * single_beamformer # (M_BS, N_UE)
            single_beamformer = np.repeat(single_beamformer[:, :, np.newaxis, np.newaxis], env.NN, axis=3)
            beamforming_vector[:, :, nb, :, :] = single_beamformer
            
            """
            # Equal power distribution.
            # (M_BS, N_RIS, N_BS, N_OP)
            single_beamformer = np.sum(np.conj(env.BS_RIS_channel[:, ris_idxs, nb, :]), axis=1, keepdims=False) # (M_BS, n_riss, N_OP) -> (M_BS, N_OP)
            bf_norm = np.sum(np.abs(single_beamformer)**2, axis=0) # (N_OP)
            single_beamformer = np.sqrt(env.ps_lin / bf_norm) * single_beamformer # (M_BS, N_OP)

            single_beamformer = np.repeat(single_beamformer[:, np.newaxis, :, np.newaxis], env.NN, axis=3) # (M_BS, 1, N_OP, NN)

            # Copy this to all assigned users.
            not_assigned_users = np.where(env.BS_UE_assoc[:, no] != nb)[0]
            single_beamformer = np.repeat(single_beamformer, env.N_UE, axis=1) # (M_BS, N_UE, N_OP, NN)
            single_beamformer[:, not_assigned_users, :, :] = 0.0 # (M_BS, N_UE, N_OP, NN)
            print(np.allclose(beamforming_vector[:, :, nb, :, :], single_beamformer))

            beamforming_vector[:, :, nb, :, :] = single_beamformer
            """
    return beamforming_vector
