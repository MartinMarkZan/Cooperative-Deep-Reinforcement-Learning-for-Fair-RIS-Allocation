import numpy as np


def est_values_pos(N_BS: int, 
                   N_RIS: int, 
                   N_UE: int, 
                   M_BS: int, 
                   M_RIS: int, 
                   M_UE: int, 
                   BS_idx: int, 
                   ps_lin: float, 
                   sigma_n2: float, 
                   pow_ue_bs: np.ndarray, 
                   pow_ris_bs: np.ndarray, 
                   pow_ris_ue: np.ndarray, 
                   BS_UE_assoc: np.ndarray, 
                   RIS_allocs: np.ndarray, 
                   IBI: np.ndarray, 
                   LOS_UE_BS: np.ndarray, 
                   LOS_RIS_BS: np.ndarray, 
                   LOS_RIS_UE: np.ndarray, 
                   K: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate values of an agent only for certain RIS allocations accounting 
    for positions and K-factors.

    Args:
        N_BS: Number of BSs per operator.
        N_RIS: Number of RISs.
        N_UE: Number of UEs per operator.
        M_BS: Number of BS antenna elements.
        M_RIS: Number of RIS elements.
        M_UE: Number of UE antenna elements.
        BS_idx: Index of the base station (agent).
        ps_lin: Linear power.
        sigma_n2: Linear noise power.
        pow_ue_bs (N_UE, N_BS): Path gain between UEs and BSs.
        pow_ris_bs (N_RIS, N_BS): Path gain between RISs and BSs.
        pow_ris_ue (N_UE, N_RIS): Path gain between RISs and UEs.
        BS_UE_assoc (N_UE,): Allocation of each UEs to BSs.
        RIS_allocs (N_allocs, N_RIS): Array of RIS allocations to consider, one-hot encoded.
        IBI (N_BS, N_BS, N_RIS): Interference between BSs at RISs.
        LOS_UE_BS (N_UE, N_BS): LOS mask (0/1) between UEs and BSs.
        LOS_RIS_BS (N_RIS, N_BS): LOS mask (0/1) between RISs and BSs.
        LOS_RIS_UE (N_RIS, N_UE): LOS mask (0/1) between RISs and UEs.
        K: Rician K-factor for LOS and NLOS.

    Returns:
        rates (n_users, n_allocs): Estimated achiveable rate for each RIS allocation.
        SINR_vals (n_users, n_allocs): Estimated SINR values for each user.
        power_vals (n_users, n_allocs): Estimated power values for each user.
        interference_vals (n_users, n_allocs): Estimated interference values for each user.
        power_allocation (N_RIS, N_UE, n_allocs): Power allocation for each RIS and user.
    """
    users = np.where(BS_UE_assoc == BS_idx)[0]
    n_users = users.size
    n_allocs = RIS_allocs.shape[0]

    assert n_users != 0, "No users associated with the BS."

    SINR_vals = np.zeros((n_users, n_allocs))
    power_vals = np.zeros((n_users, n_allocs))
    interference_vals = np.zeros((n_users, n_allocs))
    power_allocation = np.zeros((N_RIS, N_UE, n_allocs))

    # Direct channels.
    kk_direct = 0 # BSs and UEs are NLOS, otherwise: kk_direct = K[1 - LOS_UE_BS]
    # Received power over Gauss direct channels.
    pow_direct_Gauss = ps_lin * pow_ue_bs * (1 / (1 + kk_direct))
    # Magnitude of direct signals over directional channels (needed for coherent combination with RIS parts).
    mag_direct_dir = np.sqrt(pow_ue_bs) * np.sqrt(kk_direct / (1 + kk_direct))
    
    # Loop over all considered RIS allocations.
    for rr in range(n_allocs):
        ris_alloc = RIS_allocs[rr, :]  # current RIS allocation
        assigned_riss = np.flatnonzero(ris_alloc > 0)
        current_power_allocation = compute_power_allocation(N_RIS, N_UE, M_BS,
            M_RIS, BS_idx, ps_lin, pow_ris_bs, pow_ris_ue, BS_UE_assoc,
            ris_alloc, LOS_RIS_UE, K)
        
        tmp_SINR = []
        tmp_power = []
        tmp_interference = []

        # Incoherent signals from unassigned RISs (includes Gaussian channels).
        # No need for a K-factor here, since Gauss and directional channels 
        # behave the same for incoherent RISs.
        ris_ue = pow_ris_ue * (1 - ris_alloc)
        # Received power over incoherent RIS channels.
        pow_ris_incoh = ps_lin * ris_ue @ (M_RIS * pow_ris_bs)

        # Incoherent signals from assigned RISs (these are received over Gauss channels).
        if assigned_riss.shape[0] == 0:
            pow_ris_Gauss = np.zeros((N_UE, 0))
        else:
            pow_ris_Gauss = np.zeros((N_UE, assigned_riss.shape[0]))
            # Here we need a K-factor to account for relative strength compared to coherent signals.
            kk_ue = np.where(LOS_RIS_UE[np.ix_(assigned_riss, users)] == 1, K[0], K[1])  # K-factor between UE and RIS.
            ris_ue = (1.0 / (1.0 + kk_ue)).T * pow_ris_ue[np.ix_(users, assigned_riss)]
            # Received power over incoherent RIS channels.
            pow_ris_Gauss[users, :] = ris_ue * (M_BS * M_RIS * pow_ris_bs[assigned_riss, BS_idx])

        for nu in users:
            # Coherent signals from assigned RISs.
            kk_ue = K[1 - LOS_RIS_UE[:, nu]]  # K-factor between UE and RIS.
            # Coherent combination increases magnitude by M_RIS.
            ris_mag_coh = M_RIS * np.sqrt(pow_ris_bs[assigned_riss, BS_idx])
            # Coherent intended signals from RISs.
            mag_ris_intended = np.sqrt((kk_ue[assigned_riss] / (1 + kk_ue[assigned_riss]))) * np.sqrt(pow_ris_ue[nu, assigned_riss]) * ris_mag_coh * np.sqrt(M_BS)
            
            # Interfering BSs.
            interf_ind = np.delete(np.arange(N_BS, dtype=np.int32), BS_idx)
            
            # mag_ris_interf = np.zeros((len(assigned_riss), len(interf_ind)))
            # nbc = 0
            # for nb in interf_ind:  # Coherent interfering signals from RISs.
            #     nrc = 0
            #     for nr in assigned_riss:
            #         mag_ris_interf[nrc, nbc] = np.sqrt((kk_ue[nr] / (1+kk_ue[nr]))) * np.sqrt(
            #             pow_ris_ue[nu, nr]) * np.sqrt(pow_ris_bs[nr, nb]) * IBI[BS_idx, nb, nr]
            #         nrc += 1
            #     nbc += 1
            
            power_alloc_user = current_power_allocation[assigned_riss, nu]
            power_coherent = ps_lin * (mag_ris_intended @ power_alloc_user)**2 # This was inside the parenthesis: + mag_direct_dir[nu, BS_idx]
            # interf_coherent = ps_lin * sum(np.sum(mag_ris_interf**2, axis=0) + (mag_direct_dir[nu, interf_ind])**2)
            power = pow_direct_Gauss[nu, BS_idx] + power_coherent + ps_lin * pow_ris_Gauss[nu] @ (power_alloc_user)**2   # + pow_ris_incoh[nu, BS_idx]
            interference = sum(pow_direct_Gauss[nu, interf_ind]) + sum(pow_ris_incoh[nu, interf_ind]) # + interf_coherent + + sum(pow_ris_Gauss[nu, interf_ind])
            tmp_SINR.append(power / (interference + sigma_n2))
            tmp_power.append(power)
            tmp_interference.append(interference)

        SINR_vals[:, rr] = np.array(tmp_SINR)
        power_vals[:, rr] = np.array(tmp_power)
        interference_vals[:, rr] = np.array(tmp_interference)
        power_allocation[:, :, rr] = current_power_allocation

    rates = np.log2(SINR_vals + 1)
    rates /= n_users  # Time-orthogonal users.
    return rates, SINR_vals, power_vals, interference_vals, power_allocation


def compute_power_allocation(N_RIS: int, 
                             N_UE: int, 
                             M_BS: int, 
                             M_RIS: int, 
                             BS_idx: int, 
                             ps_lin: float, 
                             pow_ris_bs: np.ndarray, 
                             pow_ris_ue: np.ndarray, 
                             BS_UE_assoc: np.ndarray, 
                             RIS_alloc: np.ndarray, 
                             LOS_RIS_UE: np.ndarray, 
                             K: np.ndarray,
    ) -> np.ndarray:
    """
    Estimate power allocation for the given BS.

    Args:
        N_RIS: Number of RISs.
        N_UE: Number of UEs per operator.
        M_BS: Number of BS antenna elements.
        M_RIS: Number of RIS elements.
        BS_idx: Index of the base station (agent).
        ps_lin: Linear power.
        pow_ris_bs (N_RIS, N_BS): Path gain between RISs and BSs.
        pow_ris_ue (N_UE, N_RIS): Path gain between RISs and UEs.
        BS_UE_assoc (N_UE,): Allocation of each UEs to BSs.
        RIS_alloc (N_RIS,): array assigning each RIS to a base station, one-hot encoded.
        LOS_RIS_UE (N_RIS, N_UE): LOS mask (0/1) between RISs and UEs.
        K: Rician K-factor for LOS and NLOS.

    Returns:
        power_allocation (N_RIS, N_UE): Power allocation for each RIS and user.
    """
    power_allocation = np.zeros((N_RIS, N_UE))
    users = np.where(BS_UE_assoc == BS_idx)[0]
    assigned_riss = np.flatnonzero(RIS_alloc > 0)
    force_equal_power = False
    if force_equal_power:
        power_allocation[np.ix_(assigned_riss, users)] = 1 / np.sqrt(assigned_riss.size if assigned_riss.size > 0 else 1)
    else:
        for nu in users:
            # Coherent signals from assigned RISs.
            kk_ue = K[1 - LOS_RIS_UE[:, nu]]  # K-factor between UE and RIS.
            # Coherent combination increases magnitude by M_RIS.
            ris_mag_coh = M_RIS * np.sqrt(pow_ris_bs[assigned_riss, BS_idx])
            # Coherent intended signals from RISs.
            mag_ris_intended = np.sqrt((kk_ue[assigned_riss] / (1 + kk_ue[assigned_riss]))) * np.sqrt(pow_ris_ue[nu, assigned_riss]) * ris_mag_coh * np.sqrt(M_BS)
            current_pow_alloc = mag_ris_intended / np.linalg.norm(mag_ris_intended)
            power_allocation[assigned_riss, nu] = current_pow_alloc

    return power_allocation


def calculate_fairness_weights(current_values: dict,
                               agents: list,
                               gamma: float = 0.0,
    ) -> dict:
    """
    Calculate fairness weights for agents based on their current values.

    Args:
        current_values (n_agents): Dictionary of current values for each agent.
        agents (n_agents): List of agent identifiers.
        gamma: Fairness factor. Larger gamma increases fairness pressure.
        
    Returns:
        fairness_weights (n_agents): Dictionary of fairness weights for each agent.
    """
    total_performance = np.sum(current_values[agent]**gamma for agent in agents)
    if total_performance == 0:
        # Avoid division by zero; assign zero weights.
        return {agent: 0.0 for agent in agents}
    return {agent: current_values[agent]**gamma / (total_performance) * len(agents) for agent in agents}


def get_SINR(N_OP: int,
             N_BS: int, 
             N_UE: int,  
             sigma_n2: float, 
             NN: int, 
             BS_UE_assoc: np.ndarray, 
             beamforming_vector: np.ndarray, 
             direct_signal: np.ndarray, 
             ris_signal: np.ndarray = np.array([]),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get signal-to-interference-plus-noise ratio (SINR) of the links.
    
    Args:
        N_OP: Number of operators.
        N_BS: Number of BSs per operator.
        N_UE: Number of UEs per operator.
        sigma_n2: Linear noise power.
        NN: Number of microscopic fading realizations.
        BS_UE_assoc (N_UE, N_OP): Allocation of each UEs to BSs.
        beamforming_vector (M_BS, N_UE, N_BS, N_OP, NN): complex array.
        direct_signal (M_BS, 1, N_UE, N_BS, N_OP, NN): Direct (BS–UE) channel.
        ris_signal (M_BS, 1, N_UE, N_BS, N_OP, NN): RIS-assisted (BS–UE) channel.
    
    Returns:
        SINR (N_UE, N_OP, NN): SINR.
        UE_power (N_UE, N_OP, NN): Received power of intended signals.
        Int_power (N_UE, N_OP, NN): Received power of interfering signals.
    """
    total_signal = direct_signal + (ris_signal if ris_signal.size else 0)
    total_signal_bf = total_signal[:, 0, :, :, :, :]
    total_signal_bf = np.einsum('i...,i...->...', beamforming_vector, total_signal_bf)
    power = np.reshape(np.abs(total_signal_bf)**2, (N_UE, N_BS, N_OP, NN))

    SINR = np.zeros((N_UE, N_OP, NN))
    UE_power = np.zeros((N_UE, N_OP, NN))
    Int_power = np.zeros((N_UE, N_OP, NN))

    for no in range(N_OP):
        for nu in range(N_UE):
            server_bs = BS_UE_assoc[nu, no]

            # Intended signal power.
            UE_power[nu, no, :] = power[nu, server_bs, no, :]

            # Interference from other BSs.
            inferfering_bss = np.delete(np.arange(N_BS), server_bs)
            for interfering_bs in inferfering_bss:
                interfering_users = np.where(BS_UE_assoc[:, no] == interfering_bs)[0]
                tmp = 0.0
                for interfering_user in interfering_users:
                    tmp += np.abs(np.einsum('i...,i...->...', beamforming_vector[:, interfering_user, interfering_bs, no, :], total_signal[:, 0, nu, interfering_bs, no, :]))**2
                Int_power[nu, no, :] += tmp / interfering_users.size

            # SINR.
            SINR[nu, no, :] = 10 * np.log10(UE_power[nu, no, :] / (Int_power[nu, no, :] + sigma_n2))
    
    return SINR, UE_power, Int_power


def calculate_rate(total_SINR: np.ndarray, type: str = "sum") -> np.ndarray:
    """
    Calculate the sum/min/mean rate based on the total SINR. Because of the time-orthogonality
    between users, the sum rate is divided by the number of users.

    Args:
        total_SINR (users, NN): SINR (dB).
        type: Type of rate to calculate ("sum", "min", "mean").
    
    Returns:
        Sum/min/mean rate (NN): Sum rate (bits/s/Hz).
    """
    funcs = {
        "sum": np.sum,
        "min": np.min,
        "mean": np.mean
    }

    if type not in funcs:
        raise ValueError("Invalid rate type. Choose from 'sum', 'min', or 'mean'.")

    sinr_linear = 10 ** (total_SINR / 10)
    rates = np.log2(1 + sinr_linear)

    return funcs[type](rates, axis=0) / total_SINR.shape[0]
