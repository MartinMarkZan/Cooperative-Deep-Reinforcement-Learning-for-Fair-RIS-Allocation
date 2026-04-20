from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF

from src.config import Config
from src.sim.valuation import calculate_rate


plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    })


if __name__ == "__main__":
    cfg = Config()
    save_folder = Path("results") / f"{cfg.model_name}_{cfg.f_name}"
    data_path = save_folder / f"fairness_results_{cfg.eval_episodes}eps.npz"

    data = np.load(data_path)
    
    gammas = data["gammas"] # (n_gammas,)
    SINR = data["SINR"] # (n_gammas, eval_episodes, N_UE, N_OP, NN)
    BS_UE_assoc = data["BS_UE_assoc"] # (n_gammas, eval_episodes, N_UE, N_OP)
    BS_RIS_alloc = data["BS_RIS_alloc"] # (n_gammas, eval_episodes, N_RIS, N_OP)
    costs = data["costs"] # (n_gammas, eval_episodes, n_agents, N_OP)

    n_agents = cfg.N_BS

    n_gammas = gammas.shape[0]

    sum_rates_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_BS, cfg.N_OP))
    min_rates_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_BS, cfg.N_OP))
    mean_rates_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_BS, cfg.N_OP))
    jains_index_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_OP))
    costs_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_BS, cfg.N_OP))
    n_riss_allocated_aggregator = np.zeros((n_gammas, cfg.eval_episodes, cfg.N_BS, cfg.N_OP))

    for gamma_idx, gamma in enumerate(gammas):
        for ep_counter in range(cfg.eval_episodes):
            episode_SINR = SINR[gamma_idx, ep_counter]
            episode_BS_UE_assoc = BS_UE_assoc[gamma_idx, ep_counter]
            episode_BS_RIS_alloc = BS_RIS_alloc[gamma_idx, ep_counter]
            episode_costs = costs[gamma_idx, ep_counter]
            for nb in range(n_agents):
                ue_indices, op_indices = np.where(episode_BS_UE_assoc == nb)
                op_index = op_indices[0] # A BS can only be associated to one operator.
                sum_rate = np.mean(calculate_rate(episode_SINR[ue_indices, op_index, :], type="sum"))
                sum_rates_aggregator[gamma_idx, ep_counter, nb, op_index] = sum_rate
                min_rate = np.mean(calculate_rate(episode_SINR[ue_indices, op_index, :], type="min"))
                min_rates_aggregator[gamma_idx, ep_counter, nb, op_index] = min_rate
                mean_rate = np.mean(calculate_rate(episode_SINR[ue_indices, op_index, :], type="mean"))
                mean_rates_aggregator[gamma_idx, ep_counter, nb, op_index] = mean_rate

                costs_aggregator[gamma_idx, ep_counter, nb, op_index] = episode_costs[nb, op_index]
                n_riss_allocated_aggregator[gamma_idx, ep_counter, nb, op_index] = np.sum(episode_BS_RIS_alloc == nb)

            bs_performances = mean_rates_aggregator[gamma_idx, ep_counter, :, op_index]
            jains_index = (np.sum(bs_performances)**2) / (bs_performances.size * np.sum(bs_performances**2))
            jains_index_aggregator[gamma_idx, ep_counter, op_index] = jains_index

    sum_rates = np.sum(np.mean(sum_rates_aggregator, axis=(1, 3)), axis=1)
    sum_rates_BS0 = np.mean(sum_rates_aggregator, axis=(1, 3))[:, 0]
    sum_rates_BS1 = np.mean(sum_rates_aggregator, axis=(1, 3))[:, 1]
    min_rates = np.mean(min_rates_aggregator, axis=(1, 3))
    mean_rates = np.mean(mean_rates_aggregator, axis=(1, 3))
    jains_index = np.mean(jains_index_aggregator, axis=(1, 2))
    costs = np.mean(costs_aggregator, axis=(1, 3))
    n_riss_allocated = np.mean(n_riss_allocated_aggregator, axis=(1, 3))

    data = pd.concat([pd.Series(gammas), 
    pd.Series(sum_rates), pd.Series(sum_rates_BS0), pd.Series(sum_rates_BS1), pd.Series(min_rates[:, 0]), pd.Series(min_rates[:, 1]), 
    pd.Series(mean_rates[:, 0]), pd.Series(mean_rates[:, 1]), 
    pd.Series(jains_index), pd.Series(costs[:, 0]), pd.Series(costs[:, 1]), pd.Series(n_riss_allocated[:, 0]), 
    pd.Series(n_riss_allocated[:, 1])], axis=1)
    data.columns = ["Gamma", "Sum rate", "Sum rate BS0", "Sum rate BS1", "Min rate BS0", "Min rate BS1", "Mean rate BS0", "Mean rate BS1",  
        "Jain's index", "Cost BS0", "Costs BS1", "#RISs allocated BS0", "#RIS allocated BS1"]
    data.index = gammas
    data.map('{:.4f}'.format).to_csv(f"{save_folder}/Data_gamma_{mean_rates_aggregator.shape[1]}.csv")

    plt.figure(figsize=(7, 5))
    linestyles = ['-', '--', '-.', ':']
    for gamma_idx, gamma in enumerate(gammas[::2]):
        ecdf_rewards = ECDF(np.ndarray.flatten(mean_rates_aggregator[gamma_idx, :, 0, :]))
        plt.plot(ecdf_rewards.x, ecdf_rewards.y, label=f"Gamma {gamma}", linestyle=linestyles[gamma_idx % 4], color=plt.colormaps["Dark2"].colors[gamma_idx % 8])
        ecdf_rewards = ECDF(np.ndarray.flatten(mean_rates_aggregator[gamma_idx, :, 1, :]))
        plt.plot(ecdf_rewards.x, ecdf_rewards.y, label=f"Gamma {gamma}", linestyle=linestyles[gamma_idx % 4], color=plt.colormaps["Dark2"].colors[gamma_idx % 8])
    plt.xlabel('Average mean rate per geometry per base station (bps/Hz)')
    plt.ylabel('Empirical cumulative distribution')
    plt.legend()
    plt.savefig(f'{save_folder}/Mean_rate_gamma_{mean_rates_aggregator.shape[1]}.jpg', dpi=300, bbox_inches='tight')
    plt.clf()

    # (n_gammas, eval_episodes, N_UE, N_OP, NN)
    plt.figure(figsize=(7, 5))
    linestyles = ['-', '--', '-.', ':']
    for gamma_idx in range(n_gammas)[::2]: # Plot only for a subset of gammas to avoid clutter.
        episode_BS_UE_assoc = BS_UE_assoc[gamma_idx, 0]
        for nb in range(n_agents):
            ue_indices, op_indices = np.where(episode_BS_UE_assoc == nb)
            op_index = op_indices[0] # A BS can only be associated to one operator.
            episode_SINR = np.mean(SINR[gamma_idx, 0, ue_indices, op_index, :], axis=1)
            SINR_distribution = ECDF(np.ndarray.flatten(episode_SINR))
            plt.plot(SINR_distribution.x, SINR_distribution.y, label=f"Gamma {gammas[gamma_idx]}", linestyle=linestyles[gamma_idx % 4], color=plt.colormaps["Dark2"].colors[gamma_idx % 8])

    plt.xlabel('SINR (dB)')
    plt.ylabel('Empirical cumulative distribution')
    plt.legend()
    plt.savefig(f'{save_folder}/SINR_distribution_gamma_{mean_rates_aggregator.shape[1]}.jpg', dpi=300, bbox_inches='tight')
    plt.clf()