import numpy as np
import gymnasium as gym
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from src.sim.beamforming import compute_beamforming_vector
from src.sim.channels import RIS_alloc_response, RIS_channel, direct_channel
from src.sim.valuation import calculate_rate, compute_power_allocation, get_SINR

class RewardLogger(BaseCallback):
    """Custom callback for plotting additional values in Tensorboard."""

    def __init__(self, eval_env: Union[gym.Env, VecEnv], verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env


    def _on_step(self) -> bool:
        eval_episodes = 20
        env = self.eval_env
        agents = env.agents
        n_agents = len(agents)

        # Episode accumulators (sum within current episode).
        acc = {
            "reward": np.zeros(n_agents),
            "value_won_RIS": np.zeros(n_agents),
            "r1": np.zeros(n_agents),
            "r2": np.zeros(n_agents),
            "r3": np.zeros(n_agents),
        }

        # Logs across episodes.
        ep_rewards = []
        ep_values_won_RIS = []
        ep_r1, ep_r2, ep_r3 = [], [], []
        ep_final_values, ep_final_costs, ep_final_n_riss_allocated = [], [], []
        total_sum_rates = []
        sum_rate_improvement = []

        # Initial state.
        ep_counter = 0
        observe, infos = env.reset(ep_counter)

        def reset_acc():
            for k in acc: acc[k].fill(0)

        def on_episode_end():
            """Compute per-episode metrics at episode termination and push to logs."""
            # Build RIS allocation views.
            RIS_alloc = np.zeros((env.N_RIS,))
            # Sentinel operator index for "unassigned".
            BS_RIS_alloc = np.full((env.N_RIS,), n_agents, dtype=int)
            for agent_idx, agent in enumerate(agents):
                BS_RIS_alloc[env.BS_RIS_assignment[agent]] = agent_idx

            # Channels & beamforming.
            direct_channel_BS_UE = direct_channel(env.N_OP, env.N_BS, env.N_UE, 
                env.M_BS, env.M_UE, env.NN, env.K, env.LOS_UE_BS, 
                env.pow_ue_bs, env.UE_BS_channel, env.BS_UE_channel, 
                env.channels_BS_UE, env.BS_UE_assoc)
            RIS_resp = RIS_alloc_response(env.RIS_UE_channel, 
                env.RIS_BS_channel, RIS_alloc, BS_RIS_alloc, env.BS_UE_assoc, 
                env.rng)
            RIS_channel_BS_UE = RIS_channel(env.N_OP, env.N_BS, env.N_RIS, 
                env.N_UE, env.M_BS, env.M_RIS, env.M_UE, env.NN, env.K, 
                env.LOS_RIS_UE, env.LOS_RIS_BS, env.RIS_BS_channel, 
                env.BS_RIS_channel, env.channels_BS_RIS, env.RIS_UE_channel, 
                env.channels_RIS_UE, env.BS_UE_assoc, RIS_resp, env.pow_ris_bs, 
                env.pow_ris_ue, env.rng)
            no = 0 # Only one operator right now.
            for nb in range(env.N_BS):
                current_power_allocation = compute_power_allocation(env.N_RIS, env.N_UE, env.M_BS,
                    env.M_RIS, nb, env.ps_lin, env.pow_ris_bs[:, :, no], env.pow_ris_ue[:, :, no], env.BS_UE_assoc[:, no],
                    env.BS_RIS_assignment[agents[nb]], env.LOS_RIS_UE[:, :, no], env.K)
                env.power_alloc[nb, :, :] = current_power_allocation
            beamforming_vector = compute_beamforming_vector(env, BS_RIS_alloc, 
                    RIS_channel_BS_UE)
            direct_SINR, _, _ = get_SINR(env.N_OP, env.N_BS, env.N_UE, 
                env.sigma_n2, env.NN, env.BS_UE_assoc, beamforming_vector, 
                direct_channel_BS_UE)
            total_SINR, _, _ = get_SINR(env.N_OP, env.N_BS, env.N_UE, 
                env.sigma_n2, env.NN, env.BS_UE_assoc, beamforming_vector, 
                direct_channel_BS_UE, RIS_channel_BS_UE)

            # Sum-rate metrics per BS.
            for nb in range(env.N_BS):
                ue_idx, op_idx = np.where(env.BS_UE_assoc == nb)
                op = op_idx[0]  # BS belongs to one operator.
                total_rates = calculate_rate(total_SINR[ue_idx, op, :], type="sum")
                direct_rates = calculate_rate(direct_SINR[ue_idx, op, :], type="sum")
                total_sum_rates.append(total_rates)
                sum_rate_improvement.append(total_rates / (direct_rates + 1e-6) - 1.0)

            # Push per-episode arrays.
            ep_rewards.append(acc["reward"].copy())
            ep_values_won_RIS.append(acc["value_won_RIS"].copy())
            ep_r1.append(acc["r1"].copy())
            ep_r2.append(acc["r2"].copy())
            ep_r3.append(acc["r3"].copy())

            # Final value/costs per agent.
            vals = np.empty(n_agents)
            costs = np.empty(n_agents)
            n_riss_allocated = np.empty(n_agents)
            for agent_idx, agent in enumerate(agents):
                vals[agent_idx] = env.current_value[agent]
                costs[agent_idx] = env.acc_cost[agent]
                n_riss_allocated[agent_idx] = np.sum(env.BS_RIS_assignment[agent])
            ep_final_values.append(vals)
            ep_final_costs.append(costs)
            ep_final_n_riss_allocated.append(n_riss_allocated)

        # Main loop over episodes.
        while ep_counter < eval_episodes:
            obs = np.array([observe[agent] for agent in agents])
            actions, _ = self.model.predict(obs, deterministic=True)
            actions = {agent: actions[agent_idx, :] for agent_idx, agent in enumerate(agents)}
            observe, rewards, dones, _, infos = env.step(actions)

            # Accumulate per-agent signals.
            all_done = True
            for agent_idx, agent in enumerate(agents):
                acc["reward"][agent_idx] += rewards[agent]
                acc["value_won_RIS"][agent_idx] += infos[agent]["value_won_RIS"]
                acc["r1"][agent_idx] += infos[agent]["R1"]
                acc["r2"][agent_idx] += infos[agent]["R2"]
                acc["r3"][agent_idx] += infos[agent]["R3"]
                all_done = all_done and dones[agent]

            # End of episode.
            if all_done:
                on_episode_end()
                reset_acc()
                ep_counter += 1
                observe, infos = env.reset(ep_counter)

        # Aggregate means across episodes.
        mean_reward = np.mean(ep_rewards)
        mean_value_won_RIS = np.mean(ep_values_won_RIS)
        mean_r1 = np.mean(ep_r1)
        mean_r2 = np.mean(ep_r2)
        mean_r3 = np.mean(ep_r3)
        mean_final_costs = np.mean(ep_final_costs)
        mean_final_values = np.mean(ep_final_values)
        mean_final_n_riss_allocated = np.mean(ep_final_n_riss_allocated)
        mean_sum_rate = np.mean(total_sum_rates)
        mean_sum_rate_impr = np.mean(sum_rate_improvement)

        # Log.
        self.logger.record("custom/mean_reward", mean_reward)
        self.logger.record("custom/mean_value_won_RIS", mean_value_won_RIS)
        self.logger.record("custom/mean_costs", mean_final_costs)
        self.logger.record("custom/mean_r1", mean_r1)
        self.logger.record("custom/mean_r2", mean_r2)
        self.logger.record("custom/mean_r3", mean_r3)
        self.logger.record("custom/mean_final_values", mean_final_values)
        self.logger.record("custom/mean_n_won_RIS", mean_final_n_riss_allocated)
        self.logger.record("custom/mean_sum_rate", mean_sum_rate)
        self.logger.record("custom/sum_rate_improvement", mean_sum_rate_impr)
        return True
