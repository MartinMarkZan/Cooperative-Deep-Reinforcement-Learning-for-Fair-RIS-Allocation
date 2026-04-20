import numpy as np
import functools
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src import Config
from src.sim.channels import direct_PL, RIS_PL, direct_channel_vectors, RIS_channel_vectors
from src.sim.geometry import gen_POS_cell_edge, gen_POS_around_BS
from src.sim.valuation import calculate_fairness_weights, est_values_pos

class RISAuctionEnv(ParallelEnv):
    """Parallel PettingZoo environment for RIS auction among base stations."""
    metadata = {"render_modes": ["human"], "name": "RISAuctionEnv"}

    def __init__(self, cfg: Config, render_mode: str | None = None) -> None:
        """Store config and derive immutable environment constants/metadata."""
        self.cfg = cfg

        # ---- Reproducibility ----
        self.seed = self.cfg.seed

        # ---- Topology / geometry ----
        self.N_OP = self.cfg.N_OP
        self.N_UE = self.cfg.N_UE
        self.N_RIS = self.cfg.N_RIS
        self.N_BS = self.cfg.N_BS
        self.M_UE = self.cfg.M_UE
        self.M_RIS = self.cfg.M_RIS
        self.M_BS = self.cfg.M_BS
        self.ROI_size = self.cfg.ROI_size
        self.one_sided = False

        # ---- Reinforcement learning ----
        self.gamma = self.cfg.gamma
        self.beta = self.cfg.beta
        self.agents = ["BS" + str(r) for r in range(self.N_BS)]
        self.possible_agents = ["BS" + str(r) for r in range(self.N_BS)]
        self.normalization_factor = np.zeros(len(self.agents))
        
        # ---- Auction / economics ----
        self.start_price = self.cfg.start_price
        self.increment = self.cfg.increment
        self.max_budget = {self.agents[i]: self.cfg.budget[i] for i in range(self.N_BS)}
        self.acc_cost = {agents: 0 for agents in self.agents}

        # ---- Testing (script-level) ----
        self.NN = self.cfg.NN

        # ---- Channel ----
        self.K = self.cfg.K
        self.N0 = self.cfg.N0
        self.F = self.cfg.F
        self.Bs = self.cfg.Bs
        self.Ps = self.cfg.Ps
        self.fc = self.cfg.fc
        self.sf = self.cfg.sf
        self.ps_lin = self.cfg.ps_lin
        self.lam = self.cfg.lam
        self.sigma_n2 = self.cfg.sigma_n2
        self.power_alloc = np.zeros((self.N_BS, self.N_RIS, self.N_UE))

        # ---- Rendering ----
        self.show_plot = self.cfg.show_plot
        self.render_mode = render_mode

        # ---- Others variables ----
        self.delta = 0 # This is used as a distance between continuous ranges and "do not use" indicators.
        self.delta_val = 0


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Box:
        """Observation space consisting of normalized value of each RIS, normalized price, remaining budget and fairness weights."""
        low = np.concatenate([np.full(self.N_RIS, -1.0), np.zeros(3)], dtype=np.float32)
        high = np.concatenate([np.ones(self.N_RIS + 2), [len(self.agents)]], dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=(self.N_RIS + 3,))


    def observe(self, agent):
        """Return current observation of this agent."""
        return np.array(self.observations[agent])
    

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.MultiDiscrete:
        """Agents choose, per RIS, whether to accept the current price (1) or not (0)."""
        return spaces.MultiDiscrete(np.ones(self.N_RIS) * 2)


    def render(self) -> None:
        """Render current auction state by printing info."""
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif not hasattr(self, "state"):
            raise Exception("Environment must be reset before it can be rendered.")
        else:
            for agent in self.agents:
                print(f"Agent {agent}: {self.state[agent]}")


    def close(self) -> None:
        """Close the environment."""
        pass


    def reset(self, seed: int | None = None, options: dict | None = None
        ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """
        Reset the environment to its initial state for a new episode.
        Initializes Agents, RIS assignments, price, budget, and computes base values.
        
        Args:
            seed: Seed for RNG. If None, uses numpy’s nondeterministic seed.

        Returns:
            observations: Observations for each agent.
            infos: Empty dict with agents as keys.
        """
        # RNG and bookkeeping.
        self.rng = np.random.default_rng(seed)
        self.num_moves = 0

        # RIS assignment at base station level (used for bidding).
        self.assigned_RIS = np.zeros(self.N_RIS, dtype=bool)
        self.BS_RIS_assignment = {agent: np.zeros(self.N_RIS) for agent in self.agents}
        # Used to check if RIS-assignment changed, such that values need to be updated.
        self.prior_RIS_assignment = {agent: np.zeros(self.N_RIS) for agent in self.agents}

        # RIS assignment at operator level (kept for compatibility with operator-based logic).
        self.RIS_assignment = {agent: np.zeros(self.N_RIS) for agent in self.agents}
        
        self.price = self.start_price
        self.budget = self.max_budget.copy()
        self.acc_cost = {agent: 0 for agent in self.agents}

        # Reset environment geometry and get per-agent RIS value estimations.
        obs_values = -1
        while obs_values == -1: # Error code if there is at least one BS with no associated UE
            obs_values = self.reset_geometry()
        
        # Calculate fairness weights.
        fairness_weights = calculate_fairness_weights(self.current_value, self.agents, self.gamma)
        # Build normalized observations for each agent: [normalized RIS values, normalized price, normalized budget].
        observations = {
            agent: np.append(obs_values[agent], [self.price / self.max_budget[agent] * (1 - self.delta), 1, fairness_weights[agent]])
            for agent in self.agents
        }

        self.state = observations
        self.terminations = {agent: np.False_ for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos


    def reset_geometry(self) -> dict[str, np.ndarray]:
        """
        Initializes the geometry and baseline valuations for a new episode.

        Returns:
            obs_vales: Per-agent array of normalized per-RIS values in [0, 1].
        """
        # Generate positions.
        self.UE_pos, self.BS_pos, self.RIS_pos = gen_POS_around_BS(self.N_OP, 
            self.N_BS, self.N_UE, self.N_RIS, self.ROI_size, self.rng, 
            one_sided=self.one_sided, bs_probs=np.array([0.75, 0.25]))
        
        # Calculate path losses.
        self.PL_UE_BS, self.LOS_UE_BS, angles_UE_BS, angles_BS_UE = direct_PL(
            self.UE_pos, self.BS_pos, self.rng, self.lam, self.sf)
        (self.PL_RIS_BS, self.LOS_RIS_BS, angles_RIS_BS, angles_BS_RIS, self.PL_RIS_UE, 
         self.LOS_RIS_UE, angles_RIS_UE) = RIS_PL(self.UE_pos, self.BS_pos, 
            self.RIS_pos, self.rng, self.lam, self.sf)

        # Calculate channel gains.
        self.pow_ue_bs = 10**(-self.PL_UE_BS/10)
        self.pow_ris_bs = 10**(-self.PL_RIS_BS/10)
        self.pow_ris_ue = np.swapaxes(10**(-self.PL_RIS_UE/10), 0, 1)
        # Associate users with strongest BS.
        self.BS_UE_assoc = np.argmax(self.pow_ue_bs, axis=1) # Shape: (N_UE, N_OP).

        # Error code if there is at least one BS with no associated UE.
        for nb in range(self.N_BS):
            if not np.array(self.BS_UE_assoc == nb).any():
                return -1 

        # Get channel vectors.
        self.UE_BS_channel, self.BS_UE_channel, self.channels_BS_UE = direct_channel_vectors(
            self.M_BS, self.M_UE, self.N_UE, self.N_OP, self.N_BS, self.NN, self.rng, 
            self.lam, angles_UE_BS, angles_BS_UE)

        # Get RIS channel vectors -- needed for interference estimation of directional links.
        (self.channels_BS_RIS, self.channels_RIS_UE, self.RIS_BS_channel, self.BS_RIS_channel, 
         self.RIS_UE_channel) = RIS_channel_vectors(self.M_RIS, self.M_BS, self.M_UE, self.N_UE, 
            self.N_OP, self.N_BS, self.N_RIS, self.NN, self.rng, self.lam, angles_RIS_BS, 
            angles_BS_RIS, angles_RIS_UE)
        
        # NOT USED
        # Calculate inter-BS-interference level at RISs due to similarity of array response vectors.
        self.IBI = np.abs(np.einsum('mrbo,mrco->bcro', self.RIS_BS_channel.conj(), self.RIS_BS_channel))

        # Base values when no RISs are assigned.
        self.current_value = {}

        # TODO: Generalize to multiple operators.
        no = 0
        for agent_idx, agent in enumerate(self.agents):
            rates, SINR_values, _, _, pow_alloc = est_values_pos(self.N_BS, self.N_RIS, self.N_UE, self.M_BS, self.M_RIS, self.M_UE, agent_idx, self.ps_lin, self.sigma_n2,
                                        self.pow_ue_bs[:, :, no], self.pow_ris_bs[:, :, no], self.pow_ris_ue[:, :, no],
                                        self.BS_UE_assoc[:, no], np.zeros((1, self.N_RIS)), self.IBI[:, :, :, no], self.LOS_UE_BS[:, :, no], self.LOS_RIS_BS[:, :, no],
                                        self.LOS_RIS_UE[:, :, no], self.K)
            self.power_alloc[agent_idx] = pow_alloc[:, :, 0]
            self.current_value[agent] = np.mean(rates)

        # Force computation of initial values.
        initial_actions = {agent: np.ones(self.N_RIS, dtype=bool) for agent in self.agents}
        obs_values = self.update_value(initial_actions, first_call=True)

        return obs_values


    def step(self, actions: dict[str, np.ndarray]
        ) -> tuple[dict[str, np.ndarray], 
                   dict[str, float], 
                   dict[str, bool], 
                   dict[str, bool], 
                   dict[str, dict]]:
        """
        Advance the auction by one round.

        Args:
            actions: For each agent, a binary vector indicating a bid (1) or no 
            bid (0) for each RIS.

        Returns:
            observations: Next observation for each agent.
            rewards: Reward per agent for this step.
            terminations: Whether each agent is done (price exceeds budget or no
                feasible RIS remains).
            truncations: Always False for all agents (no time-limit truncation here).
            infos: Metadata for each agent.
        """
        self.num_moves += 1
        infos = {agent: {} for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # Mask invalid actions:
        # Bidding on already assigned RISs and RISs not bid before -- activity rule,
        # or if BS is terminated.
        for agent in self.agents:
            if self.terminations[agent]: # If agent is terminated, no bids are accepted.
                actions[agent] = np.zeros(self.N_RIS, dtype=bool)
            else:
                unavail_RIS = self.state[agent][0:self.N_RIS] == -1.0
                actions[agent] = actions[agent].astype(bool)
                actions[agent][unavail_RIS] = False
        
        # Resolve single-bidder RIS winners.
        bid_matrix = np.stack([actions[agent] for agent in self.agents], axis=0)
        # If there is just one bidder --> winner.
        auction_winner_bool = np.logical_and(~self.assigned_RIS, np.sum(bid_matrix, axis=0) == 1)
        # Index of agent that was single bidder.
        auction_winner_ind = np.argmax(bid_matrix, axis=0)
        self.assigned_RIS = np.logical_or(self.assigned_RIS, auction_winner_bool)
        
        # Apply wins, update budgets and rewards.
        rewards = {agent: {} for agent in self.agents}
        temp_assignment = np.zeros(self.N_RIS, dtype=bool) # Perform a consistency check.
        # TODO: Generalize for multiple operators.
        no = 0
        for agent_idx, agent in enumerate(self.agents):
            self.prior_RIS_assignment[agent] = self.BS_RIS_assignment[agent]

            won_ris = np.logical_and(auction_winner_ind == agent_idx, auction_winner_bool)        
            self.BS_RIS_assignment[agent] = np.logical_or(self.BS_RIS_assignment[agent], won_ris)
            temp_assignment = np.logical_or(temp_assignment, self.BS_RIS_assignment[agent])
            
            if won_ris.any():
                normalized_costs = won_ris.sum() * (self.price / self.max_budget[agent])
                self.acc_cost[agent] += normalized_costs

                # Get the value of the current allocation.
                RIS_alloc = np.zeros((1, self.N_RIS))
                RIS_alloc[0, :] = self.BS_RIS_assignment[agent]

                prior_value = self.current_value[agent]
                rates, SINR_values, _, _, pow_alloc = est_values_pos(self.N_BS, self.N_RIS, self.N_UE, self.M_BS, self.M_RIS, self.M_UE, agent_idx, self.ps_lin, self.sigma_n2,
                                                        self.pow_ue_bs[:, :, no], self.pow_ris_bs[:, :, no], self.pow_ris_ue[:, :, no],
                                                        self.BS_UE_assoc[:, no], RIS_alloc, self.IBI[:, :, :, no], self.LOS_UE_BS[:, :, no], self.LOS_RIS_BS[:, :, no],
                                                        self.LOS_RIS_UE[:, :, no], self.K)
                self.power_alloc[agent_idx] = pow_alloc[:, :, 0]
                self.current_value[agent] = np.mean(rates)
                value_won_RIS = self.current_value[agent] - prior_value
            else:
                normalized_costs = 0
                value_won_RIS = 0
            
            infos[agent]["value_won_RIS"] = value_won_RIS
            infos[agent]["costs"] = normalized_costs

            bid_costs = actions[agent].sum() * self.price / self.max_budget[agent]
            overshoot = max(bid_costs - self.budget[agent] / self.max_budget[agent], 0)

            infos[agent]["R1"] = np.sum(self.state[agent][0:self.N_RIS][actions[agent]])
            infos[agent]["R2"] = self.beta * bid_costs
            infos[agent]["R3"] = self.beta * 2 * overshoot
            rewards[agent] = infos[agent]["R1"] - self.state[agent][self.N_RIS+2] * (infos[agent]["R2"] + infos[agent]["R3"])

            self.budget[agent] = max(self.budget[agent] - won_ris.sum() * self.price, 0)
            
        if (temp_assignment != self.assigned_RIS).any(): 
            raise Exception("Something inconsistent with RIS assignment")
        
        # Update values of remaining RISs.
        obs_values = self.update_value(actions)

        # Increase price for next round.
        self.price += self.increment

        # Update observations.
        observations = {agent: {} for agent in self.agents}
        # Calculate fairness weights.
        fairness_weights = calculate_fairness_weights(self.current_value, self.agents, self.gamma) 
        for agent in self.agents:
            RIS_values = obs_values[agent]
            norm_price = self.price / self.max_budget[agent] * (1 - self.delta)
            if norm_price > (1 - self.delta): # Price higher than budget.
                norm_price = 1
            observations[agent] = np.append(RIS_values, norm_price)
            
            norm_budget = self.budget[agent] / self.max_budget[agent]
            observations[agent] = np.append(observations[agent], norm_budget)

            observations[agent] = np.append(observations[agent], fairness_weights[agent])

            if (RIS_values[self.assigned_RIS] > 0).any():
                raise Exception("Something inconsistent with available RIS") 
        
        self.state = observations

        # Check terminations.
        for agent in self.agents:
            # TODO: Known error: 0.05 + 0.05 + ... + 0.05 (7 times) > 0.35,
            # so agent will be kicked out of the auction.
            price_too_high = self.price > self.budget[agent]
            has_any_ris = (observations[agent][0:self.N_RIS] > 0).any()
            done = price_too_high or (not has_any_ris)
            self.terminations[agent] = done

        if self.render_mode == "human":
            self.render()

        infos['Price'] = self.price
        infos['OP_RIS_Assignment'] = self.RIS_assignment
        infos['BS_RIS_Assignment'] = self.BS_RIS_assignment
        
        return observations, rewards, self.terminations, truncations, infos


    def update_value(self, actions: dict[str, np.ndarray], first_call: bool = False
        ) -> dict[str, np.ndarray]:
        """
        Recomputes per-agent normalized valuations for currently available RISs.

        Args:
            actions: For each agent, a binary vector indicating a bid (1) or no 
                bid (0) for each RIS.
            first_call: If True, forces valuation computation and calculates 
                normalization factor.

        Returns:
            obs_values: For each agent, the updated array of normalized per-RIS values.
        """
        obs_values = {}

        # TODO: Generalize to multiple operators.
        no = 0
        
        for agent_idx, agent in enumerate(self.agents):
            changed = np.any(self.BS_RIS_assignment[agent] != self.prior_RIS_assignment[agent])
            unavailable = self.assigned_RIS | ~actions[agent].astype(bool)
            candidates = np.nonzero(~unavailable)[0]

            # Case 1: no recomputation needed.
            if not (first_call or changed) or (candidates.size == 0):
                values = self.state[agent][0:self.N_RIS]
                values[unavailable] = -1.0
                obs_values[agent] = values
                continue

            # Case 2: recomputation needed.
            # Build candidate RIS allocations (one-hot plus current).
            current_alloc = self.BS_RIS_assignment[agent].astype(bool)
            add_one_hot = np.zeros((len(candidates), self.N_RIS), dtype=bool)
            add_one_hot[np.arange(len(candidates)), candidates] = True
            RIS_allocs = add_one_hot | current_alloc

            # Evaluate.
            rates, SINR_values, _, _, _ = est_values_pos(self.N_BS, self.N_RIS, self.N_UE, self.M_BS, self.M_RIS, self.M_UE, agent_idx, self.ps_lin, self.sigma_n2,
                            self.pow_ue_bs[:, :, no], self.pow_ris_bs[:, :, no], self.pow_ris_ue[:, :, no],
                            self.BS_UE_assoc[:, no], RIS_allocs, self.IBI[:, :, :, no], self.LOS_UE_BS[:, :, no], self.LOS_RIS_BS[:, :, no],
                            self.LOS_RIS_UE[:, :, no], self.K)
            gains = np.mean(rates, axis=0)
            
            gains -= self.current_value[agent]
            gains = np.maximum(gains, 0) # ReLu: We only care about positive rewards.

            # Simple
            if np.max(gains) != 0:
                gains /= np.max(gains)

            # Place into RIS vector.
            agent_values = np.zeros(self.N_RIS)
            agent_values[candidates] = gains
            agent_values[unavailable] = -1.0
            obs_values[agent] = agent_values

        return obs_values


    def _check_user_distributions(self, n_geometries=200) -> None:
        custom_rng = np.random.default_rng(2)
        average_users_per_bs = []
        for _ in range(n_geometries):
            UE_pos, BS_pos, _ = gen_POS_around_BS(self.N_OP, 
                self.N_BS, self.N_UE, self.N_RIS, self.ROI_size, custom_rng, one_sided=True, bs_probs=np.array([0.85, 0.15]))
            PL_UE_BS, _, _, _ = direct_PL(
                UE_pos, BS_pos, custom_rng, self.lam, self.sf)
            pow_ue_bs = 10**(-PL_UE_BS/10)
            BS_UE_assoc = np.argmax(pow_ue_bs, axis=1) # Shape: (N_UE, N_OP).
            users_per_bs = []
            for nb in range(self.N_BS):
                users_per_bs.append(np.sum(BS_UE_assoc == nb))
            average_users_per_bs.append(users_per_bs)
        average_users_per_bs = np.mean(np.array(average_users_per_bs), axis=0)
        print("Average users per BS over", n_geometries, "geometries:", average_users_per_bs)
