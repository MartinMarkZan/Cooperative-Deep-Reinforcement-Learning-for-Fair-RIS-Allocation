from typing import Optional
import numpy as np

class Config:
    quick_test = False # Overwrite values here and in "test" for quick testing.

    # ---- Reproducibility ----
    seed: int = 2

    # ---- Topology / geometry ----
    N_OP: int = 1       # number of operators
    N_UE: int = 20      # number of users
    N_RIS: int = 10     # number of reconfigurable intelligent surfaces
    N_BS: int = 2       # number of base stations
    M_UE: int = 1       # number of antennas at UEs
    M_RIS: int = 250    # number of RIS elements
    M_BS: int = 50      # number of antennas at BS
    ROI_size: np.ndarray = np.array([[-10, -10], [10, 10]]) * 5  # meters size of ROI

    # ---- Reinforcement learning ----
    gamma : float = 1.0         # fairness factor, to turn it off set to 0
    beta: float = 1.0           # bid intensity, to turn it off set to 1
    model_name: str = f"2025_11_20"   # name of the RL model # _beta{int(beta)}

    # ---- Auction / economics ----
    budget: np.array = 1 * np.ones(N_BS)        # available budget for bidding (can be different for operators)
    increment: float = 0.05                     # price increment from one round to the next
    start_price: float = increment              # price at beginning of auction

    # ---- Training (script-level) ----
    num_vec_envs: int = 4               # number of vector environments
    n_steps: int = 2048                 # number of steps for each environment per update
    timesteps: int = int(3e6)           # total number of steps to train on

    # ---- Testing (script-level) ----
    budget_test: bool = False        # test different budgets
    performance_test: bool = False   # test performance
    fairness_test: bool = True         # test different fairness factors
    brute_force_test: bool = False     # test brute-force optimal allocation
    eval_episodes: int = 200        # number of macroscopic fading realizations for evaluation
    NN: int = 20                    # number of microscopic fading realizations

    # ---- Channel ----
    K: np.array = np.array([1e2, 3*1e0]) # Rician K-factor for LOS and NLOS
    N0: float = -174.0       # dBm/Hz noise PSD
    F: float = 6             # dB noise figure
    Bs: float = 15e3         # subcarrier bandwidth
    Ps: float = 20.0         # dBm power per subcarrier
    fc: float = 26e9         # carrier frequency
    sf: float = 10           # shadow fading variance

    # ---- Rendering ----
    show_plot: bool = False

    if quick_test:
        model_name = "test"
        n_steps = 8
        timesteps = 30
        budget_test = True
        performance_test = True
        fairness_test = True
        brute_force_test = False
        eval_episodes = 2
        NN = 2

    # ---- Derived (computed from others) ----
    ps_lin: Optional[float] = 10**((Ps - 30) / 10)              # linear power
    lam: float = 3e8 / fc                                       # wavelength
    sigma_n2: float = 10**((N0 + 10*np.log10(Bs) + F - 30)/10)  # linear noise power
    f_name = str(N_BS) + 'BS_' + str(N_UE) + 'UE_' + str(N_RIS) + 'RIS_' + str(M_RIS) + 'M' # file name
    