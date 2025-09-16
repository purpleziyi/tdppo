import numpy as np
import gymnasium as gym

class TDEnv(gym.Env):
    """
    Custom reinforcement learning environment for Technical Debt prioritization.

    Attributes:
        X (np.ndarray): Feature matrix of shape (K, d), where K = number of issues,
                        d = number of features per issue.
        meta (dict): Metadata dictionary (contains rewards, issue_ids, etc.).
        mode (str): Determines reward type ("theory" or "practice").
        max_select (int): Maximum number of issues that can be selected in one episode.
        K (int): Number of issues.
        d (int): Number of features per issue.
    """

    def __init__(self, X, meta, mode="theory", max_select=5):
        """
        Initialize the environment.

        Args:
            X (np.ndarray): Feature matrix (K x d).
            meta (dict): Metadata dictionary containing reward signals (e.g., "theory_score").
            mode (str): Reward mode, either "theory" (severity-effort score) or "practice" (developer rank).
            max_select (int): Maximum number of selections allowed in one episode.
        """

        super().__init__()
        self.max_select = max_select
        self.mode = mode
        self.X = X
        self.meta = meta

        self.K, self.d = X.shape # K= TD items quantity, d= TD features quantity

        self.action_space = gym.spaces.Discrete(self.K)  # Action: select one of the K issues
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.K, self.d+1), dtype=np.float32
        ) # Observation: feature matrix + selection mask

        self.reset()  # for initialize environment

    def reset(self,seed=None, options=None):
        """
        Reset environment to the initial state.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Extra environment options.

        Returns:
            tuple: (obs, info)
                - obs (np.ndarray): Initial observation matrix.
                - info (dict): Additional info (empty dict here).
        """
        super().reset(seed=seed)
        self.selected = np.zeros(self.K, dtype=np.int32)  # register each TD been selected(1) or not(0)
        self.n_selected = 0  # Selected quantity
        return self._obs(), {} # return obs and info(in form of empty dictionary)

    """ build observation state """
    def _obs(self):
        """
        Build the observation matrix by concatenating features with a mask column.

        Returns:
            np.ndarray: Observation of shape (K, d+1).
        """
        mask_col = (1 - self.selected).reshape(-1, 1)  # 1=Can be selected, 0=Already selected
        return np.hstack([self.X, mask_col]).astype(np.float32)  # Concatenate the original feature X and mask
    
    def step(self, action):
        """
        Execute one step in the environment by selecting a technical debt item.

        Args:
            action (int): Index of the selected item (0 <= action < K).

        Returns:
            Tuple:
                - obs (np.ndarray): Next observation state (K x (d+1) matrix).
                - reward (float): Reward value based on theory score or developer rank.
                - terminated (bool): Whether the episode has reached its natural end
                  (e.g., max_select items chosen).
                - truncated (bool): Whether the episode ended prematurely (always False here).
                - info (dict): Additional debugging information (empty in this case).
        """
        # If the selected item was already chosen → small penalty and terminate
        if self.selected[action] == 1:
            return self._obs(), -2.0, True, False, {}

        # Otherwise mark the item as selected
        self.selected[action] = 1
        self.n_selected += 1

        # Compute reward depending on mode("theory"/"practice")
        if self.mode == "theory":
            reward = self.meta["theory_score"][action]  # Use theory score
        else:
            dev_rank = self.meta["dev_rank"].get(action, None)
            reward = 1.0 / dev_rank if dev_rank is not None else -0.01

        # Termination condition: maximum number of selections reached
        terminated = (self.n_selected >= self.max_select)

        return self._obs(), float(reward), terminated, False, {}

