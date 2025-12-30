"""
QlearingAgent: Tabular Q-learning agent with epsilon-greedy exploration.

The agent maintains a Q-table as a nested dictionary and implements:
- Epsilon-greedy action selection
- Q-learning update rule
- Epsilon decay
- Save/load functionality
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pickle
from pathlib import Path

State = Tuple[int, int, int, int, int, int]  # (loc, bat, queue, load, pickup, drop)

class QLearningAgent:
    """
    Tabular Q-learning agent for discrete state-action spaces.

    Parameters
    ----------
    alpha : float
        Learning rate (step size for Q-value updates).
    gamma : float
        Discount factor for future rewards.
    epsilon : float
        Initial exploration rate for epsilon-greedy policy.
    epsilon_min : float
        Minimum exploration rate (lower bound for decay).
    epsilon_decay : float
        Multiplicative decay factor for epsilon after each episode.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None
    ):
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: Q[state][action] = value
        self.Q: Dict[State, Dict[int, float]] = {}

        # Random number generation
        self.rng = np.random.default_rng(seed)

    def _ensure_state_exist(self, state: State) -> None:
        """
        Ensure that a state exists in the Q-table.
        If not, initialize it with an empty action dict.

        Parameters
        ----------
        state : State
            State tuple (location_id, battery_level, queue_level, load_state)
        """

        if state not in self.Q:
            self.Q[state] = {}

    def _ensure_action_exists(self, state:State, action: int) -> None:
        """
        Ensure that a state-action pait exists in the Q-table.
        If not, initialize Q(state, action) to 0.0.

        Parameters
        ----------
        state : TState
            State tuple
        action : int
            Action (node_id)
        """

        self._ensure_state_exist(state)
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0

    def get_q_value(self, state: State, action: int) -> float:
        """
        Get Q-value for a state-action pair.
        Returns 0.0 if the pair has not been seen before.

        Parameters
        ----------
        state : State
            State tuple
        action : int
            Action (node_id)
        
        Returns
        -------
        q_value : float
            Q(state, action)
        """

        self._ensure_action_exists(state, action)
        return self.Q[state][action]

    def select_action(
        self,
        state: State,
        valid_actions: List[int]
    ) -> int:
        """
        Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : State
            Current state
        valid_actions : List[int]
            List of valid actions (neighbor nodes)
        
        Returns
        -------
        action : int
            Selected action (node_id)
        """

        assert len(valid_actions) > 0, "valid actions cannot be empty"

        # Epsilon-greedy: explore with probability epsilon
        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        else:
            for action in valid_actions:
                self._ensure_action_exists(state, action)
            
            q_values = [self.Q[state][action] for action in valid_actions]

            max_q = max(q_values)

            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]

            return self.rng.choice(best_actions)
    
    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_valid_actions: List[int],
        done: bool
    ) -> None:
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) ← Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        
        Parameters
        ----------
        state : Tuple[int, int, int, int]
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : Tuple[int, int, int, int]
            Next state
        next_valid_actions : List[int]
            Valid actions in next state
        done : bool
            Whether the episode terminated
        """

        self._ensure_action_exists(state, action)

        # Compute target
        if done:
            target = reward
        else:
            for next_action in next_valid_actions:
                self._ensure_action_exists(next_state, next_action)

            max_next_q = max(self.Q[next_state][next_action] for next_action in next_valid_actions)

            target =  reward + self.gamma * max_next_q
        
        # Q-learning update
        current_q = self.Q[state][action]
        self.Q[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """
        Decay epsilon by the decay factor, respecting the minimum bound.
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str) -> None:
        """
        Save teh Q-table and hyperparameters to disk.

        Parameters
        ----------
        path : str
            File path to save to "./results/models/q_table.pkl"
        """

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Package data
        data = {
            "Q": self.Q,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

        # Save with pickle
        with open(path, "wb") as f:
            pickle.dump(data, f)

        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the Q-table and hyperparameters from disk.

        Parameters
        ----------
        path : str
            File path to load from
        """

        with open(path, "rb") as f:
            data = pickle.load(f)

            # Restore data
            self.Q = data["Q"]
            self.alpha = data["alpha"]
            self.gamma = data["gamma"]
            self.epsilon = data ["epsilon"]
            self.epsilon_min = data["epsilon_min"]
            self.epsilon_decay = data["epsilon_decay"]

            print(f"Agent loaded from {path}")
            print(f" Q-table size: {len(self.Q)} states")
            print(f" Current epsilon: {self.epsilon:.4f}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Q-table.

        Returns
        -------
        stats : Dict[str, Any]
            Dictionary with Q-table statistics
        """

        num_states = len(self.Q)
        num_state_action_pairs = sum(len(actions) for actions in self.Q.values())

        # Get all Q-values
        all_q_values = [
            q for state_actions in self.Q.values()
            for q in state_actions.values()
        ]
        
        stats = {
            "num_states": num_states,
            "num_state_action_pairs": num_state_action_pairs,
            "epsilon": self.epsilon,
        }

        if all_q_values:
            stats.update({
                "q_mean": np.mean(all_q_values),
                "q_std": np.std(all_q_values),
                "q_min": np.min(all_q_values),
                "q_max": np.max(all_q_values)
            })
        
        return stats