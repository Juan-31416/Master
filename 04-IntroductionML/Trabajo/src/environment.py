"""
RoutingEnv: Discrete tabular RL environment for single-vehicle routing.

Each episode represents one transport job:
- Start at start_node (empty)
- Navigate to pickup_node (automatic loading)
- Navigate to drop_node (automatic unloading, episode ends)

State: (location_id, battery_level, queue_level, load_state)
Action: move to a neighbor node
Reward: per-step penalty + queue penalty + task completion bonus/failure penalty
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


class RoutingEnv:
    """
    Routing environment for a single industrial vehicle (AGV).
    
    Parameters
    ----------
    graph : Dict[int, List[int]]
        Adjacency list representing the plant layout.
    node_roles : Dict[str, Set[int]]
        Dictionary mapping role names to sets of node IDs.
    reward_params : Dict[str, float]
        Reward function parameters:
            - "r_step": per-step penalty (typically -1.0)
            - "r_queue_factor": queue penalty factor (typically -0.3)
            - "r_task_completion": reward for successful job completion (typically +20.0)
            - "r_failure": penalty for failures (typically -30.0)
    max_steps : int
        Maximum steps per episode before timeout.
    start_candidates : List[int]
        Candidate nodes for episode start position.
    pickup_candidates : List[int]
        Candidate nodes for pickup location.
    drop_candidates : List[int]
        Candidate nodes for drop location.
    queue_sampling : str
        How to sample queue_level at reset:
            - "uniform": sample uniformly from {0, 1, 2}
            - "fixed": use fixed_queue_level
    fixed_queue_level : Optional[int]
        If queue_sampling="fixed", use this value for all episodes.
    seed : Optional[int]
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        graph: Dict[int, List[int]],
        node_roles: Dict[str, Set[int]],
        reward_params: Dict[str, float],
        max_steps: int,
        start_candidates: List[int],
        pickup_candidates: List[int],
        drop_candidates: List[int],
        queue_sampling: str = "uniform",
        fixed_queue_level: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        # Graph structure
        self.graph = graph
        self.node_roles = node_roles
        
        
        # Reward parameters
        self.r_step = reward_params["r_step"]
        self.r_queue_factor = reward_params["r_queue_factor"]
        self.r_task_completion = reward_params["r_task_completion"]
        self.r_failure = reward_params["r_failure"]
        
        # Episode constraints
        self.max_steps = max_steps
        
        # Job generation
        self.start_candidates = start_candidates
        self.pickup_candidates = pickup_candidates
        self.drop_candidates = drop_candidates
        
        # Queue sampling
        assert queue_sampling in ["uniform", "fixed"], \
            f"queue_sampling must be 'uniform' or 'fixed', got {queue_sampling}"
        self.queue_sampling = queue_sampling
        self.fixed_queue_level = fixed_queue_level
        if queue_sampling == "fixed":
            assert fixed_queue_level is not None, \
                "fixed_queue_level must be provided when queue_sampling='fixed'"
            assert fixed_queue_level in [0, 1, 2], \
                f"fixed_queue_level must be in {{0,1,2}}, got {fixed_queue_level}"
        
        # Random number generator
        self.rng = np.random.default_rng(seed)
        
        # Current episode state (initialized in reset)
        self.location_id: Optional[int] = None
        self.battery_level: int = 2  # Fixed to high in v1
        self.queue_level: Optional[int] = None
        self.load_state: int = 0
        
        # Current job parameters
        self.start_node: Optional[int] = None
        self.pickup_node: Optional[int] = None
        self.drop_node: Optional[int] = None
        
        # Episode tracking
        self.step_count: int = 0
        self.done: bool = False
        self.termination_reason: Optional[str] = None
        
    def reset(self) -> Tuple[int, int, int, int]:
        """
        Reset the environment for a new episode (new job).
        
        Returns
        -------
        state : Tuple[int, int, int, int]
            Initial state (location_id, battery_level, queue_level, load_state)
        """
        # Sample job parameters
        self.start_node = self.rng.choice(self.start_candidates)
        self.pickup_node = self.rng.choice(self.pickup_candidates)
        self.drop_node = self.rng.choice(self.drop_candidates)
        
        # Sample or fix queue level
        if self.queue_sampling == "uniform":
            self.queue_level = self.rng.choice([0, 1, 2])
        else:  # "fixed"
            self.queue_level = self.fixed_queue_level
        
        # Initialize state
        self.location_id = self.start_node
        self.battery_level = 2  # High battery in v1
        self.load_state = 0  # Empty
        
        # Reset episode tracking
        self.step_count = 0
        self.done = False
        self.termination_reason = None
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : int
            The node ID to move to (must be a neighbor of current location).
        
        Returns
        -------
        next_state : Tuple[int, int, int, int]
            Next state (location_id, battery_level, queue_level, load_state)
        reward : float
            Reward for this transition
        done : bool
            Whether the episode has terminated
        info : Dict[str, Any]
            Additional information (termination_reason, etc.)
        """
        assert not self.done, "Episode is already done. Call reset() to start a new episode."
        
        # Initialize reward with per-step and queue penalties
        reward = self.r_step + self.r_queue_factor * self.queue_level
        
        # Validate action (must be a neighbor)
        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            # Invalid action: terminal failure
            reward += self.r_failure
            self.done = True
            self.termination_reason = "failure_invalid_action"
            info = {"termination_reason": self.termination_reason}
            return self._get_state(), reward, self.done, info
        
        # Execute movement
        self.location_id = action
        self.step_count += 1
        
        # Check for automatic loading at pickup
        if self.location_id == self.pickup_node and self.load_state == 0:
            self.load_state = 1
        
        # Check for automatic unloading at drop (job completion)
        if self.location_id == self.drop_node and self.load_state == 1:
            reward += self.r_task_completion
            self.done = True
            self.termination_reason = "success"
        
        # Check for timeout
        if not self.done and self.step_count >= self.max_steps:
            reward += self.r_failure
            self.done = True
            self.termination_reason = "failure_timeout"
        
        # Build info dict
        info = {}
        if self.done:
            info["termination_reason"] = self.termination_reason
        
        return self._get_state(), reward, self.done, info
    
    def get_valid_actions(self, location_id: Optional[int] = None) -> List[int]:
        """
        Get valid actions (neighbor nodes) for a given location.
        
        Parameters
        ----------
        location_id : Optional[int]
            Location to get neighbors for. If None, uses current location.
        
        Returns
        -------
        valid_actions : List[int]
            List of neighbor node IDs.
        """
        if location_id is None:
            location_id = self.location_id
        return self.graph[location_id]
    
    def _get_state(self) -> Tuple[int, int, int, int]:
        """
        Get current state as a tuple.
        
        Returns
        -------
        state : Tuple[int, int, int, int]
            (location_id, battery_level, queue_level, load_state)
        """
        return (self.location_id, self.battery_level, self.queue_level, self.load_state)
    
    def render(self) -> None:
        """
        Print current episode state for debugging.
        """
        print("=" * 60)
        print(f"Episode Step: {self.step_count}/{self.max_steps}")
        print(f"Job: Start={self.start_node}, Pickup={self.pickup_node}, Drop={self.drop_node}")
        print(f"State: location={self.location_id}, battery={self.battery_level}, "
              f"queue={self.queue_level}, load={self.load_state}")
        print(f"Done: {self.done}, Reason: {self.termination_reason}")
        print("=" * 60)