"""
Training script for the Q-learning routing agent.

This script:
- Initializes the environment and agent
- Runs the training loop for a specified number of episodes
- Logs training metrics (reward, steps, termination reason)
- Saves teh trained Q-table periodically and at the end
"""

from sqlite3.dbapi2 import Timestamp
import sys
sys.path.insert(0, '../')

import numpy as np
from pathlib import Path
from datetime import datetime

from src.environment import RoutingEnv
from src.agent import QLearningAgent
from src.utils import setup_logging, log_episode, print_training_progress
from config.graphs import (
    DEFAULT_DROP_CANDIDATES_TOY,
    DEFAULT_PICKUP_CANDIDATES_TOY,
    DEFAULT_START_CANDIDATES_TOY,
    GRAPH_PLANT,
    GRAPH_TOY,
    MAX_STEPS,
    MAX_STEPS_TOY,
    NODE_ROLES_PLANT,
    NODE_ROLES_TOY,
    REWARD_PARAMS_DEFAULT,
    DEFAULT_START_CANDIDATES,
    DEFAULT_PICKUP_CANDIDATES,
    DEFAULT_DROP_CANDIDATES,
)
from collections import Counter

termination_counter = Counter() # For debugging purposses
 
# ===================================================
#               TRAINING CONFIGURATION
# ===================================================

# Agent hyperparameters
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_MIN = 0.05      # Minimum exploration rate
EPSILON_DECAY = 0.999   # Multiplicative decay per episode

# Training parameters
NUM_EPISODES = 3000     # Total number of training episodes
PRINT_EVERY = 100       # Print progress every N episodes
SAVE_EVERY = 500        # Save Q-table every N episodes

# Random seed for reproducibility
SEED = 42

# Output paths
RESULTS_DIR = Path("../results")
LOGS_DIR = RESULTS_DIR / "logs"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ===================================================
#              MAIN TRAINING FUNCTION
# ===================================================

def train():
    """
    Main training loop for Q-learning agent.
    """

    #Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"train_log_{timestamp}.csv"
    logger = setup_logging(log_path)

    print("=" * 80)
    print("Q-LEARNING ROUTING AGENT - TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f" Episodes: {NUM_EPISODES}")
    print(f" Max steps per episode: {MAX_STEPS}")
    print(f" Learning rate (α): {ALPHA}")
    print(f" Discount factor (γ): {GAMMA}")
    print(f"  Exploration: ε={EPSILON_START} → {EPSILON_MIN} (decay={EPSILON_DECAY})")
    print(f" Random seed: {SEED}")
    print(f"\nReward parameters:")
    for key, value in REWARD_PARAMS_DEFAULT.items():
        print(f"\n {key}: {value}")
    print(f"\nLog file: {log_path}")
    print("=" * 80)
    print()

    # Initialize toy env
    """
    env = RoutingEnv(
        graph=GRAPH_TOY,
        node_roles=NODE_ROLES_TOY,
        reward_params=REWARD_PARAMS_DEFAULT,
        max_steps=MAX_STEPS_TOY,
        start_candidates=DEFAULT_START_CANDIDATES_TOY,
        pickup_candidates=DEFAULT_PICKUP_CANDIDATES_TOY,
        drop_candidates=DEFAULT_DROP_CANDIDATES_TOY,
        queue_sampling="uniform",
        seed=SEED,
    )
    """

    # Initiallize environment
    
    env = RoutingEnv(
        graph=GRAPH_PLANT,
        node_roles=NODE_ROLES_PLANT,
        reward_params=REWARD_PARAMS_DEFAULT,
        max_steps=MAX_STEPS,
        start_candidates=DEFAULT_START_CANDIDATES,
        pickup_candidates=DEFAULT_PICKUP_CANDIDATES,
        drop_candidates=DEFAULT_DROP_CANDIDATES,
        queue_sampling="uniform",
        seed=SEED,
        penalize_revisits=True,
        revisit_penalty_factor=-2.0,
    )
    

    # Initialize agent
    agent = QLearningAgent(
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=SEED,
    )

    # Training metrics (for progress tracking)
    episode_rewards = []
    episode_steps = []
    episode_terminations = []

    # Training loop
    print("Starting training...\n")

    for episode in range(1, NUM_EPISODES + 1):
        # Reset environment
        state = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0

        # Episode loop
        while not done:
            # Select action
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Get valid actions for net state (for Q-learnig update)
            if not done:
                next_valid_actions = env.get_valid_actions()
            else:
                next_valid_actions= []
            
            # Update Q-table
            agent.update(state, action, reward, next_state, next_valid_actions, done)

            # Update state and metrics
            state = next_state
            episode_reward += reward
            step_count += 1
        
       

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        termination_reason = info.get("termination_reason", "unknown")

        if episode <= 10 or episode % 100 == 0:
            print(f"Ep {episode}: start={env.start_node}, pickup={env.pickup_node}, drop={env.drop_node}, "
                f"reward={episode_reward:.1f}, steps={step_count}, reason={termination_reason}")

        termination_reason = info.get("termination_reason", "unknown")
        episode_terminations.append(termination_reason)
        termination_counter[termination_reason] += 1

        # Decay epsilon after episode
        agent.decay_epsilon()

        # Log episode
        log_episode(
            logger,
            episode=episode,
            total_reward=episode_reward,
            steps=step_count,
            termination_reason=termination_reason,
            epsilon=agent.epsilon,
        )

        # Print progress
        if episode % PRINT_EVERY == 0:
            print_training_progress(
                episode=episode,
                total_episodes=NUM_EPISODES,
                recent_rewards=episode_rewards[-PRINT_EVERY:],
                recent_steps=episode_steps[-PRINT_EVERY:],
                recent_terminations=episode_terminations[-PRINT_EVERY:],
                epsilon=agent.epsilon,
            )

        # Save Q-table periodically
        if episode % SAVE_EVERY == 0:
            save_path = MODELS_DIR / f"q_table_episode_{episode}.pkl"
            agent.save(str(save_path))
            print(f"  → Model saved: {save_path.name}")

    # Final save
    final_save_path = MODELS_DIR / f"q_table_final_{timestamp}.pkl"
    agent.save(str(final_save_path))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nFinal Q-table statistics:")
    stats = agent.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f" {key}: {value:.4f}")
        else:
            print(f" {key}: {value}")
    print(f"Final model saved: {final_save_path}")
    print(f"Training log saved: {log_path}")
    print("\n" + "=" * 80)

    # Debugging
    print("\nTermination breakdown over all episodes:")
    for reason, count in termination_counter.items():
       print(f"  {reason}: {count} ({count / NUM_EPISODES:.1%})")

# ================================================
#                   ENTRY POINT
# ================================================

if __name__ == "__main__":
    train()