"""
Evaluatin for trained Q-learning agent.

This script:
- Loads a trained Q-table
- Evaluates the agent on multiple queue levels (greedy policy, ε=0)
- Computes metrics: avg reward, avg steps, uccess rate
- Saves evaluation results to JSON
"""

import sys
sys.path.insert(0, '../')

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

from src.environment import RoutingEnv
from src.agent import QLearningAgent
from config.graphs import (
    GRAPH_PLANT,
    NODE_ROLES_PLANT,
    DEFAULT_START_CANDIDATES,
    DEFAULT_PICKUP_CANDIDATES,
    DEFAULT_DROP_CANDIDATES,
    REWARD_PARAMS_DEFAULT,
    MAX_STEPS
)


# =================================================
#           EVALUATION CONFIGURATION
# =================================================

# Evaluation parameters
NUM_EVAL_EPISODES = 100     # Episodes per queue level
QUEUE_LEVELS_TO_TEST = [0, 1, 2]  # Test different queue levels
SEED = 999  # Different seed from training for fair evaluation

# Paths
RESULTS_DIR = Path("../results")
MODELS_DIR = RESULTS_DIR / "models"
EVAL_DIR = RESULTS_DIR / "evaluation"

# Create directories
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ================================================
#              EVALUATION FUNCTIONS
# ================================================

def evaluate_agent_on_queue_level(
    agent: QLearningAgent,
    env: RoutingEnv,
    queue_level: int,
    num_episodes: int,
) -> Dict[str, Any]:
    """
    Evalate agent on a fixed queue level.

    Parameters
    ----------
    agent : QLearningAgent
        Trained agent (will use greedy policy, ε=0).
    env : RoutingEnv
        Environment configured with fixed queue level.
    queue_level : int
        Queue level to test.
    num_episodes : int
        Number of evaluation episodes.
    
    Returns
    -------
    results : Dict[str, Any]
        Evaluation metrics.
    """

    # Store original epsilon and set it to 0 (greedy)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    episode_rewards = []
    episode_steps = []
    episode_terminations = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        episode_reward = 0.0
        step_count = 0

        while not done:
            valid_actions = env.get_valid_actions()
            action =agent.select_action(state, valid_actions)

            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward
            step_count += 1

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        termination_reason = info.get("termination_reason", "unknown")
        episode_terminations.append(termination_reason)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon

    # Compute metrics
    success_count = sum(1 for t in episode_terminations if t == "success")
    failure_count = num_episodes - success_count

    results = {
        "queue_level": queue_level,
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_steps": float(np.mean(episode_steps)),
        "std_steps": float(np.std(episode_steps)),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / num_episodes,
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "min_steps": int(np.min(episode_steps)),
        "max_steps": int(np.max(episode_steps)),
    }
    return(results)

def evaluate_agent(
    model_path: str,
    queue_levels: list,
    num_episodes: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate trained agent across multiple queue levels.

    Parameters
    ----------
    model_path : str
        Path to trained Q-table (.pkl file).
    queue_levels : list
        List of queue levels to test.
    num_episodes : int
        Number of episodes per queue level.
    
    Returns
    -------
    all_results : Dict[int, Dict[str, Any]]
        Results keyed by queue_level.
    """

    print("=" * 80)
    print("Q-LEARNING AGENT - EVALUATION")
    print("=" * 80)
    print(f"\nLoading model: {model_path}")
    
    # Load agent
    agent = QLearningAgent()
    agent.load(model_path)
    
    print(f"\nEvaluation configuration:")
    print(f"  Episodes per queue level: {num_episodes}")
    print(f"  Queue levels to test: {queue_levels}")
    print(f"  Policy: Greedy (ε=0)")
    print(f"  Random seed: {SEED}")
    print("=" * 80)
    print()
    
    all_results = {}

    for  queue_level in queue_levels:
        print(f"Evaluating on queue_level={queue_level}...")

        # Create environment with fixed queue level
        env = RoutingEnv(
            graph=GRAPH_PLANT,
            node_roles=NODE_ROLES_PLANT,
            reward_params=REWARD_PARAMS_DEFAULT,
            max_steps=MAX_STEPS,
            start_candidates=DEFAULT_START_CANDIDATES,
            pickup_candidates=DEFAULT_PICKUP_CANDIDATES,
            drop_candidates=DEFAULT_DROP_CANDIDATES,
            queue_sampling="fixed",
            fixed_queue_level=queue_level,
            seed=SEED,
        )

        # Evaluate
        results = evaluate_agent_on_queue_level(
            agent=agent,
            env=env,
            queue_level=queue_level,
            num_episodes=num_episodes
        )

        all_results[queue_level] = results

        # Print results
        print(f"  Avg reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Avg steps: {results['avg_steps']:.1f} ± {results['std_steps']:.1f}")
        print(f"  Success rate: {results['success_rate']:.1%} ({results['success_count']}/{num_episodes})")
        print()
    
    return all_results

# =============================================
#                   MAIN
# =============================================

def main():
    """
    Main evaluation function.
    """

    # Find the most recent final model
    model_files = sorted(MODELS_DIR.glob("q_table_final*.pkl"))

    if not model_files:
        print("ERROR: No trained model found in ", MODELS_DIR)
        print("Please run train.py first")
        return
    
    model_path = model_files[-1]    # Most recent

    # Run evaluation
    all_results = evaluate_agent(
        model_path=str(model_path),
        queue_levels=QUEUE_LEVELS_TO_TEST,
        num_episodes=NUM_EVAL_EPISODES,
    )

    # Save results to JSON
    output_path = EVAL_DIR / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    for queue_level, results in all_results.items():
        print(f"  Queue {queue_level}: "
              f"Reward={results['avg_reward']:.2f}, "
              f"Steps={results['avg_steps']:.1f}, "
              f"Success={results['success_rate']:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()