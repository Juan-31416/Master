"""
Utility functions for logging, plotting, and analysis.
"""

import csv
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==============================================
#               LOGGING UTILITIES
# ==============================================

def setup_logging(log_path: Path) -> Any:
    """
    Setup CSV logger for training

    Parameters
    ----------
    log_path : Path
        Path to the log file.
    
    Returns
    -------
    logger : file handle
        Open CSV file handle for logging.
    """

    log_file = open(log_path, 'w', newline='')
    logger = csv.writer(log_file)

    # Write header
    logger.writerow([
        "episode",
        "total_reward",
        "steps",
        "termination_reason",
        "epsilon",
    ])

    log_file.flush()

    return log_file

def log_episode(
    logger: Any,
    episode: int,
    total_reward: float,
    steps: int,
    termination_reason: str,
    epsilon: float,
) -> None:
    """
    Log a single episode to CSV.

    Parameters
    ----------
    logger : file handle
        CSV logger from setup_logging.
    episode : int
        Episode number.
    total_reward : float
        Total reward for the episode.
    steps : int
        Number of steps taken.
    termination_reason : str
        Reason for episode termination.
    epsilon : float
        Current exploration rate.
    """

    writer = csv.writer(logger)
    writer.writerow([
        episode,
        f"{total_reward:.4f}",
        steps,
        termination_reason,
        f"{epsilon:.6f}",
    ])
    logger.flush()

def print_training_progress(
    episode: int,
    total_episodes: int,
    recent_rewards: List[float],
    recent_steps: List[int],
    recent_terminations: List[str],
    epsilon: float,
) -> None:
    """
    Print trainning progress summary.

    Parameters
    ----------
    episode : int
        Current episode number.
    total_episodes : int
        Total number of episodes.
    recent_rewards : List[float]
        Recent episode rewards (e.g., last 100).
    recent_steps : List[int]
        Recent episode step counts.
    recent_terminations : List[str]
        Recent termination reasons.
    epsilon : float
        Current exploration rate.
    """

    avg_reward = sum(recent_rewards) / len(recent_rewards)
    avg_steps = sum(recent_steps) / len(recent_steps)

    # Count termination types
    success_count = sum(1 for t in recent_terminations if t == "success")
    failure_count = len(recent_terminations) - success_count
    success_rate = success_count / len(recent_terminations)

    print(f"Episode {episode}/{total_episodes}")
    print(f" Avg reward (last {len(recent_rewards)}): {avg_reward:.2f}")
    print(f" Avg steps (last {len(recent_steps)}): {avg_steps:.1f}")
    print(f" Success rate: {success_rate:.1%} ({success_count}/{len(recent_terminations)})")
    print(f" Epsilon: {epsilon:.4f}")
    print()

# ===================================================
#           PLOTTING UTILITIES (Phase 4)
# ===================================================

def plot_learning_curves(log_path: Path, out_dir: Path, window: int = 100):
    """
    Plot learning curves from training log.

    Parameters
    ----------
    log_path : Path
        Path to training log CSV.
    out_dir : Path
        Directory to save plots.
    window : int
        Window size for moving average.
    """  

    # Load training log
    df = pd.read_csv(log_path)

    # Compute moving averages
    df['reward_ma'] = df['total_reward'].rolling(window=window, min_periods=1).mean()
    df['steps_ma'] = df['steps'].rolling(window=window, min_periods=1).mean()

    # Compute success rate (rolling)
    df['is_success'] = (df['termination_reason'] == 'success').astype(int)
    df['success_rate_ma'] = df['steps'].rolling(window=window, min_periods=1).mean()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Total Reward    
    axes[0].plot(df['episode'], df['total_reward'], alpha=0.3, label='Raw', color='blue')
    axes[0].plot(df['episode'], df['reward_ma'], label=f'MA({window})', color='darkblue', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Progress: Total Reward per Episode')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Steps per Episode
    axes[1].plot(df['episode'], df['steps'], alpha=0.3, label='Raw', color='green')
    axes[1].plot(df['episode'], df['steps_ma'], label=f'MA({window})', color='darkgreen', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Training Progress: Steps per Episode')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Success Rate
    axes[2].plot(df['episode'], df['success_rate_ma'], label=f'Success Rate MA({window})', color='purple', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Training Progress: Success Rate')
    axes[2].set_ylim([0, 1.05])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()

    # Save
    out_path = out_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {out_path}")

    plt.close()


def plot_queue_comparison(eval_results: Dict[int, Dict[str, float]], out_path: Path):
    """
    Plot comparison of performance across queue levels.
    
    Parameters
    ----------
    eval_results : Dict[int, Dict[str, float]]
        Evaluation results keyed by queue_level.
    out_path : Path
        Path to save plot.
    """
    queue_levels = sorted(eval_results.keys())


    avg_rewards = [eval_results[q]['avg_reward'] for q in queue_levels]
    std_rewards = [eval_results[q]['std_reward'] for q in queue_levels]

    avg_steps = [eval_results[q]['avg_steps'] for q in queue_levels]
    std_steps = [eval_results[q]['std_steps'] for q in queue_levels]

    success_rates = [eval_results[q]['success_rate'] for q in queue_levels]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Average Reward
    axes[0].errorbar(queue_levels, avg_rewards, yerr=std_rewards, marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0].set_xlabel('Queue Level')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Average Reward vs Queue Level')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(queue_levels)
    
    # Plot 2: Average Steps
    axes[1].errorbar(queue_levels, avg_steps, yerr=std_steps, marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Queue Level')
    axes[1].set_ylabel('Average Steps')
    axes[1].set_title('Average Steps vs Queue Level')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(queue_levels)
    
    # Plot 3: Success Rate
    axes[2].bar(queue_levels, success_rates, color='purple', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Queue Level')
    axes[2].set_ylabel('Success Rate')
    axes[2].set_title('Success Rate vs Queue Level')
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xticks(queue_levels)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Queue comparison plot saved to: {out_path}")
    
    plt.close()

def plot_epsilon_decay(log_path: Path, out_path: Path):
    """
    Plot epsilon decay over training.
    
    Parameters
    ----------
    log_path : Path
        Path to training log CSV.
    out_path : Path
        Path to save plot.
    """

    df = pd.read_csv(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df['episode'], df['epsilon'], linewidth=2, color='red')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon (ε)')
    plt.title('Exploration Rate Decay')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Epsilon decay plot saved to: {out_path}")
    
    plt.close()