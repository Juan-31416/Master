"""
Utility functions for training, evaluation, logging, and plotting.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#                  LOGGING UTILITIES
# ============================================================

def setup_logging(log_path: Path):
    """
    Create a CSV logger for training episodes.
    Columns include success flag for plotting success rate.
    """
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "episode",
        "total_reward",
        "steps",
        "termination_reason",
        "epsilon",
        "success"
    ])

    class LoggerWrapper:
        def __init__(self, writer, file_obj):
            self.writer = writer
            self.file_obj = file_obj

        def log(self, row: Iterable[Any]):
            self.writer.writerow(row)
            self.file_obj.flush()

        def close(self):
            self.file_obj.close()

    return LoggerWrapper(writer, log_file)


def log_episode(
    logger,
    episode: int,
    total_reward: float,
    steps: int,
    termination_reason: str,
    epsilon: float,
    success: bool,
):
    """
    Log one training episode with success flag.
    """
    logger.log([
        episode,
        total_reward,
        steps,
        termination_reason,
        epsilon,
        int(success),  # store 0/1
    ])


def print_training_progress(
    episode: int,
    total_episodes: int,
    recent_rewards: List[float],
    recent_steps: List[int],
    recent_terminations: List[str],
    epsilon: float,
):
    """
    Print console progress summary for recent episodes.
    """
    avg_reward = np.mean(recent_rewards)
    avg_steps = np.mean(recent_steps)

    termination_counts = {}
    for reason in recent_terminations:
        termination_counts[reason] = termination_counts.get(reason, 0) + 1

    print("=" * 80)
    print(f" Episode {episode}/{total_episodes}")
    print(f"  ε = {epsilon:.3f}")
    print(f"  Avg reward (last {len(recent_rewards)}): {avg_reward:.2f}")
    print(f"  Avg steps  (last {len(recent_steps)}): {avg_steps:.2f}")
    print("  Termination reasons (recent):")
    for reason, count in termination_counts.items():
        print(f"   - {reason}: {count}")
    print("=" * 80)


# ============================================================
#                    PLOTTING HELPERS
# ============================================================

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving average with given window size.
    """
    if window <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def plot_learning_curves(
    log_path: Path,
    plots_dir: Path,
    window: int = 100,
):
    """
    Plot training progress:
        1) Total reward per episode + moving average
        2) Steps per episode + moving average
        3) Success rate (moving average of success flag)
    """
    data = np.genfromtxt(log_path, delimiter=",", names=True)

    episodes = data["episode"]
    rewards = data["total_reward"]
    steps = data["steps"]
    success = data["success"]  # 0/1

    ma_rewards = _moving_average(rewards, window)
    ma_steps = _moving_average(steps, window)
    ma_success = _moving_average(success, window)

    # Align x-axis for moving average
    ma_x = episodes[window - 1:] if window > 1 else episodes

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1) Rewards
    axes[0].plot(episodes, rewards, alpha=0.2, color="tab:blue", label="Raw")
    axes[0].plot(ma_x, ma_rewards, color="navy", linewidth=2, label=f"MA({window})")
    axes[0].set_ylabel("Total Reward")
    axes[0].set_title("Training Progress: Total Reward per Episode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2) Steps
    axes[1].plot(episodes, steps, alpha=0.2, color="tab:green", label="Raw")
    axes[1].plot(ma_x, ma_steps, color="darkgreen", linewidth=2, label=f"MA({window})")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Training Progress: Steps per Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3) Success rate
    axes[2].plot(ma_x, ma_success, color="purple", linewidth=2, label=f"Success Rate MA({window})")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Training Progress: Success Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "learning_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Learning curves saved to {out_path}")


def plot_epsilon_decay(log_path: Path, out_path: Path):
    """
    Plot epsilon vs episode from training log.
    """
    data = np.genfromtxt(log_path, delimiter=",", names=True)
    episodes = data["episode"]
    epsilon = data["epsilon"]

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, epsilon, color="tab:orange", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay During Training")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → Epsilon decay plot saved to {out_path}")


def plot_queue_comparison(
    eval_results: Dict[int, Dict[str, float]],
    out_path: Path,
):
    """
    Plot queue level comparison with three subplots:
        - Average reward vs queue level
        - Average steps vs queue level
        - Success rate vs queue level
    
    eval_results: {queue_level: {"avg_reward": ..., "avg_steps": ..., "success_rate": ..., "std_reward": ..., "std_steps": ...}}
    """
    levels = sorted(eval_results.keys())
    avg_rewards = [eval_results[l]["avg_reward"] for l in levels]
    avg_steps = [eval_results[l]["avg_steps"] for l in levels]
    success_rates = [eval_results[l]["success_rate"] for l in levels]
    
    # Get standard deviations if available
    std_rewards = [eval_results[l].get("std_reward", 0) for l in levels]
    std_steps = [eval_results[l].get("std_steps", 0) for l in levels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Reward
    axes[0].errorbar(levels, avg_rewards, yerr=std_rewards, fmt="o-", capsize=5, linewidth=2, markersize=8)
    axes[0].set_xlabel("Queue Level")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Average Reward vs Queue Level")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(levels)

    # 2) Steps
    axes[1].errorbar(levels, avg_steps, yerr=std_steps, fmt="s-", color="tab:green", capsize=5, linewidth=2, markersize=8)
    axes[1].set_xlabel("Queue Level")
    axes[1].set_ylabel("Average Steps")
    axes[1].set_title("Average Steps vs Queue Level")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(levels)

    # 3) Success Rate
    axes[2].bar(levels, success_rates, color="orchid", alpha=0.8)
    axes[2].set_xlabel("Queue Level")
    axes[2].set_ylabel("Success Rate")
    axes[2].set_title("Success Rate vs Queue Level")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(levels)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Queue comparison plot saved to {out_path}")


# ============================================================
#                 EVALUATION UTILITIES
# ============================================================

def compute_eval_metrics(episodes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregated metrics from evaluation episodes.
    
    episodes_data: list of dicts with keys:
        "total_reward", "steps", "success", "queue_level"

    Returns dict with overall metrics and per-queue breakdown.
    """
    rewards = np.array([e["total_reward"] for e in episodes_data], dtype=float)
    steps = np.array([e["steps"] for e in episodes_data], dtype=float)
    success = np.array([e["success"] for e in episodes_data], dtype=float)
    queue_levels = np.array([e["queue_level"] for e in episodes_data], dtype=int)

    metrics = {
        "n_episodes": len(episodes_data),
        "avg_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "avg_steps": float(steps.mean()),
        "std_steps": float(steps.std(ddof=1)) if len(steps) > 1 else 0.0,
        "success_rate": float(success.mean()),
    }

    # Per-queue metrics
    queue_metrics: Dict[int, Dict[str, float]] = {}
    for q in sorted(set(queue_levels)):
        mask = queue_levels == q
        queue_rewards = rewards[mask]
        queue_steps = steps[mask]
        queue_success = success[mask]
        queue_metrics[q] = {
            "avg_reward": float(queue_rewards.mean()),
            "std_reward": float(queue_rewards.std(ddof=1)) if len(queue_rewards) > 1 else 0.0,
            "avg_steps": float(queue_steps.mean()),
            "std_steps": float(queue_steps.std(ddof=1)) if len(queue_steps) > 1 else 0.0,
            "success_rate": float(queue_success.mean()),
            "n_episodes": int(mask.sum()),
        }

    metrics["per_queue"] = queue_metrics
    return metrics


def save_metrics_table_csv(
    metrics_per_policy: Dict[str, Dict[str, Any]],
    out_path: Path,
):
    """
    Save comparison table with one row per policy.
    
    Columns: policy_name, graph, n_episodes, avg_reward, std_reward, avg_steps, std_steps, success_rate
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "policy_name",
        "graph",
        "n_episodes",
        "avg_reward",
        "std_reward",
        "avg_steps",
        "std_steps",
        "success_rate",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for policy_name, m in metrics_per_policy.items():
            row = [
                policy_name,
                m.get("graph", "unknown"),
                m["n_episodes"],
                f"{m['avg_reward']:.2f}",
                f"{m['std_reward']:.2f}",
                f"{m['avg_steps']:.2f}",
                f"{m['std_steps']:.2f}",
                f"{m['success_rate']:.3f}",
            ]
            writer.writerow(row)

    print(f"  → Metrics table saved to {out_path}")


def save_json(obj: Any, path: Path):
    """
    Save object as JSON file.
    Handles numpy types by converting them to native Python types.
    """
    import numpy as np
    
    def convert_numpy(o):
        """Recursively convert numpy types to native Python types."""
        if isinstance(o, dict):
            return {convert_numpy(k): convert_numpy(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert_numpy(item) for item in o]
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return o
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert_numpy(obj), f, indent=2)
    print(f"  → JSON saved to {path}")


def load_json(path: Path) -> Any:
    """
    Load JSON file.
    """
    with open(path, "r") as f:
        return json.load(f)