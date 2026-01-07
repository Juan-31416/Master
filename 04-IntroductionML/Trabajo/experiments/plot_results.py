"""
Script to generate all plots from training and evaluation results.

Generates:
1. Learning curves (reward, steps, success rate) for TOY graph
2. Learning curves (reward, steps, success rate) for PLANT graph
3. Epsilon decay for TOY graph
4. Epsilon decay for PLANT graph
5. Queue comparison for TOY graph
6. Queue comparison for PLANT graph
7. Learning curves comparison (TOY vs PLANT)
"""

import sys
sys.path.insert(0, '../')

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import (
    plot_learning_curves,
    plot_queue_comparison,
    plot_epsilon_decay,
    load_json,
)

# Paths
RESULTS_DIR = Path("../results")
LOGS_DIR = RESULTS_DIR / "logs"
EVAL_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"

print("DEBUG RESULTS_DIR:", RESULTS_DIR.resolve())
print("DEBUG LOGS_DIR:", LOGS_DIR.resolve())
print("DEBUG log_files:", list(LOGS_DIR.glob("train_log_*.csv")))

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_learning_curves_comparison(
    toy_log_path: Path,
    plant_log_path: Path,
    out_path: Path,
    window: int = 100,
):
    """
    Plot side-by-side comparison of learning curves for toy vs plant graphs.
    """
    # Load data
    toy_data = np.genfromtxt(toy_log_path, delimiter=",", names=True)
    plant_data = np.genfromtxt(plant_log_path, delimiter=",", names=True)
    
    # Moving average helper
    def ma(x, w):
        if w <= 1:
            return x
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[w:] - cumsum[:-w]) / float(w)
    
    # Toy
    toy_episodes = toy_data["episode"]
    toy_rewards = toy_data["total_reward"]
    toy_steps = toy_data["steps"]
    toy_success = toy_data["success"]
    toy_ma_rewards = ma(toy_rewards, window)
    toy_ma_steps = ma(toy_steps, window)
    toy_ma_success = ma(toy_success, window)
    toy_ma_x = toy_episodes[window - 1:] if window > 1 else toy_episodes
    
    # Plant
    plant_episodes = plant_data["episode"]
    plant_rewards = plant_data["total_reward"]
    plant_steps = plant_data["steps"]
    plant_success = plant_data["success"]
    plant_ma_rewards = ma(plant_rewards, window)
    plant_ma_steps = ma(plant_steps, window)
    plant_ma_success = ma(plant_success, window)
    plant_ma_x = plant_episodes[window - 1:] if window > 1 else plant_episodes
    
    # Create figure with 3 rows, 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Row 1: Rewards
    axes[0, 0].plot(toy_episodes, toy_rewards, alpha=0.2, color="tab:blue")
    axes[0, 0].plot(toy_ma_x, toy_ma_rewards, color="navy", linewidth=2, label=f"MA({window})")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].set_title("TOY Graph - Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(plant_episodes, plant_rewards, alpha=0.2, color="tab:blue")
    axes[0, 1].plot(plant_ma_x, plant_ma_rewards, color="navy", linewidth=2, label=f"MA({window})")
    axes[0, 1].set_ylabel("Total Reward")
    axes[0, 1].set_title("PLANT Graph - Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Row 2: Steps
    axes[1, 0].plot(toy_episodes, toy_steps, alpha=0.2, color="tab:green")
    axes[1, 0].plot(toy_ma_x, toy_ma_steps, color="darkgreen", linewidth=2, label=f"MA({window})")
    axes[1, 0].set_ylabel("Steps")
    axes[1, 0].set_title("TOY Graph - Steps")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(plant_episodes, plant_steps, alpha=0.2, color="tab:green")
    axes[1, 1].plot(plant_ma_x, plant_ma_steps, color="darkgreen", linewidth=2, label=f"MA({window})")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].set_title("PLANT Graph - Steps")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Row 3: Success Rate
    axes[2, 0].plot(toy_ma_x, toy_ma_success, color="purple", linewidth=2, label=f"Success Rate MA({window})")
    axes[2, 0].set_xlabel("Episode")
    axes[2, 0].set_ylabel("Success Rate")
    axes[2, 0].set_ylim(-0.05, 1.05)
    axes[2, 0].set_title("TOY Graph - Success Rate")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(plant_ma_x, plant_ma_success, color="purple", linewidth=2, label=f"Success Rate MA({window})")
    axes[2, 1].set_xlabel("Episode")
    axes[2, 1].set_ylabel("Success Rate")
    axes[2, 1].set_ylim(-0.05, 1.05)
    axes[2, 1].set_title("PLANT Graph - Success Rate")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → Learning curves comparison saved to {out_path}")


def main():
    """Generate all plots."""
    
    print("=" * 80)
    print("GENERATING ALL PLOTS")
    print("=" * 80)
    print()
    
    # Find training logs
    toy_logs = sorted(LOGS_DIR.glob("train_log_toy_*.csv"))
    plant_logs = sorted(LOGS_DIR.glob("train_log_plant_*.csv"))
    
    if not toy_logs and not plant_logs:
        print("ERROR: No training logs found in", LOGS_DIR)
        print("Please run train.py first.")
        return
    
    # ========================================
    # 1 & 2: Individual learning curves
    # ========================================
    if toy_logs:
        toy_log = toy_logs[-1]
        print(f"1. Generating TOY learning curves from {toy_log.name}...")
        plot_learning_curves(toy_log, PLOTS_DIR / "toy", window=100)
        
        print(f"2. Generating TOY epsilon decay...")
        plot_epsilon_decay(toy_log, PLOTS_DIR / "toy" / "epsilon_decay_toy.png")
    
    if plant_logs:
        plant_log = plant_logs[-1]
        print(f"3. Generating PLANT learning curves from {plant_log.name}...")
        plot_learning_curves(plant_log, PLOTS_DIR / "plant", window=100)
        
        print(f"4. Generating PLANT epsilon decay...")
        plot_epsilon_decay(plant_log, PLOTS_DIR / "plant" / "epsilon_decay_plant.png")
    
    # ========================================
    # 5 & 6: Queue comparison plots
    # ========================================
    toy_eval_json = sorted(EVAL_DIR.glob("evaluation_results_*_toy.json"))
    plant_eval_json = sorted(EVAL_DIR.glob("evaluation_results_*_plant.json"))
    
    if toy_eval_json:
        print(f"\n5. Generating TOY queue comparison...")
        for json_file in toy_eval_json:
            policy_name = json_file.stem.replace("evaluation_results_", "").replace("_toy", "")
            eval_data = load_json(json_file)
            eval_data = {int(k): v for k, v in eval_data.items()}
            out_path = PLOTS_DIR / "toy" / f"queue_comparison_{policy_name}_toy.png"
            plot_queue_comparison(eval_data, out_path)
    else:
        print("\n5. Skipping TOY queue comparison (no evaluation results)")
    
    if plant_eval_json:
        print(f"\n6. Generating PLANT queue comparison...")
        for json_file in plant_eval_json:
            policy_name = json_file.stem.replace("evaluation_results_", "").replace("_plant", "")
            eval_data = load_json(json_file)
            eval_data = {int(k): v for k, v in eval_data.items()}
            out_path = PLOTS_DIR / "plant" / f"queue_comparison_{policy_name}_plant.png"
            plot_queue_comparison(eval_data, out_path)
    else:
        print("\n6. Skipping PLANT queue comparison (no evaluation results)")
    
    # ========================================
    # 7: Learning curves comparison (TOY vs PLANT)
    # ========================================
    if toy_logs and plant_logs:
        print(f"\n7. Generating TOY vs PLANT learning curves comparison...")
        comparison_path = PLOTS_DIR / "learning_curves_comparison_toy_vs_plant.png"
        plot_learning_curves_comparison(toy_logs[-1], plant_logs[-1], comparison_path, window=100)
    else:
        print("\n7. Skipping comparison (need both toy and plant training logs)")
    
    print("\n" + "=" * 80)
    print("ALL PLOTS GENERATED")
    print("=" * 80)
    print(f"\nPlots saved to: {PLOTS_DIR}")
    print("  - toy/learning_curves.png")
    print("  - toy/epsilon_decay_toy.png")
    print("  - toy/queue_comparison_*.png")
    print("  - plant/learning_curves.png")
    print("  - plant/epsilon_decay_plant.png")
    print("  - plant/queue_comparison_*.png")
    print("  - learning_curves_comparison_toy_vs_plant.png")
    print("=" * 80)


if __name__ == "__main__":
    main()