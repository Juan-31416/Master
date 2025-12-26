"""
Script to generate all plots form training and evaluation results
"""

import sys
sys.path.insert(0, '../')

import json
from pathlib import Path
from src.utils import plot_learning_curves, plot_queue_comparison, plot_epsilon_decay

# Paths
RESULTS_DIR = Path("../results")
LOGS_DIR = RESULTS_DIR / "logs"
EVAL_DIR = RESULTS_DIR / "evaluation"
PLOTS_DIR = RESULTS_DIR / "plots"

# Create plots directory
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """
    Generate all plots
    """

    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    print()

    # Find most recent training log
    log_files = sorted(LOGS_DIR.glob("train_log_*.csv"))
    
    if not log_files:
        print("ERROR: No training log found in", LOGS_DIR)
        print("Please run train.py first.")
        return
    
    log_path = log_files[-1]
    print(f"Using training log: {log_path.name}")
    
    # Plot 1: Learning curves
    print("\n1. Generating learning curves...")
    plot_learning_curves(log_path, PLOTS_DIR, window=100)
    
    # Plot 2: Epsilon decay
    print("\n2. Generating epsilon decay plot...")
    epsilon_plot_path = PLOTS_DIR / "epsilon_decay.png"
    plot_epsilon_decay(log_path, epsilon_plot_path)
    
    # Plot 3: Queue comparison (if evaluation results exist)
    eval_results_path = EVAL_DIR / "evaluation_results.json"

    if eval_results_path.exists():
        print("\n3. Generating queue comparison plot...")

        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        # Convert string keys to int
        eval_results = {int(k): v for k, v in eval_results.items()}

        queue_plot_path = PLOTS_DIR / "queue_comparison.png"
        plot_queue_comparison(eval_results, queue_plot_path)
    else:
        print("\n3. Skipping queue comparison plot (no evaluation results found)")
        print(f"   Run evaluate.py first to generate {eval_results_path}")
    
    print("\n" + "=" * 80)
    print("PLOTS GENERATED")
    print("=" * 80)
    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()