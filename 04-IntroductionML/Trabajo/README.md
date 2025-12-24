# RL-Based Routing for Industrial Vehicle

## Project Overview

This project implements a **tabular Q-learning agent** for routing a single industrial vehicle (AGV-like) in a discrete plant layout. The goal is to minimize job completion time while accounting for system load (queue level).

## Problem Formulation

- **State**: `(location_id, battery_level, queue_level, load_state)`
- **Action**: Move to a neighboring node in the plant graph
- **Reward**: Step penalty + queue penalty + completion bonus/failure penalty
- **Algorithm**: Tabular Q-learning with ε-greedy exploration

## Project Structure

- `config/`: Graph definitions and node roles
- `src/`: Core modules (environment, agent, utilities)
- `experiments/`: Training and evaluation scripts
- `results/`: Saved models, logs, and plots
- `tests/`: Unit tests

## Setup

```bash
pip install -r requirements.txt