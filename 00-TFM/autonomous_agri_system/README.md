# Navegación Autónoma Híbrida en BEV con Optimización de Rutas para Seguimiento de Cultivos

Comprehensive research-oriented prototype for a Master's thesis with three coordinated modules:

1. **Perception (BEV Transformer)** for multimodal camera+LiDAR-like fusion
2. **Hybrid Path Planning** combining global coverage, local ML scoring/lattice selection, and MPC tracking
3. **Agricultural Computer Vision** based on DeepLabV3+ with EfficientNet backbone for crop/disease/weed segmentation

---

## 1. Project Structure

```text
autonomous_agri_system/
├── README.md
├── requirements.txt
├── configs/
│   ├── perception.yaml
│   ├── planning.yaml
│   └── identification.yaml
├── perception/
│   ├── __init__.py
│   ├── bev_transformer.py
│   ├── data_generator.py
│   ├── train_bev.py
│   └── evaluate_bev.py
├── planning/
│   ├── __init__.py
│   ├── global_planner.py
│   ├── local_planner.py
│   ├── mpc_controller.py
│   ├── trajectory_scorer_net.py
│   ├── train_planner.py
│   └── simulate_planning.py
├── identification/
│   ├── __init__.py
│   ├── deeplabv3plus.py
│   ├── dataset.py
│   ├── train_segmentation.py
│   ├── evaluate_segmentation.py
│   └── inference.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   ├── metrics.py
│   └── config.py
├── tests/
│   ├── test_metrics.py
│   ├── test_global_planner.py
│   └── test_local_planner.py
└── notebooks/
    ├── demo_bev_perception.ipynb
    ├── demo_planning.ipynb
    └── demo_identification.ipynb
```

---

## 2. Installation

```bash
cd /home/ubuntu/autonomous_agri_system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Perception Module (BEV Transformer)

### Features
- Camera + LiDAR simulation and multimodal feature extraction
- Cross-attention fusion block
- Multi-task heads:
  - Occupancy grid
  - Semantic segmentation
  - Object detection (prototype vector)
  - Confidence map
- Synthetic data generation for training without real field BEV labels

### Train
```bash
python -m perception.train_bev --config configs/perception.yaml
```

### Evaluate
```bash
python -m perception.evaluate_bev \
  --config configs/perception.yaml \
  --checkpoint checkpoints/bev/best_bev.pt
```

---

## 4. Hybrid Planning Module

### Features
- Global coverage planner (boustrophedon / zigzag)
- Local lattice trajectory generation
- Neural trajectory scorer (`TrajectoryScorerNet`)
- Hybrid trajectory selection (ML score + classical safety/cost)
- Kinematic-bicycle MPC-style tracking controller

### Train trajectory scorer
```bash
python -m planning.train_planner --config configs/planning.yaml
```

### Simulate end-to-end planning
```bash
python -m planning.simulate_planning \
  --config configs/planning.yaml \
  --checkpoint checkpoints/planning/best_planner_scorer.pt
```

---

## 5. Identification Module (DeepLabV3+)

### Features
- DeepLabV3+ with EfficientNet backbone (via SMP)
- Fallback to torchvision DeepLabV3 if SMP is unavailable
- Data loader and augmentations for agricultural segmentation
- Metrics: mIoU, per-class IoU, macro F1
- Inference script with mask and overlay outputs

### Expected dataset format

```text
data/agri_segmentation/
  images/
    train/
    val/
    test/
  masks/
    train/
    val/
    test/
```

> Masks are expected as single-channel PNG class-index images. File names must match image basenames.

### Train
```bash
python -m identification.train_segmentation --config configs/identification.yaml
```

### Evaluate
```bash
python -m identification.evaluate_segmentation \
  --config configs/identification.yaml \
  --checkpoint checkpoints/identification/best_segmentation.pt
```

### Inference
```bash
python -m identification.inference \
  --config configs/identification.yaml \
  --checkpoint checkpoints/identification/best_segmentation.pt \
  --input_dir /path/to/new/images \
  --output_dir outputs/inference
```

---

## 6. Reproducibility and Thesis Notes

- Config-driven experiments in `configs/*.yaml`
- Train/validation/test split support in synthetic modules and folder-based splits for user dataset
- Checkpointing included in all training scripts
- Qualitative outputs generated during evaluation
- Unit test examples provided for critical functions

This codebase is intended as a **working research prototype** for educational and thesis contribution purposes.
