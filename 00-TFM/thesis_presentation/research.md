# Navegación Autónoma Híbrida en BEV con Optimización de Rutas para Seguimiento de Cultivos
*Pedro Canales — August 19, 2026*

---

## 1. Thesis Purpose, Scope, and Research Questions

### Core Problem
No publicly available system integrates multimodal BEV perception, hybrid coverage planning, and agricultural crop-health identification in a single validated architecture. Five critical gaps drive the work: absence of agricultural BEV benchmarks; no lightweight BEV analysis for embedded agricultural hardware; no ground-vehicle BEV crop-health integration; no hybrid CPP with simultaneous health monitoring; and no established domain adaptation protocols for agricultural scenes.

### Scope
- Single vehicle, known field geometry, static obstacles
- Out of scope: dynamic obstacles, multi-vehicle coordination, automatic implement docking
- Validation level: **simulation and prototype only** — field validation on real agricultural terrain has not been established

### Four Research Questions

| RQ | Question | Declared Target |
|----|----------|----------------|
| RQ1 | Can compact dual-modality BEV provide sufficient scene understanding for safe coverage? | mIoU > 0.70 |
| RQ2 | Does Boustrophedon + lattice local planner + sampled MPC meet safety and tracking targets? | Collision-free > 95%; RMSE < 0.5 m |
| RQ3 | Can DeepLabV3+ crop-health ID reach mIoU > 0.70 within prototype resource limits? | mIoU > 0.70 |
| RQ4 | Can the integrated pipeline sustain real-time closed-loop operation? | Throughput > 5 Hz |

---

## 2. End-to-End System Architecture and Data Flow

The system follows a **five-layer modular pipeline**, ensuring isolated failure domains and classical fallbacks for safety-critical functions:

```
Sensor & Localisation  →  BEV Perception Module  →  Hybrid Planning Module
                                                           ↓
                         Crop Identification Module  ←────┘  (async health map)
                                                           ↓
                                              Logging & Evaluation
```

- **BEV Transformer** runs at 10 Hz; feeds occupancy maps to both planning layers
- **Global Planner** runs at 1 Hz; **Local Planner** at 5–10 Hz; **MPC** at 10–20 Hz
- **Crop Identifier** runs asynchronously; outputs a BEV-registered health map that feeds back to the global planner as a route cost modifier
- Middleware: ROS2 Humble; logging via rosbag2; containerised with Docker on Ubuntu 22.04

---

## 3. BEV Transformer Design

### Sensor Inputs
- Two RGB cameras (front/rear, minimum 1280×720, 10–30 Hz)
- Single 3D LiDAR (30–100 m range), rasterised to a BEV grid
- IMU/GNSS at 100–200 Hz; EKF fusion for ego-motion

### Architecture (Prototype Implementation)

| Stage | Detail |
|-------|--------|
| Camera encoder | 3-stage ConvEncoder → 128-channel feature map from (B, 3, H, W) |
| LiDAR encoder | 3-stage ConvEncoder → 128-channel feature map from (B, 1, H, W) |
| Fusion | Bidirectional 8-head `MultiheadAttention` + LayerNorm on averaged attended streams |
| BEV grid | 200 × 200 cells, 128 channels |
| Output heads | Binary occupancy · 6-class semantic · Confidence scalar · 6-dim detection |

> **Design distinction:** The prototype uses two parallel ConvEncoders with cross-attention. A full BEVFormer-inspired spatiotemporal architecture with multi-scale view transformation targets a future, more capable hardware platform.

### Training Configuration

- 2,000 synthetic samples; 256 px images; batch size 8; 8 epochs
- Adam optimiser, lr = 3 × 10⁻⁴; random seed 42
- Losses: binary cross-entropy + Focal (occupancy); cross-entropy + Dice (segmentation)
- Synthetic data generated in CARLA/Gazebo with domain randomisation

---

## 4. Hybrid Three-Layer Planning

### Layer 1 — Global Coverage Planner (Classical)
- Boustrophedon cellular decomposition; parallel swaths at implement working width
- Frontier-based fill-in for residual uncovered cells
- Route optimisation objective: weighted sum of mission time, energy, obstacle risk, manoeuvre complexity, soil/crop damage — with hard constraints on turning radius, speed, geofence, and clearance

### Layer 2 — Local Trajectory Planner (Hybrid)
- Generates a lattice of **20-step candidates** at 0.2 s per step
- Speed candidates: {0.8, 1.1, 1.4} m/s; curvature candidates: {−0.2, −0.1, 0, 0.1, 0.2}
- Scoring: `S_i = 0.65 · ML_score + 0.35 · (classical_cost)⁻¹` (α = 0.65)
- **Inflated-cell safety gate**: 2-cell margin; any trajectory whose inflated region intersects an occupied cell is rejected
- **Stationary fallback**: invoked when no collision-free trajectory exists

### Layer 3 — MPC Trajectory Tracker
- Solver-free sampled kinematic bicycle controller
- Wheelbase 1.2 m; discretisation 0.1 s; horizon 12 steps
- Control grid: 9 steering × 7 acceleration combinations (63 candidates evaluated per cycle)
- Limits: steering ±0.45 rad; acceleration ±1.2 m/s²
- Cost: weighted position error, heading error, speed tracking, control effort

---

## 5. DeepLabV3+ Agricultural CV and BEV Integration

- **Architecture:** DeepLabV3+ with ImageNet-pretrained EfficientNet-B4 encoder via `segmentation_models_pytorch`; declared fallback to torchvision DeepLabV3 + ResNet-50 when preferred dependency is unavailable
- **Training:** AdamW; cosine annealing with linear warmup; loss = cross-entropy + Dice + Focal
- **Pre-training sources:** PlantVillage (54,306 images, 14 crops, 38 classes) and DeepWeeds (17,509 images, 8 weed species)
- **Domain adaptation protocol:** pre-train on public datasets → fine-tune on field images partitioned by recording day (not random frame sampling, which inflates accuracy)
- **BEV integration:** segmentation masks projected to BEV frame via camera intrinsics/extrinsics and depth from BEV Transformer → persistent geo-referenced health map → anomalous zones receive modified traversal cost in the global planner

---

## 6. Dataset and Training Methodology

| Aspect | Detail |
|--------|--------|
| BEV perception dataset | 2,000 synthetic sample pairs with auto-labelled occupancy masks |
| Generation tool | CARLA / Gazebo with domain randomisation (lighting, texture, camera noise, dust) |
| Crop ID datasets | PlantVillage + DeepWeeds (public); field fine-tuning recommended as next step |
| Software stack | PyTorch 2.0+; torchvision; segmentation-models-pytorch; OpenCV; PCL; ROS2 Humble |
| Testing | pytest unit tests: metrics, global planner swath/frontier fill-in, local planner lattice/safety gate/selection |

---

## 7. Validated Prototype/Simulation Results

**All results are simulation- and prototype-level. Field validation has not been established.**

| Metric | Reported Result | Evidence Status |
|--------|----------------|-----------------|
| BEV mIoU (aggregate) | **> 0.70** | Reported aggregate; per-class breakdown not recorded |
| Collision-free rate | **> 95%** | Reported aggregate; per-scenario breakdown not recorded |
| MPC tracking RMSE | **< 0.5 m** | Reported aggregate; per-trajectory breakdown not recorded |
| Crop ID validation mIoU | **> 0.70** | Reported aggregate; per-class breakdown not recorded |
| Full pipeline throughput | **> 5 Hz** | Reported aggregate |
| Statistical significance | Not recorded | Requires multiple independent runs with variance |

> The minimum viable system targets (mIoU > 0.60, collision-free > 95%, RMSE < 0.5 m, crop mIoU > 0.70) are all met. The "should-have" target of mIoU > 0.70 is also reported as met.

---

## 8. Experimental Design and Ablation Studies

### Completed Validation
- Simulation runs across randomly generated obstacle layouts
- Inflated-cell safety gate triggering stationary fallback confirmed as a hard constraint (not a soft cost term)

### Planned — Not Yet Conducted

| Study | Description | Priority |
|-------|-------------|----------|
| Modality ablation | Camera-only vs. LiDAR-only vs. fused BEV mIoU comparison | Highest |
| Per-class BEV IoU | IoU for each of 6 semantic classes | High |
| Per-scenario collision rate | Breakdown by obstacle density scenario | High |
| Per-module inference latency | Profiling on target hardware | High |
| Coverage fraction / overlap | Field-level coverage measurement | High |
| Temporal fusion ablation | Single-frame vs. multi-frame BEV | Future |

> Literature expectation for modality ablation (grounded in BEVFusion findings): fused configuration expected to outperform either single-modality baseline — camera providing richer semantics, LiDAR providing geometric precision. This expectation is **not yet measured**.

### Recommended Field Protocol
1. Sensor fusion: odometry drift < 5% over 30-minute run (RTK-GNSS ground truth)
2. Coverage planner: ≥ 95% coverage at first integration; ≥ 98% after tuning
3. Obstacle avoidance: zero collision events across repeated runs with known obstacles
4. Safety supervisor: each failure mode induced; verified fallback within 1-second response
5. Domain adaptation: held-out field images partitioned by recording day

---

## 9. Limitations, Risks, and Future Work

### Key Limitations
- **Simulation gap:** real terrain introduces wheel slip, IMU vibration, dust occlusion, seasonal variation, GNSS degradation under canopy — all unmodelled
- **Training data scale:** 2,000 synthetic samples is orders of magnitude smaller than BEVFormer/BEVFusion training sets; expansion to 10,000–20,000 frames recommended
- **Crop ID field generalisation:** unknown; PlantVillage accuracy figures are upper bounds for field performance due to documented background bias; field drop to ~31% accuracy reported in literature
- **Computational feasibility:** per-module inference latency not recorded; deployment on resource-constrained hardware (Raspberry Pi 3B) not yet demonstrated

### Risk Register Summary

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sim-to-real sensor noise gap | High | Early real-world testing; visual SLAM fallback |
| Perception overfits to synthetic data | Medium | Domain randomisation; real data fine-tuning |
| Insufficient BEV training data | High | Expand synthetic dataset; transfer learning |
| Crop ID fails on new species | Medium | Domain adaptation; uncertainty flagging |
| Computational budget exceeded on target hardware | High | Model compression; INT8 quantisation |

### Future Work (Priority Order)
1. Field validation on real agricultural terrain
2. Modality ablation studies (camera vs. LiDAR vs. fused)
3. Agricultural BEV dataset collection from ground vehicles
4. Lightweight embedded deployment characterisation (quantisation, latency profiling)
5. Reinforcement learning integration (SAC) within a verified safe envelope — positioned as future work, not current core
6. Automatic implement docking (6D pose estimation)

### Implementation Context (Six-Month Roadmap)
Months 1–2: foundation and simulation setup; Months 3–4: core BEV and planning development and integration; Month 5: optimisation and comprehensive simulation experiments; Month 6: thesis writing and documentation.

---

## 10. Conclusions

The thesis demonstrates that a compact modular architecture — two parallel ConvEncoders with bidirectional cross-attention, Boustrophedon global coverage, lattice local planning with a learned scorer, inflated-cell safety gate, and sampled MPC — meets all four declared research question targets in simulation:

- **RQ1 Met (sim.):** BEV mIoU > 0.70
- **RQ2 Met (sim.):** Collision-free > 95%; RMSE < 0.5 m
- **RQ3 Met (sim.):** Crop ID mIoU > 0.70
- **RQ4 Met (sim.):** Throughput > 5 Hz

The sim-to-real gap and the absence of field validation remain the principal unresolved threats. The thesis makes a credible contribution to integrating BEV perception, hybrid planning, and agricultural CV, and provides a reproducible experimental foundation for future field validation campaigns.

---

## 11. Recommended Technical Diagrams and Tables for Slides

| Item | Type | Content |
|------|------|---------|
| System pipeline | Vertical flow diagram | Five layers: Sensors → BEV → Planning → Crop ID → Logging, with feedback arrow from Crop ID to Global Planner |
| BEV Transformer | Architecture diagram | Dual ConvEncoders → Bidirectional cross-attention → 200×200 BEV grid → four output heads |
| Hybrid planning safety gate | Flow diagram | Lattice generation → trajectory scoring → inflated-cell gate → stationary fallback or MPC |
| Data-flow diagram | Module interaction map | BEV at 10 Hz; Global at 1 Hz; Local at 5–10 Hz; MPC at 10–20 Hz; Crop ID async |
| BEV architecture comparison table | Comparison table | BEVFormer / BEVDet4D / PETR / BEVFusion / TPVFormer / SparseOcc vs. agricultural suitability |
| Planning approach comparison table | Comparison table | Boustrophedon / Hybrid A* / Lattice / IL scoring / RL / MPC vs. safety and thesis suitability |
| Agricultural CV dataset table | Comparison table | PlantVillage / DeepWeeds / Agriculture-Vision / CropAndWeed vs. size, task, key limitation |
| Configuration parameters table | Parameter table | All BEV, training, local planner, and MPC numerical parameters |
| Aggregate results table | Results summary | Five metrics with reported values and evidence status (per-class breakdown not recorded) |
| Research-objective traceability matrix | Traceability table | RQ → Key Result → Status (Met sim. / Future work) |
| Failure modes and fallbacks table | Risk table | Seven failure modes with detection mechanism and fallback strategy |
| Risk register | Risk table | Six risks with impact, probability, and mitigation |
| Ablation study plan table | Gap table | Five planned ablations with completion status (planned / not conducted) |
| Recommended evaluation metrics table | Protocol table | Per-module metrics, definitions, and future field targets |