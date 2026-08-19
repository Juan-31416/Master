# Slide 18

Template: `chart_feature`

One data slide built around a single deterministic chart (bar, horizontal bar, or donut) sitting in a ~62%-wide panel, optionally paired with 1-3 big-number stat cards in a sidebar and a one-line takeaway. Use to make ONE quantitative point land — a trend, a split, a share, a projection — the way a hand-built 'chart + KPI callouts' slide would; reach for it for ANY quantitative comparison of >=3 values (plot it, never list numbers as prose). For headline numbers with no series to plot use kpi_stats or stat_bento; for a parallel set of qualitative ideas use icon_grid or content_body.

Objective: Mostrar el stack tecnológico y las frecuencias de ejecución por módulo

## Filled values

- **eyebrow**: "Implementación"
- **title**: "Stack software y frecuencias por módulo"
- **subtitle**: "ROS2 Humble, PyTorch 2.0+ y contenedores Docker sobre Ubuntu 22.04"
- **chart_data**: {"kind": "bar", "labels": ["BEV Transformer", "Planif. global", "Planif. local (máx.)", "MPC (máx.)", "Pipeline cerrado"], "values": [10, 1, 10, 20, 5], "unit": "Hz"}
- **stats**: [{"value": "Humble", "label": "ROS2 como middleware; registro con rosbag2"}, {"value": "2.0+", "label": "PyTorch, torchvision y segmentation-models"}, {"value": "pytest", "label": "Tests de métricas, swaths y compuerta"}]
- **takeaway**: "Cada capa corre a su propio ritmo; el identificador de cultivo opera de forma asíncrona"
- **footer_left**: "Navegación Autónoma en BEV"
- **footer_right**: "2026"
