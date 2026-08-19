# Slide 15

Template: `chart_feature`

One data slide built around a single deterministic chart (bar, horizontal bar, or donut) sitting in a ~62%-wide panel, optionally paired with 1-3 big-number stat cards in a sidebar and a one-line takeaway. Use to make ONE quantitative point land — a trend, a split, a share, a projection — the way a hand-built 'chart + KPI callouts' slide would; reach for it for ANY quantitative comparison of >=3 values (plot it, never list numbers as prose). For headline numbers with no series to plot use kpi_stats or stat_bento; for a parallel set of qualitative ideas use icon_grid or content_body.

Objective: Detallar los parámetros del controlador MPC de seguimiento de trayectoria como panel numérico de un vistazo

## Filled values

- **eyebrow**: "Capa 3 — Control"
- **footer_left**: "Tesis — Navegación BEV"
- **footer_right**: "2026"
- **title**: "MPC: bicicleta cinemática muestreada"
- **subtitle**: "Controlador cinemático muestreado sin solver, evaluado en cada ciclo"
- **chart_data**: {"kind": "bar", "labels": ["Opciones de dirección", "Opciones de aceleración", "Candidatos por ciclo", "Pasos de horizonte"], "values": [9, 7, 63, 12], "unit": "n"}
- **stats**: [{"value": "1,2", "unit": "m", "label": "Batalla del vehículo"}, {"value": "0,1", "unit": "s", "label": "Discretización temporal"}, {"value": "±1,2", "unit": "m/s²", "label": "Límite de aceleración"}]
- **takeaway**: "Barrido exhaustivo de 63 combinaciones por ciclo, ejecutado a 10–20 Hz."
