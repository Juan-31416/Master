# Slide 21

Template: `image_band_stats`

A header (eyebrow + title) over a wide full-width feature image band, with a bottom strip of 2-4 stat items (each an optional icon, one big value and a short supporting label) split by thin hairline dividers. Use to pair one strong supporting photo with a few headline numbers or takeaways that caption it — a strong choice whenever a relevant on-topic photo exists. For a pure numbers slide use kpi_stats; when the image needs explanatory prose beside it rather than numbers under it, use image_feature.

Objective: Presentar los resultados de identificación de cultivo: mIoU > 0,70 en validación y el sesgo de fondo aún pendiente

## Filled values

- **eyebrow**: "Resultados · RQ3"
- **title**: "Identificación de cultivo: mIoU > 0,70 en validación"
- **band**: ""
- **stats**: [{"icon": "leaf", "value": "> 0,70", "label": "mIoU de validación: objetivo RQ3 cumplido en simulación"}, {"icon": "database", "value": "54.306", "label": "Imágenes PlantVillage: 14 cultivos, 38 clases"}, {"icon": "triangle-alert", "value": "~31%", "label": "Precisión en campo reportada en literatura por sesgo de fondo"}]
- **footer_left**: "Navegación Autónoma Híbrida en BEV"
- **footer_right**: "2026"
