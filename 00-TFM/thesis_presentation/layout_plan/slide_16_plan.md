# Slide 16

Template: `image_feature`

Magazine split: a tall half-bleed feature image down one side beside a text column carrying eyebrow, title, an optional intro paragraph, 2-3 optional captioned points and an optional image caption; a footer strip under the text column. Use for a visual break with one strong supporting image + explanatory text — reach for it whenever a relevant on-topic photo supports the point (one real image required). For a full-bleed cover use cover_full_bleed instead.

Objective: Presentar la arquitectura de visión agrícola: DeepLabV3+ con encoder EfficientNet-B4 y su fallback declarado

## Filled values

- **eyebrow**: "Módulo 3 · Visión agrícola"
- **title**: "DeepLabV3+ para salud del cultivo"
- **body**: "Segmentación de cultivo, maleza y anomalías con DeepLabV3+ y encoder EfficientNet-B4, vía segmentation_models_pytorch."
- **points**: [{"heading": "Encoder EfficientNet-B4", "body": "Preentrenado en ImageNet; buen balance precisión-coste"}, {"heading": "Fallback declarado", "body": "torchvision DeepLabV3 + ResNet-50 si falta la dependencia"}, {"heading": "Objetivo RQ3", "body": "mIoU > 0,70 dentro de los límites de recursos del prototipo"}]
- **caption**: "Las máscaras se proyectan al marco BEV con intrínsecos"
- **footer_left**: "Navegación Autónoma en BEV"
- **footer_right**: "2026"
