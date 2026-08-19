# Slide 6

Template: `comparison_table`

Full-width data table: header (numbered eyebrow, title, subtitle, gold rule with caption), 4 fixed columns [entity | era/attribute | numeric value + unit | description], 4-6 zebra rows, one row may be flagged for emphasis tint; footer with source note and brand. Use to compare items across a few consistent attributes. Avoid for a two-sided prose contrast (use split_panels) or when fewer than 4 comparable rows exist.

Objective: Situar el prototipo frente a las arquitecturas BEV de referencia y señalar las brechas agrícolas que ninguna cubre

## Filled values

- **eyebrow**: "Revisión de literatura · Percepción BEV"
- **title**: "Estado del arte en BEV"
- **subtitle**: "Arquitecturas de referencia en conducción autónoma y su encaje con un vehículo agrícola"
- **rule_cap**: "Idoneidad agrícola"
- **col1_label**: "Arquitectura"
- **col1_sub**: "Referencia"
- **col2_label**: "Enfoque"
- **col2_sub**: "Cambio de vista"
- **col3_label**: "Modalidad"
- **col3_sub**: "Sensores de entrada"
- **col4_label**: "Encaje"
- **col4_sub**: "Uso en esta tesis"
- **rows**: [{"name": "BEVFormer", "region": "Atención espaciotemporal", "era": "Multiescala", "value": "Cámara", "known": "Inspira una arquitectura futura con más GPU."}, {"name": "BEVDet4D", "region": "Fusión multi-frame", "era": "Temporal", "value": "Cámara", "known": "Justifica una ablación temporal pendiente."}, {"name": "PETR", "region": "Consultas de posición 3D", "era": "Implícito", "value": "Cámara",…
- **source**: "Brechas detectadas: no existe un benchmark BEV agrícola ni un análisis de BEV ligero para hardware embebido."
- **brand**: "Máster en IA"
