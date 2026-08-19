# Slide 10

Template: `process_steps`

Vertical numbered steps: header beside 2-5 rows, each a large step number, a bold step name, a one-line description and an optional right-hand aside. Use for UNDATED mechanisms and methods — how something works or gets made (brewing, enrollment, a pipeline), instructions, sequences of actions. The moment items carry dates or years it is a chronology: use timeline_vertical/timeline_phases instead; for a forward plan with timeframes use roadmap.

Objective: Explicar el diseño interno de la red BEV: encoders duales, fusión por atención cruzada, rejilla y cabezas de salida

## Filled values

- **eyebrow**: "Módulo 1 · BEV Transformer"
- **title**: "Diseño de la red BEV"
- **subtitle**: "Dos ConvEncoders paralelos, atención cruzada bidireccional y cuatro cabezas de salida"
- **steps**: [{"no": "01", "name": "Encoder de cámara", "detail": "ConvEncoder de 3 etapas sobre (B,3,H,W): mapa de características de 128 canales"}, {"no": "02", "name": "Encoder LiDAR", "detail": "LiDAR 3D rasterizado a rejilla BEV; ConvEncoder de 3 etapas a 128 canales"}, {"no": "03", "name": "Fusión por atención cruzada", "detail": "MultiheadAttention bidireccional de 8 cabezas + LayerNorm sobre flujos pro…
- **footer_left**: "Navegación Autónoma Híbrida en BEV"
- **footer_right**: "2026"
