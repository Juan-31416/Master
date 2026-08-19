# Slide 11

Template: `mosaic_tiles`

An asymmetric editorial mosaic: one tall photo tile, one filled accent tile carrying a hero stat, two surface tiles each holding a short titled point, and one wide dark tile with a takeaway line — five tiles locked into a 2x3 grid under a compact header. The most magazine-like layout in the library; use mid-deck to fuse ONE strong photo + ONE number + two supporting points into a single energetic slide, whenever a relevant on-topic photo and a headline number both exist.

Objective: Explicar la metodología de entrenamiento del BEV Transformer con datos sintéticos y aleatorización de dominio

## Filled values

- **eyebrow**: "Módulo BEV · Entrenamiento"
- **title**: "Datos sintéticos y aleatorización de dominio"
- **stat_value**: "2.000"
- **stat_label**: "Pares sintéticos con máscaras de ocupación autoetiquetadas"
- **points**: [{"heading": "Generación en CARLA/Gazebo", "body": "Aleatoriza iluminación, textura, ruido y polvo para acercar sintético al campo."}, {"heading": "Setup de entrenamiento", "body": "256 px, batch 8, 8 épocas, Adam lr 3x10⁻⁴, semilla 42 reproducible."}]
- **takeaway**: "Pérdidas combinadas: BCE+Focal en ocupación y CE+Dice en segmentación semántica."
- **footer_left**: "Navegación Autónoma en BEV"
- **footer_right**: "2026"
