# Detección de defectos MVTec AD · dashboard ejecutivo

Comparativa de cinco modelos de detección de anomalías (PatchCore, PaDiM, DFR, autoencoder SSIM y VAE/DAE) y una línea base clásica sobre las 15 categorías de MVTec AD, con una simulación de mantenimiento predictivo (Fase 3).

El dashboard está hecho con Streamlit y lee las tablas de `Comparativa_Grupo/`.

## Ejecutar en local

```
python -m pip install -r requirements.txt
python -m streamlit run Comparativa_Grupo/dashboard_streamlit/app.py
```

## Contenido

- `Comparativa_Grupo/dashboard_streamlit/app.py`: el dashboard.
- `Comparativa_Grupo/*.csv`: métricas consolidadas, resumen por modelo, significancia, robustez y AUROC por tipo de defecto.
- `Comparativa_Grupo/fase3_simulacion/`: simulación de eventos discretos del mantenimiento y sus resultados (`salidas/`). Ver `LEEME.md`.
