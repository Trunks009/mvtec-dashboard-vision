# Dashboard ejecutivo en Streamlit (Fase 4)

Lee los CSV de `Comparativa_Grupo/` (los mismos del dashboard web y del informe). Estilo Tufte: fondo blanco, gris para lo normal, negro para el modelo recomendado y rojo solo para fallos.

## Ejecutar

```
pip install -r requirements.txt
streamlit run app.py
```

Desde esta carpeta (`Comparativa_Grupo/dashboard_streamlit`). El tema claro está fijado en `.streamlit/config.toml`; si se ejecuta desde otra carpeta, Streamlit no lo lee y en modo oscuro las tablas pierden contraste.

En Colab: subir la carpeta `Comparativa_Grupo` completa y usar `streamlit run app.py` con un túnel (por ejemplo `cloudflared` o `localtunnel`).

## Secciones

1. Conclusión generada desde los datos y 5 cifras del modelo recomendado contra la línea base.
2. Ranking global: AUROC de imagen por categoría (marcas) y media (punto).
3. Diagnóstico por categoría con selector de métrica (AUROC imagen/píxel, F1, Dice).
4. Precisión contra velocidad.
5. Pruebas de significancia (Friedman, Wilcoxon-Holm).
6. Robustez ante brillo, rotación y traslación (falsas alarmas o AUROC).
7. Impacto por cada 1 000 piezas con la tasa de defectos como supuesto ajustable (1–20 %).
8. Tipos de defecto más difíciles para el modelo recomendado.

Si cambian los CSV de los integrantes, regenerar las tablas con `../scripts/consolidar.py` y `../scripts/prep_dash.py` (ver `../scripts/LEEME.txt`).
