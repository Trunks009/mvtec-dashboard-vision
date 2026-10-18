# Fase 3 · Simulación de mantenimiento con las alertas del modelo

Simulación de eventos discretos de una planta con 15 líneas, una por producto de MVTec AD. Responde cuánto cambia el costo, las fallas, la calidad y la vida útil de las máquinas al pasar de mantenimiento reactivo a predictivo, usando como sensor el sistema de inspección visual de cada integrante.

## Ejecutar

Requiere numpy, pandas, scipy y matplotlib (vienen en Colab).

```
python simulacion_mantenimiento.py
```

Tarda unos 8 minutos: 20 escenarios × 30 réplicas, más la sensibilidad (21 variaciones de supuestos × 7 sistemas × 10 réplicas). Opciones:

| Opción | Para qué |
| --- | --- |
| `--replicas 100` | Intervalos de confianza más estrechos (unos 12 min con la sensibilidad) |
| `--sin-sensibilidad` | Solo los escenarios base (unos 2 min) |
| `--replicas-sensibilidad 20` | Más réplicas en cada variación de la sensibilidad |
| `--resultados <carpeta>` | Dónde están los `predicciones_*.csv` (por defecto, la carpeta de resultados del grupo) |
| `--salida <carpeta>` | Dónde escribir (por defecto `salidas/`) |

En Colab: subir `Comparativa_Grupo/` y la carpeta de resultados, y correr `!python simulacion_mantenimiento.py --resultados <ruta>`.

## Qué viene de los datos y qué es supuesto

**De los resultados del grupo:** la probabilidad de que cada sistema detecte una pieza defectuosa y de que marque una buena, por categoría y tipo de defecto (`predicciones_*.csv`), y cómo cambian con 30 % menos de brillo (`robustez_resumen.csv`). En cada réplica estas tasas se sortean de su distribución posterior Beta, para que el tamaño pequeño del test (42 a 167 imágenes por categoría) se refleje en los intervalos.

**Supuestos** (tabla `SUPUESTOS` al inicio del script, exportada a `salidas/supuestos.csv`): producción, tasas de defecto según el desgaste, tiempos de desgaste y reparación, costos y vida útil. Para cambiar uno, se edita su valor en la tabla y se vuelve a correr. El análisis de sensibilidad mueve los 10 más influyentes a la mitad y al 150 % y corre, en cada variación, los 6 sistemas en predictivo contra la situación actual. Así no solo dice si la recomendación ahorra, sino si sigue siendo la mejor cuando cambia cada supuesto.

## Escenarios

- **Sistema de inspección:** inspección manual por muestreo (situación actual), o PatchCore, PaDiM, DFR, VAE/DAE, AE SSIM y la línea base clásica.
- **Política de mantenimiento:**
  - **Reactivo:** solo se repara cuando la máquina falla.
  - **Calendario:** preventivo cada 250 h de operación.
  - **Predictivo:** preventivo cuando las alertas del último turno superan lo esperable con la máquina sana. El umbral se calibra sin defectos, igual que el umbral de imagen del proyecto, para que una máquina sana dispare como mucho un mantenimiento innecesario al año.

## Salidas (`salidas/`)

| Archivo | Contenido |
| --- | --- |
| `resumen_escenarios.csv` | Por escenario: costo anual (total, calidad, paradas, inspección), ahorro frente a la situación actual, escapes (ppm), fallas, preventivos, preventivos con la máquina sana, disponibilidad y vida útil; con IC 95 % |
| `replicas.csv` | Cada réplica, para análisis propios |
| `escapes_por_tipo_defecto.csv` | Escapes anuales esperados por tipo de defecto con política predictiva |
| `sensibilidad_todos_los_sistemas.csv`, `fig_sensibilidad_sistemas.png` | Ahorro y puesto de cada sistema en cada variación de supuestos; en negrita el mejor de cada fila, en rojo si no es PatchCore |
| `sensibilidad.csv`, `fig_sensibilidad.png` | Ahorro de PatchCore predictivo frente a la situación actual con cada supuesto a la mitad y al 150 % |
| `fig_costo_por_escenario.png` | Costo anual de los 20 escenarios |
| `supuestos.csv` | Todos los supuestos con su justificación |

El script empieza con una autoprueba (un detector perfecto no deja escapes, uno nulo no detecta nada, el calendario dispara y el predictivo no dispara sin alertas); si falla, se detiene.
