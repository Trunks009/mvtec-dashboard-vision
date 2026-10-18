"""Fase 3 · Simulación de eventos discretos: de mantenimiento reactivo a predictivo con las alertas del modelo de visión.

Planta: 15 líneas, una por producto de MVTec AD. Cada línea tiene una máquina que se desgasta
(sana -> desgaste leve -> desgaste severo -> falla). Cuanto más desgastada, más piezas defectuosas produce.
Al final de cada línea, un sistema de inspección (manual por muestreo o uno de los 6 sistemas del grupo)
marca piezas como defectuosas.

Qué sale de los DATOS del grupo (predicciones_*.csv y robustez_resumen.csv):
  - probabilidad de detectar una pieza defectuosa, por sistema, categoría y tipo de defecto
  - probabilidad de falsa alarma sobre una pieza buena, por sistema y categoría
  - cómo cambian ambas con 30 % menos de brillo (deriva de iluminación)
  - mezcla de tipos de defecto por categoría
  La incertidumbre por tamaño de muestra se propaga: en cada réplica las tasas se sortean de su posterior
  Beta(k + 0.5, n - k + 0.5) (prior de Jeffreys), en vez de tomar el valor puntual del test.
Qué es SUPUESTO (tabla SUPUESTOS, exportada a supuestos.csv, con análisis de sensibilidad):
  producción, tasas de defecto por estado de desgaste, tiempos de desgaste y reparación, costos y vida útil.

Políticas de mantenimiento comparadas:
  reactivo    se repara solo cuando la máquina falla (situación actual)
  calendario  mantenimiento preventivo cada `intervalo_calendario_h` horas de operación
  predictivo  mantenimiento cuando las alertas de las últimas `ventana_h` horas superan un umbral calibrado solo con
              la máquina sana (como el umbral del proyecto, que sale de piezas normales). Como se revisa cada hora,
              el percentil es 1 − `disparos_falsos_anio` / `horas_anio`: con un percentil 99 por revisión, una máquina
              sana dispararía decenas de mantenimientos innecesarios al año

Avance del tiempo en pasos fijos de 1 h (fixed-increment time advance, Law & Kelton); fallas, reparaciones,
mantenimientos y derivas de iluminación son eventos que cambian el estado de cada línea dentro de ese reloj.

Uso (desde cualquier carpeta; requiere numpy, pandas, scipy y matplotlib, todos incluidos en Colab):
  python simulacion_mantenimiento.py                        # 30 réplicas + sensibilidad, unos 3 minutos
  python simulacion_mantenimiento.py --replicas 100         # intervalos de confianza más estrechos
  python simulacion_mantenimiento.py --sin-sensibilidad     # solo los escenarios base
  python simulacion_mantenimiento.py --resultados <carpeta con predicciones_*.csv> --salida <carpeta>
"""
import argparse
import glob
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

AQUI = Path(__file__).resolve().parent
CATS = ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
        'carpet', 'grid', 'leather', 'tile', 'wood']
SISTEMAS = {'patchcore_resnet50': 'PatchCore', 'padim_vgg16': 'PaDiM', 'vgg16_dfr': 'DFR', 'vae_dae': 'VAE / DAE',
            'ae_conv_ssim': 'AE SSIM', 'lb_ensamble': 'Línea base clásica'}
POLITICAS = ['reactivo', 'calendario', 'predictivo']
RECOMENDADO = 'patchcore_resnet50'

# ------------------------------------------------------------------------------------------------------------
# SUPUESTOS: lo que no está en los resultados del grupo. Cambiar aquí; todo se exporta a supuestos.csv.
# ------------------------------------------------------------------------------------------------------------
SUPUESTOS = [
    # clave, valor, unidad, descripción, justificación
    ('horas_anio', 4000, 'h/año', 'Horas de operación por línea', '2 turnos de 8 h, 250 días'),
    ('piezas_h', 200, 'piezas/h', 'Producción por línea', 'Una pieza cada 18 s; muy por debajo de la capacidad del sistema más lento (PatchCore: 65 ms/imagen, unas 55 000 piezas/h)'),
    ('p_def_0', 0.005, 'fracción', 'Piezas defectuosas con la máquina sana', 'Proceso estable: 5 000 ppm'),
    ('p_def_1', 0.02, 'fracción', 'Piezas defectuosas con desgaste leve', '4 veces la tasa sana'),
    ('p_def_2', 0.08, 'fracción', 'Piezas defectuosas con desgaste severo', '16 veces la tasa sana, antes de la falla'),
    ('media_h_0', 300, 'h', 'Tiempo medio sana antes de pasar a desgaste leve', 'Exponencial'),
    ('media_h_1', 150, 'h', 'Tiempo medio en desgaste leve antes de desgaste severo', 'Exponencial'),
    ('media_h_2', 60, 'h', 'Tiempo medio en desgaste severo antes de la falla', 'Exponencial; ciclo medio sin mantenimiento de unas 510 h'),
    ('correctivo_media_h', 8, 'h', 'Duración media de una reparación por falla', 'Lognormal, incluye diagnóstico y repuestos'),
    ('correctivo_desv_h', 4, 'h', 'Desviación de la duración de la reparación', ''),
    ('preventivo_h', 2, 'h', 'Duración de un mantenimiento preventivo', 'Planificado, con repuestos a mano'),
    ('intervalo_calendario_h', 250, 'h', 'Intervalo del preventivo por calendario', 'Poco menos que la mitad del ciclo medio hasta la falla'),
    ('ventana_h', 8, 'h', 'Ventana de alertas que vigila la política predictiva', 'Un turno'),
    ('disparos_falsos_anio', 1, 'por línea/año', 'Mantenimientos innecesarios tolerados con la máquina sana', 'Fija el umbral del predictivo: percentil 1 − 1/4 000 por revisión horaria'),
    ('p_deriva_h', 0.001, 'prob./h', 'Probabilidad por hora de una deriva de iluminación (−30 % de brillo)', 'Una cada 1 000 h por línea; el efecto en cada sistema sale de la robustez medida'),
    ('deriva_h', 24, 'h', 'Duración de la deriva si nadie recalibra', 'Se corrige al día siguiente o en el próximo mantenimiento'),
    ('manual_fraccion', 0.10, 'fracción', 'Piezas que revisa el inspector en la situación actual', 'Muestreo del 10 %'),
    ('manual_recall', 0.80, 'fracción', 'Defectos que ve el inspector en las piezas que revisa', 'Inspección visual humana'),
    ('manual_fpr', 0.01, 'fracción', 'Piezas buenas que el inspector rechaza', ''),
    ('costo_inspeccion_manual', 0.25, 'USD/pieza', 'Costo de revisar a mano una pieza del muestreo', '30 s a 30 USD/h'),
    ('costo_escape', 10, 'USD/pieza', 'Pieza defectuosa que llega al cliente', 'Devolución, reposición y reclamo'),
    ('costo_descarte', 2, 'USD/pieza', 'Pieza defectuosa detectada y descartada', 'Material y proceso perdidos'),
    ('costo_revision_alerta', 0.5, 'USD/alerta', 'Revisar a mano una pieza marcada por el sistema', '1 min a 30 USD/h'),
    ('costo_h_no_plan', 1000, 'USD/h', 'Hora de parada no planificada', 'Producción perdida y personal ocioso'),
    ('costo_h_plan', 400, 'USD/h', 'Hora de parada planificada', '40 % de la no planificada: se reprograma la producción'),
    ('repuestos_correctivo', 2000, 'USD/evento', 'Repuestos y servicio en una falla', ''),
    ('repuestos_preventivo', 300, 'USD/evento', 'Repuestos en un preventivo', ''),
    ('costo_fijo_auto', 4000, 'USD/línea/año', 'Cámara, iluminación y GPU amortizadas para inspección automática', ''),
    ('vida_nominal_h', 40000, 'h-equivalentes', 'Vida útil de la máquina operando siempre sana', '10 años a 4 000 h/año'),
    ('desgaste_mult_1', 1.5, 'veces', 'Desgaste por hora en desgaste leve frente a sana', ''),
    ('desgaste_mult_2', 3.0, 'veces', 'Desgaste por hora en desgaste severo frente a sana', ''),
    ('dano_falla_h', 300, 'h-equivalentes', 'Vida consumida por cada falla', 'Daño adicional de operar hasta romper'),
]
S0 = {k: v for k, v, *_ in SUPUESTOS}

# supuestos que se mueven en la sensibilidad (×0.5 y ×1.5) y cómo se aplican
SENSIBILIDAD = {'piezas_h': 'Producción por hora', 'p_def': 'Tasas de defecto', 'media_h': 'Tiempos de desgaste',
                'correctivo_media_h': 'Duración de una reparación', 'costo_escape': 'Costo de un defecto que escapa',
                'costo_h_no_plan': 'Costo de una hora de parada', 'costo_revision_alerta': 'Costo de revisar una alerta',
                'p_deriva_h': 'Frecuencia de derivas de luz', 'ventana_h': 'Ventana del vigilante predictivo',
                'costo_fijo_auto': 'Costo fijo del sistema automático'}


def escalar(S, clave, f):
    S = dict(S)
    grupo = {'p_def': ['p_def_0', 'p_def_1', 'p_def_2'], 'media_h': ['media_h_0', 'media_h_1', 'media_h_2']}
    for k in grupo.get(clave, [clave]):
        S[k] = S[k] * f
    S['ventana_h'] = max(1, int(round(S['ventana_h'])))
    return S


# ------------------------------------------------------------------------------------------------------------
# Datos del grupo
# ------------------------------------------------------------------------------------------------------------
def cargar_datos(dir_res, ruta_rob):
    pred = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(dir_res, 'predicciones_*.csv'))])
    pred = pred[pred.modelo.isin(SISTEMAS)]
    faltan = set(SISTEMAS) - set(pred.modelo)
    assert not faltan, f'Faltan predicciones de {faltan} en {dir_res}'
    malas, buenas = pred[pred.etiqueta == 1], pred[pred.etiqueta == 0]
    idx = pd.MultiIndex.from_product([list(SISTEMAS), CATS], names=['modelo', 'categoria'])
    det = malas.groupby(['modelo', 'categoria']).pred.agg(['sum', 'count']).reindex(idx)
    fa = buenas.groupby(['modelo', 'categoria']).pred.agg(['sum', 'count']).reindex(idx)
    assert not det.isna().any().any() and not fa.isna().any().any(), 'Hay categorías sin predicciones'
    forma = (len(SISTEMAS), len(CATS))
    tasas = {'k_det': det['sum'].to_numpy().reshape(forma), 'n_def': det['count'].to_numpy().reshape(forma),
             'k_fa': fa['sum'].to_numpy().reshape(forma), 'n_norm': fa['count'].to_numpy().reshape(forma)}

    # tipos de defecto: mezcla por categoría (del test) y detección por tipo
    tipos = malas.groupby(['modelo', 'categoria', 'tipo_defecto']).pred.agg(['sum', 'count']).reset_index()
    tipos['recall'] = (tipos['sum'] + 0.5) / (tipos['count'] + 1)

    rob = pd.read_csv(ruta_rob).set_index(['modelo', 'perturbacion'])
    deriva = {}
    for m in SISTEMAS:
        limpio, oscuro = rob.loc[(m, 'limpio')], rob.loc[(m, 'brillo x0.7')]
        # con menos luz: la falsa alarma pasa a la medida con brillo ×0.7 y la detección se escala por la misma razón
        deriva[m] = dict(fpr=float(oscuro.fpr), factor_recall=float(oscuro.recall / max(limpio.recall, 1e-6)))
    return tasas, tipos, deriva


def sortear_tasas(tasas, rng):
    """Una réplica de recall y FPR por sistema y categoría desde su posterior Beta (prior de Jeffreys)."""
    r = rng.beta(tasas['k_det'] + 0.5, tasas['n_def'] - tasas['k_det'] + 0.5)
    f = rng.beta(tasas['k_fa'] + 0.5, tasas['n_norm'] - tasas['k_fa'] + 0.5)
    return r, f


# ------------------------------------------------------------------------------------------------------------
# Simulación de una planta (15 líneas en paralelo) durante un año
# ------------------------------------------------------------------------------------------------------------
def simular(recall, fpr, politica, S, rng, recall_deriva=None, fpr_deriva=None, horas=None):
    L, H, W = len(recall), int(horas or S['horas_anio']), int(S['ventana_h'])
    recall_deriva = recall if recall_deriva is None else recall_deriva
    fpr_deriva = fpr if fpr_deriva is None else fpr_deriva
    p_def = np.array([S['p_def_0'], S['p_def_1'], S['p_def_2']])
    p_sube = 1 - np.exp(-1 / np.array([S['media_h_0'], S['media_h_1'], S['media_h_2']]))
    mult = np.array([1.0, S['desgaste_mult_1'], S['desgaste_mult_2']])
    sigma = np.sqrt(np.log(1 + (S['correctivo_desv_h'] / S['correctivo_media_h']) ** 2))
    mu = np.log(S['correctivo_media_h']) - sigma ** 2 / 2
    n_h = int(S['piezas_h'])
    # umbral predictivo por línea: percentil de las alertas esperadas en W horas con la máquina sana y luz normal
    p_sana = p_def[0] * recall + (1 - p_def[0]) * fpr
    umbral = binom.ppf(1 - S['disparos_falsos_anio'] / S['horas_anio'], W * n_h, p_sana)

    estado = np.zeros(L, int)
    parada_rest = np.zeros(L, int)          # horas que faltan de parada
    parada_tipo = np.zeros(L, int)          # 1 correctiva, 2 preventiva
    deriva_rest = np.zeros(L, int)
    h_desde_mant = np.zeros(L, int)
    ventana = np.zeros((W, L), int)
    c = {k: np.zeros(L) for k in ['piezas', 'defectuosas', 'detectadas', 'escapes', 'falsas_alarmas', 'fallas',
                                  'preventivos', 'preventivos_sana', 'h_no_plan', 'h_plan', 'desgaste', 'h_deriva']}
    for h in range(H):
        parada = parada_rest > 0
        c['h_no_plan'] += parada & (parada_tipo == 1)
        c['h_plan'] += parada & (parada_tipo == 2)
        parada_rest -= parada
        op = ~parada

        # producción e inspección de la hora
        n = np.where(op, n_h, 0)
        d = rng.binomial(n, p_def[estado])
        en_deriva = deriva_rest > 0
        det = rng.binomial(d, np.where(en_deriva, recall_deriva, recall))
        fa = rng.binomial(n - d, np.where(en_deriva, fpr_deriva, fpr))
        ventana[h % W] = det + fa
        for k, v in (('piezas', n), ('defectuosas', d), ('detectadas', det), ('escapes', d - det), ('falsas_alarmas', fa)):
            c[k] += v
        c['desgaste'] += op * mult[estado]
        c['h_deriva'] += op & en_deriva
        h_desde_mant += op

        # evento: deriva de iluminación
        deriva_rest = np.maximum(deriva_rest - 1, 0)
        nueva = op & (deriva_rest == 0) & (rng.random(L) < S['p_deriva_h'])
        deriva_rest[nueva] = int(S['deriva_h'])

        # evento: avance del desgaste y falla
        sube = op & (rng.random(L) < p_sube[estado])
        estado = estado + sube
        falla = estado == 3
        if falla.any():
            parada_rest[falla] = np.ceil(rng.lognormal(mu, sigma, falla.sum())).astype(int)
            parada_tipo[falla] = 1
            c['fallas'] += falla
            c['desgaste'] += falla * S['dano_falla_h']

        # evento: mantenimiento preventivo según la política
        libre = op & ~falla
        if politica == 'calendario':
            dispara = libre & (h_desde_mant >= S['intervalo_calendario_h'])
        elif politica == 'predictivo':
            dispara = libre & (h_desde_mant >= W) & (ventana.sum(0) > umbral)
        else:
            dispara = np.zeros(L, bool)
        if dispara.any():
            parada_rest[dispara] = int(S['preventivo_h'])
            parada_tipo[dispara] = 2
            c['preventivos'] += dispara
            c['preventivos_sana'] += dispara & (estado == 0)  # mantenimiento sin desgaste: alerta falsa del vigilante

        # toda intervención deja la máquina sana, revisa la iluminación y reinicia la vigilancia
        rep = falla | dispara
        estado[rep] = 0
        deriva_rest[rep] = 0
        h_desde_mant[rep] = 0
        ventana[:, rep] = 0
    return c


def costo(c, S, manual):
    alertas = c['detectadas'] + c['falsas_alarmas']
    parada = (c['h_no_plan'] * S['costo_h_no_plan'] + c['h_plan'] * S['costo_h_plan'] +
              c['fallas'] * S['repuestos_correctivo'] + c['preventivos'] * S['repuestos_preventivo'])
    calidad = c['escapes'] * S['costo_escape'] + c['detectadas'] * S['costo_descarte']
    if manual:
        inspeccion = c['piezas'] * S['manual_fraccion'] * S['costo_inspeccion_manual'] + c['falsas_alarmas'] * S['costo_revision_alerta']
    else:
        inspeccion = alertas * S['costo_revision_alerta'] + S['costo_fijo_auto']
    return dict(costo_parada=parada.sum(), costo_calidad=calidad.sum(), costo_inspeccion=inspeccion.sum(),
                costo_total=(parada + calidad + inspeccion).sum())


def kpis(c, S, manual):
    L, H = len(c['piezas']), S['horas_anio']
    k = {k: v.sum() for k, v in c.items() if k not in ('desgaste',)}
    k.update(costo(c, S, manual))
    k['disponibilidad'] = 1 - (k['h_no_plan'] + k['h_plan']) / (L * H)
    k['escapes_ppm'] = 1e6 * k['escapes'] / max(k['piezas'], 1)
    k['vida_util_anios'] = S['vida_nominal_h'] / c['desgaste'].mean()  # desgaste medio por línea en un año simulado
    return k


def correr_escenario(sistema, politica, rep, S, datos, semilla):
    """Una réplica de un escenario. Mismas semillas por réplica en todos los escenarios (números aleatorios comunes)."""
    tasas, _, deriva = datos
    r_all, f_all = sortear_tasas(tasas, np.random.default_rng([semilla, rep, 1]))
    rng = np.random.default_rng([semilla, rep, 2])
    if sistema == 'manual':
        r = np.full(len(CATS), S['manual_fraccion'] * S['manual_recall'])
        f = np.full(len(CATS), S['manual_fraccion'] * S['manual_fpr'])
        c = simular(r, f, politica, S, rng)
    else:
        i = list(SISTEMAS).index(sistema)
        r, f = r_all[i], f_all[i]
        dv = deriva[sistema]
        c = simular(r, f, politica, S, rng, recall_deriva=np.minimum(r * dv['factor_recall'], 1),
                    fpr_deriva=np.maximum(f, dv['fpr']))
    k = kpis(c, S, sistema == 'manual')
    k.update(sistema=sistema, politica=politica, replica=rep)
    return k, c


# ------------------------------------------------------------------------------------------------------------
# Autoprueba: casos donde el resultado se sabe de antemano
# ------------------------------------------------------------------------------------------------------------
def autoprueba():
    S = dict(S0, p_deriva_h=0)
    rng = np.random.default_rng(0)
    uno, cero = np.ones(len(CATS)), np.zeros(len(CATS))
    c = simular(uno, cero, 'reactivo', S, rng, horas=500)
    assert c['escapes'].sum() == 0 and c['falsas_alarmas'].sum() == 0, 'detector perfecto con escapes o falsas alarmas'
    c = simular(cero, cero, 'reactivo', S, rng, horas=500)
    assert c['detectadas'].sum() == 0 and c['escapes'].sum() == c['defectuosas'].sum(), 'detector nulo detectó algo'
    c = simular(uno, cero, 'calendario', dict(S, intervalo_calendario_h=10, media_h_0=1e9), rng, horas=100)
    assert c['fallas'].sum() == 0 and c['preventivos'].min() >= 7, 'el calendario no dispara'
    c = simular(cero, cero, 'predictivo', S, rng, horas=300)
    assert c['preventivos'].sum() == 0, 'el predictivo disparó sin ninguna alerta'
    assert (c['piezas'] <= S['piezas_h'] * 300).all()
    print('Autoprueba OK')


# ------------------------------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------------------------------
def nombre_esc(sistema, politica):
    return ('Inspección manual' if sistema == 'manual' else SISTEMAS[sistema]) + ' · ' + politica


def resumir(df):
    num = ['costo_total', 'costo_calidad', 'costo_parada', 'costo_inspeccion', 'escapes', 'escapes_ppm', 'falsas_alarmas',
           'fallas', 'preventivos', 'preventivos_sana', 'h_no_plan', 'h_plan', 'disponibilidad', 'vida_util_anios', 'defectuosas', 'detectadas']
    g = df.groupby(['sistema', 'politica'])[num]
    media, ee = g.mean(), g.std() / np.sqrt(g.size().to_numpy()[:, None])
    res = media.copy()
    for col in ['costo_total', 'escapes', 'fallas', 'disponibilidad', 'vida_util_anios']:
        res[col + '_ic95'] = 1.96 * ee[col]
    base = res.loc[('manual', 'reactivo'), 'costo_total']
    res['ahorro_vs_actual'] = base - res['costo_total']
    res['ahorro_pct'] = res['ahorro_vs_actual'] / base
    res = res.reset_index()
    res.insert(0, 'escenario', [nombre_esc(s, p) for s, p in zip(res.sistema, res.politica)])
    return res.sort_values('costo_total')


def escapes_por_tipo(datos, por_cat):
    """Escapes anuales esperados por tipo de defecto: escapes de la categoría repartidos según la mezcla de tipos
    del test y la detección medida de cada tipo."""
    _, tipos, _ = datos
    filas = []
    for (m, cat), g in tipos.groupby(['modelo', 'categoria']):
        if m not in por_cat:
            continue
        peso = g['count'] * (1 - g['recall'])
        esc_cat = por_cat[m][CATS.index(cat)]
        for (_, t), w in zip(g.iterrows(), peso / max(peso.sum(), 1e-12)):
            filas.append(dict(sistema=m, categoria=cat, tipo_defecto=t.tipo_defecto, recall_tipo=round(t.recall, 3),
                              escapes_anuales=round(esc_cat * w, 1)))
    return pd.DataFrame(filas).sort_values(['sistema', 'escapes_anuales'], ascending=[True, False])


def grafico_costos(res, ruta):
    r = res.sort_values('costo_total', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(r) + 1.2))
    y = np.arange(len(r))
    rec = (r.sistema == RECOMENDADO) & (r.politica == 'predictivo')
    actual = r.loc[(r.sistema == 'manual') & (r.politica == 'reactivo'), 'costo_total'].iloc[0] / 1e6
    ax.axvline(actual, color='#9a9a9a', lw=0.8, ls=(0, (2, 3)))
    ax.text(actual, len(r) - 0.3, ' situación actual', color='#6b6b6b', fontsize=8, va='bottom')
    ax.errorbar(r.costo_total / 1e6, y, xerr=r.costo_total_ic95 / 1e6, fmt='none', ecolor='#c4c4c4', lw=1)
    ax.scatter(r.costo_total / 1e6, y, s=np.where(rec, 36, 22), c=np.where(rec, '#1a1a1a', '#8a8a8a'), zorder=3)
    ax.set_yticks(y, r.escenario, fontsize=8)
    for t, es in zip(ax.get_yticklabels(), rec):
        t.set_color('#1a1a1a' if es else '#6b6b6b')
        t.set_fontweight('bold' if es else 'normal')
    ax.set_xlabel('Costo anual de la planta (millones de USD, media e IC 95 %)', color='#6b6b6b', fontsize=9)
    ax.set_title('Costo anual por sistema de inspección y política de mantenimiento', loc='left', fontsize=11)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#cfcfcf')
    ax.tick_params(colors='#6b6b6b', length=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def grafico_sensibilidad(sens, ruta):
    s = sens.assign(rango=lambda d: (d.ahorro_alto - d.ahorro_bajo).abs()).sort_values('rango')
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(s) + 1.3))
    y = np.arange(len(s))
    base = s.ahorro_base.iloc[0] / 1e6
    ax.axvline(base, color='#9a9a9a', lw=0.8, ls=(0, (2, 3)))
    ax.axvline(0, color='#c0392b', lw=0.8)
    ax.hlines(y, s.ahorro_bajo / 1e6, s.ahorro_alto / 1e6, color='#c4c4c4', lw=6)
    ax.scatter(s.ahorro_bajo / 1e6, y, c='#8a8a8a', s=18, zorder=3, label='supuesto × 0.5')
    ax.scatter(s.ahorro_alto / 1e6, y, c='#1a1a1a', s=18, zorder=3, label='supuesto × 1.5')
    ax.set_yticks(y, s.supuesto.map(SENSIBILIDAD), fontsize=8)
    ax.set_xlabel('Ahorro anual de PatchCore predictivo frente a la situación actual (millones de USD)', color='#6b6b6b', fontsize=9)
    lo, hi = s[['ahorro_bajo', 'ahorro_alto']].min().min() / 1e6, s[['ahorro_bajo', 'ahorro_alto']].max().max() / 1e6
    veredicto = 'nunca baja de cero' if lo > 0 else 'puede volverse negativo'
    fig.suptitle(f'Con cada supuesto a la mitad o al 150 %, el ahorro de PatchCore predictivo\nva de {lo:.1f} a {hi:.1f} M USD al año: {veredicto}',
                 x=0.01, ha='left', fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#cfcfcf')
    ax.tick_params(colors='#6b6b6b', length=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160)
    plt.close(fig)


def grafico_sensibilidad_sistemas(sens, ruta):
    """Tabla-gráfico: ahorro anual (M USD) de cada sistema en cada variación; negrita el más barato de la fila,
    rojo si el más barato no es la recomendación."""
    filas = list(dict.fromkeys(zip(sens.supuesto, sens.factor)))
    cols = list(SISTEMAS)
    fig, ax = plt.subplots(figsize=(9, 0.26 * len(filas) + 1.4))
    ax.axis('off')
    x0, dx, y0, dy = 0.44, 0.105, 0.93, 0.86 / len(filas)
    corto = dict(SISTEMAS, lb_ensamble='Línea base')
    ax.text(0, 0.99, 'Ahorro anual frente a la situación actual (millones de USD), mantenimiento predictivo',
            fontsize=10, transform=ax.transAxes, va='top')
    for j, m in enumerate(cols):
        ax.text(x0 + j * dx, y0 + 0.01, corto[m], fontsize=7.5, ha='right', color='#6b6b6b', transform=ax.transAxes)
    for i, (k, f) in enumerate(filas):
        y = y0 - (i + 1) * dy
        etiqueta = 'Supuestos base' if k == 'base' else f'{SENSIBILIDAD[k]} × {f:g}'
        ax.text(0, y, etiqueta, fontsize=7.5, color='#1a1a1a' if k == 'base' else '#4a4a4a', transform=ax.transAxes,
                fontweight='bold' if k == 'base' else 'normal')
        g = sens[(sens.supuesto == k) & (sens.factor == f)].set_index('sistema')
        mejor = g.ahorro_vs_actual.idxmax()
        for j, m in enumerate(cols):
            es_mejor = m == mejor
            color = '#c0392b' if es_mejor and m != RECOMENDADO else ('#1a1a1a' if es_mejor else '#8a8a8a')
            ax.text(x0 + j * dx, y, f'{g.ahorro_vs_actual[m] / 1e6:.2f}', fontsize=7.5, ha='right', color=color,
                    fontweight='bold' if es_mejor else 'normal', transform=ax.transAxes)
    fig.savefig(ruta, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--resultados', default=str(AQUI.parent.parent / 'Resultados de todos los intetgrantes' / 'Resultados de lo integrantes'))
    ap.add_argument('--robustez', default=str(AQUI.parent / 'robustez_resumen.csv'))
    ap.add_argument('--salida', default=str(AQUI / 'salidas'))
    ap.add_argument('--replicas', type=int, default=30)
    ap.add_argument('--replicas-sensibilidad', type=int, default=10)
    ap.add_argument('--semilla', type=int, default=42)
    ap.add_argument('--sin-sensibilidad', action='store_true')
    a = ap.parse_args()
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)

    autoprueba()
    datos = cargar_datos(a.resultados, a.robustez)
    pd.DataFrame(SUPUESTOS, columns=['clave', 'valor', 'unidad', 'descripcion', 'justificacion']).to_csv(out / 'supuestos.csv', index=False)

    # 1) escenarios base
    escenarios = [('manual', p) for p in ('reactivo', 'calendario')] + [(s, p) for s in SISTEMAS for p in POLITICAS]
    t0, filas, esc_cat = time.time(), [], {}
    for i, (s, p) in enumerate(escenarios, 1):
        acum = np.zeros(len(CATS))
        for rep in range(a.replicas):
            k, c = correr_escenario(s, p, rep, S0, datos, a.semilla)
            filas.append(k)
            acum += c['escapes']
        if p == 'predictivo':
            esc_cat[s] = acum / a.replicas
        print(f'  [{i:2d}/{len(escenarios)}] {nombre_esc(s, p):40s} {time.time() - t0:6.0f} s')
    reps = pd.DataFrame(filas)
    reps.to_csv(out / 'replicas.csv', index=False)
    res = resumir(reps)
    res.round(4).to_csv(out / 'resumen_escenarios.csv', index=False)
    escapes_por_tipo(datos, esc_cat).to_csv(out / 'escapes_por_tipo_defecto.csv', index=False)
    grafico_costos(res, out / 'fig_costo_por_escenario.png')

    # 2) sensibilidad: cada supuesto a la mitad y al 150 %, con los 6 sistemas en predictivo contra la situación actual,
    #    para ver si la recomendación sigue siendo la mejor y no solo si ahorra
    if not a.sin_sensibilidad:
        variaciones = [('base', 1.0)] + [(k, f) for k in SENSIBILIDAD for f in (0.5, 1.5)]
        filas_s = []
        for j, (clave, f) in enumerate(variaciones, 1):
            S = S0 if clave == 'base' else escalar(S0, clave, f)
            costos = {sis: np.mean([correr_escenario(sis, pol, r, S, datos, a.semilla)[0]['costo_total']
                                    for r in range(a.replicas_sensibilidad)])
                      for sis, pol in [('manual', 'reactivo')] + [(m, 'predictivo') for m in SISTEMAS]}
            orden = sorted(SISTEMAS, key=costos.get)
            for m in SISTEMAS:
                filas_s.append(dict(supuesto=clave, factor=f, sistema=m, costo_total=costos[m],
                                    ahorro_vs_actual=costos['manual'] - costos[m], puesto=orden.index(m) + 1))
            print(f'  sensibilidad [{j:2d}/{len(variaciones)}] {clave:22s} x{f:<4} mejor: {SISTEMAS[orden[0]]:12s} {time.time() - t0:6.0f} s')
        sens = pd.DataFrame(filas_s)
        sens.round({'costo_total': 0, 'ahorro_vs_actual': 0}).to_csv(out / 'sensibilidad_todos_los_sistemas.csv', index=False)
        grafico_sensibilidad_sistemas(sens, out / 'fig_sensibilidad_sistemas.png')
        # tornado de la recomendación, con las mismas corridas
        r = sens[sens.sistema == RECOMENDADO].set_index(['supuesto', 'factor']).ahorro_vs_actual
        tornado = pd.DataFrame([dict(supuesto=k, ahorro_base=r[('base', 1.0)], ahorro_bajo=r[(k, 0.5)], ahorro_alto=r[(k, 1.5)])
                                for k in SENSIBILIDAD])
        tornado.round(0).to_csv(out / 'sensibilidad.csv', index=False)
        grafico_sensibilidad(tornado, out / 'fig_sensibilidad.png')
        gana = (sens[sens.sistema == RECOMENDADO].puesto == 1)
        print(f'\n{SISTEMAS[RECOMENDADO]} es el más barato en {gana.sum()} de {len(gana)} variaciones de supuestos.')
        otros = sens[(sens.puesto == 1) & (sens.sistema != RECOMENDADO)]
        for _, o in otros.iterrows():
            print(f'  gana {SISTEMAS[o.sistema]} con {SENSIBILIDAD[o.supuesto]} × {o.factor}')

    # 3) lectura en consola
    pd.set_option('display.width', 200)
    cols = ['escenario', 'costo_total', 'ahorro_pct', 'escapes_ppm', 'fallas', 'preventivos', 'disponibilidad', 'vida_util_anios']
    print('\nResumen (planta de 15 líneas, un año, media de', a.replicas, 'réplicas):')
    print(res[cols].to_string(index=False, formatters={'costo_total': '{:,.0f}'.format, 'ahorro_pct': '{:+.1%}'.format,
                                                       'escapes_ppm': '{:,.0f}'.format, 'fallas': '{:.1f}'.format,
                                                       'preventivos': '{:.1f}'.format, 'disponibilidad': '{:.2%}'.format,
                                                       'vida_util_anios': '{:.1f}'.format}))
    print('\nArchivos en', out)


if __name__ == '__main__':
    main()
