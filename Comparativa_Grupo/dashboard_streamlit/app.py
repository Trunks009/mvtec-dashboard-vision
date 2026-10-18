"""Dashboard ejecutivo (Fase 4) · Detección de defectos MVTec AD · comparativa del grupo.

Lee los CSV de Comparativa_Grupo/ (salida de scripts/consolidar.py y scripts/prep_dash.py).
Estilo Tufte: fondo neutro, gris para lo normal, negro para el modelo recomendado y rojo solo para fallos.
Ejecutar:  streamlit run Comparativa_Grupo/dashboard_streamlit/app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATOS = Path(__file__).resolve().parent.parent
OFICIAL = ['patchcore_resnet50', 'padim_vgg16', 'vgg16_dfr', 'ae_conv_ssim', 'vae_dae', 'lb_ensamble']
GRIS, NEGRO, ROJO, ROJO_FONDO = '#9a9a9a', '#1a1a1a', '#c0392b', '#f6dcd9'
PERT = {'limpio': 'Limpio', 'brillo x0.7': 'Brillo ×0.7', 'brillo x1.3': 'Brillo ×1.3',
        'rotacion +8 grados': 'Rotación +8°', 'traslacion +6 px': 'Traslación +6 px'}

st.set_page_config(page_title='Detección de defectos MVTec AD', layout='wide')
st.markdown("""<style>
  .block-container {max-width: 1150px; padding-top: 2rem;}
  h1 {font-weight: 400; font-size: 2rem;} h2 {font-weight: 500; font-size: 1.15rem; margin-top: 1.2rem;}
  [data-testid="stMetricValue"] {font-weight: 400;}
  .nota {color: #6b6b6b; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


def mostrar(ch):
    # sin rejillas ni marcos: solo los datos y etiquetas en gris
    st.altair_chart(ch.configure_view(stroke=None).configure_axis(
        grid=False, domainColor='#cfcfcf', tickColor='#cfcfcf', labelColor='#6b6b6b', titleColor='#6b6b6b',
        titleFontWeight='normal'), width='stretch')


@st.cache_data
def cargar():
    res = pd.read_csv(DATOS / 'resumen_modelos.csv')
    met = pd.read_csv(DATOS / 'metricas_consolidadas_grupo.csv')
    sig = pd.read_csv(DATOS / 'significancia.csv')
    rob = pd.read_csv(DATOS / 'robustez_resumen.csv')
    tip = pd.read_csv(DATOS / 'auroc_por_tipo_defecto.csv')
    res['corto'] = res.nombre.str.split(' · ').str[0].replace({'Línea base clásica': 'Línea base'})
    return res.sort_values('puesto'), met, sig, rob, tip


res, met, sig, rob, tip = cargar()
of = res[res.oficial].copy()
rec = of.iloc[0]
lb = res.set_index('modelo').loc['lb_ensamble']
corto = dict(zip(res.modelo, res.corto))
nombre = dict(zip(res.modelo, res.nombre))
orden_nombres = list(res.nombre)


def grilla(df, fila, col, valor, malo, fmt, orden_filas, orden_cols, alto):
    """Tabla como gráfico: el número en cada celda y fondo rojo solo donde `malo` es verdadero."""
    d = df.assign(_txt=df[valor].map(lambda v: format(v, fmt)), _malo=malo(df))
    base = alt.Chart(d).encode(
        x=alt.X(f'{col}:N', sort=orden_cols, axis=alt.Axis(orient='top', title=None, labelAngle=0, domain=False, ticks=False, labelLimit=140)),
        y=alt.Y(f'{fila}:N', sort=orden_filas, axis=alt.Axis(title=None, domain=False, ticks=False, labelLimit=220)))
    rect = base.mark_rect().encode(color=alt.condition('datum._malo', alt.value(ROJO_FONDO), alt.value('transparent')))
    txt = base.mark_text(fontSize=12).encode(text='_txt:N', color=alt.condition('datum._malo', alt.value(ROJO), alt.value(NEGRO)),
                                             tooltip=[fila, col, alt.Tooltip(valor, format='.3f')])
    return (rect + txt).properties(height=alto)


# --- encabezado y conclusión -----------------------------------------------------------------
st.title('Detección de defectos MVTec AD: comparativa del grupo')
pares = sig[(sig.metrica == 'auroc_img') & sig.par.str.contains(rec.modelo, regex=False)].copy()
pares['otro'] = pares.apply(lambda p: corto[p.par.split('|')[1] if p.par.split('|')[0] == rec.modelo else p.par.split('|')[0]], axis=1)
si, no = pares[pares.significativo == 'sí'], pares[pares.significativo != 'sí']
texto_no = ('; frente a ' + ', '.join(no.otro) + ' la ventaja no es significativa (p = ' +
            ', '.join(f'{p:.2f}' for p in no.p_holm) + ')') if len(no) else ''
st.markdown(f"**{rec.nombre}** es el modelo que mejor generaliza en las {met.categoria.nunique()} categorías: "
            f"AUROC de imagen medio **{rec.auroc_img:.3f}** y rango promedio **{rec.auroc_img_rango6:.2f}** (1 = el mejor en todas). "
            f"Supera con significancia estadística a {', '.join(si.otro)} (Wilcoxon-Holm, p < 0.05){texto_no}.")
st.markdown('<p class="nota">Cinco modelos, uno por integrante, más la línea base clásica, medidos con el mismo protocolo: '
            'imágenes de 256 px, entrenamiento solo con piezas normales y umbral fijado en validación (percentil 99), nunca con el test.</p>',
            unsafe_allow_html=True)

c = st.columns(5)
c[0].metric('AUROC de imagen', f'{rec.auroc_img:.3f}', f'{rec.auroc_img - lb.auroc_img:+.3f} vs. línea base')
c[1].metric('Defectos detectados', f'{rec.recall:.0%}', f'{(rec.recall - lb.recall) * 100:+.0f} pts vs. línea base')
c[2].metric('Falsas alarmas', f'{rec.fpr:.1%}', f'{(rec.fpr - lb.fpr) * 100:+.1f} pts vs. línea base', delta_color='inverse')
c[3].metric('Localización (AUROC píxel)', f'{rec.auroc_px:.3f}', f'{rec.auroc_px - lb.auroc_px:+.3f} vs. línea base')
c[4].metric('Tiempo por imagen', f'{rec.ms_img:.0f} ms', f'{rec.ms_img - lb.ms_img:+.0f} ms vs. línea base', delta_color='inverse')

# --- ranking global --------------------------------------------------------------------------
st.header('Ranking global: cada marca es una categoría, el punto es la media')
st.markdown('<p class="nota">AUROC de imagen en las 15 categorías. Rojo: categorías bajo 0.70, donde el modelo no es confiable. '
            'Línea punteada: azar (0.5). Las filas «LB ·» son las partes de la línea base.</p>', unsafe_allow_html=True)
m = met.assign(bajo=met.auroc_img < 0.7)
y = alt.Y('nombre:N', sort=orden_nombres, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=240))
xs = alt.X('auroc_img:Q', scale=alt.Scale(domain=[0.2, 1]), title='AUROC de imagen')
marcas = alt.Chart(m).mark_tick(thickness=1.5, size=14).encode(
    x=xs, y=y, color=alt.condition('datum.bajo', alt.value(ROJO), alt.value('#c4c4c4')),
    tooltip=['nombre', 'categoria', alt.Tooltip('auroc_img', format='.3f')])
medias = alt.Chart(res.assign(es_rec=res.modelo == rec.modelo)).mark_circle(size=70, opacity=1).encode(
    x='auroc_img:Q', y=y, color=alt.condition('datum.es_rec', alt.value(NEGRO), alt.value('#6b6b6b')),
    tooltip=['nombre', alt.Tooltip('auroc_img', format='.3f', title='media'), alt.Tooltip('auroc_img_std', format='.3f', title='desv.')])
etiq = alt.Chart(res).mark_text(align='left', dx=10, fontSize=11, color='#4a4a4a').encode(
    x=alt.datum(1.0), y=y, text=alt.Text('auroc_img:Q', format='.3f'))
azar = alt.Chart(pd.DataFrame({'x': [0.5]})).mark_rule(strokeDash=[2, 3], color='#9a9a9a').encode(x='x:Q')
mostrar((azar + marcas + medias + etiq).properties(height=300))

# --- diagnóstico por categoría -----------------------------------------------------------------
st.header('Diagnóstico por categoría: dónde falla cada modelo')
etq = {'AUROC imagen': 'auroc_img', 'AUROC píxel': 'auroc_px', 'F1': 'f1', 'Dice': 'dice'}
sel = st.radio('Métrica', list(etq), horizontal=True, label_visibility='collapsed', key='met_cat')
col = etq[sel]
lim = 0.3 if col in ('f1', 'dice') else 0.7
st.markdown(f'<p class="nota">En rojo los valores bajo {lim:.2f}. Es diagnóstico: elegir un modelo distinto por producto '
            'exigiría mirar las anomalías del test, que el protocolo prohíbe.</p>', unsafe_allow_html=True)
mc = met[met.modelo.isin(OFICIAL)].assign(modelo_c=lambda d: d.modelo.map(corto),
                                         cat=lambda d: d.categoria + d.grupo.map({'objeto': '', 'textura': ' (textura)'}))
orden_cat = list(dict.fromkeys(mc.sort_values(['grupo', 'categoria']).cat))
mostrar(grilla(mc, 'cat', 'modelo_c', col, lambda d: d[col] < lim, '.3f', orden_cat,
               [corto[x] for x in of.modelo], 15 * 24))

# --- velocidad y significancia: una sola columna, las tablas no caben en dos ---
st.header('Precisión contra velocidad')
st.markdown('<p class="nota">ms por imagen en escala logarítmica. Arriba a la izquierda es mejor.</p>', unsafe_allow_html=True)
sp = res.assign(es_rec=res.modelo == rec.modelo, etiqueta=res.corto.where(res.oficial, ''))
base = alt.Chart(sp).encode(
    x=alt.X('ms_img:Q', scale=alt.Scale(type='log', domain=[3, 90]), title='ms por imagen'),
    y=alt.Y('auroc_img:Q', scale=alt.Scale(domain=[0.55, 1]), title='AUROC de imagen'))
pts = base.mark_circle(size=80, opacity=1).encode(
    color=alt.condition('datum.es_rec', alt.value(NEGRO), alt.value(GRIS)),
    tooltip=['nombre', alt.Tooltip('auroc_img', format='.3f'), alt.Tooltip('ms_img', format='.1f'), alt.Tooltip('t_ajuste_s', format='.0f', title='ajuste (s)')])
# VAE y DFR quedan muy cerca: la etiqueta de VAE va debajo del punto
lab = base.mark_text(align='left', dx=8, dy=-7, fontSize=11, color='#4a4a4a').encode(
    text='etiqueta:N').transform_filter(alt.datum.modelo != 'vae_dae')
lab_abajo = base.mark_text(align='left', dx=8, dy=12, fontSize=11, color='#4a4a4a').encode(
    text='etiqueta:N').transform_filter(alt.datum.modelo == 'vae_dae')
mostrar((pts + lab + lab_abajo).properties(height=340))
st.header('¿Las diferencias son reales?')
f = sig[sig.metrica == 'auroc_img'].iloc[0]
st.markdown(f'<p class="nota">Friedman sobre las 15 categorías: χ² = {f.friedman_chi2:.1f}, p = {f.friedman_p:.1e}. '
            'Wilcoxon pareado con corrección de Holm; diferencia mediana de AUROC de imagen (A − B).</p>', unsafe_allow_html=True)
t = sig[sig.metrica == 'auroc_img'].assign(A=lambda d: d.a.str.split(' · ').str[0], B=lambda d: d.b.str.split(' · ').str[0])
st.dataframe(t[['A', 'B', 'dif_mediana', 'p_holm', 'significativo']], hide_index=True, width="stretch", height=360,
             column_config={'dif_mediana': st.column_config.NumberColumn('Dif.', format='%+.3f'),
                            'p_holm': st.column_config.NumberColumn('p (Holm)', format='%.3f'),
                            'significativo': '¿Significativa?'})

# --- robustez --------------------------------------------------------------------------------
st.header('Robustez: qué pasa si cambia la luz o la posición de la pieza')
modo = st.radio('Medida', ['Piezas buenas marcadas como defecto', 'AUROC imagen'], horizontal=True, label_visibility='collapsed', key='rob')
r = rob[rob.modelo.isin(OFICIAL)].assign(cond=lambda d: d.perturbacion.map(PERT), modelo_c=lambda d: d.modelo.map(corto))
if modo.startswith('Piezas'):
    st.markdown('<p class="nota">Media de bottle, screw y carpet con el umbral fijado en validación limpia. En rojo, más del 30 % de falsas alarmas.</p>', unsafe_allow_html=True)
    g = grilla(r, 'modelo_c', 'cond', 'fpr', lambda d: d.fpr > 0.3, '.0%', [corto[x] for x in of.modelo], list(PERT.values()), 6 * 30)
else:
    st.markdown('<p class="nota">En rojo, una caída de AUROC mayor a 0.10 frente a la imagen limpia.</p>', unsafe_allow_html=True)
    g = grilla(r, 'modelo_c', 'cond', 'auroc_img', lambda d: d.caida_vs_limpio < -0.1, '.3f', [corto[x] for x in of.modelo], list(PERT.values()), 6 * 30)
mostrar(g)

# --- impacto y defectos difíciles --------------------------------------------------------------
st.header('Impacto en planta por cada 1 000 piezas')
prev = st.slider('Tasa de piezas defectuosas (supuesto)', 1, 20, 5, format='%d %%') / 100
st.markdown('<p class="nota">La tasa de defectos es un supuesto; la detección y las falsas alarmas son las medias medidas en el test. '
            'Revisión manual = alertas que un operario debe confirmar.</p>', unsafe_allow_html=True)
malas = 1000 * prev
imp = of.assign(Detectados=(malas * of.recall).round().astype(int))
imp = imp.assign(Escapan=round(malas) - imp.Detectados, **{'Falsas alarmas': ((1000 - malas) * imp.fpr).round().astype(int)})
imp['Revisión manual'] = imp.Detectados + imp['Falsas alarmas']
st.dataframe(imp.sort_values('Escapan')[['corto', 'Detectados', 'Escapan', 'Falsas alarmas', 'Revisión manual']]
             .rename(columns={'corto': 'Modelo'}), hide_index=True, width="stretch")
st.header('Defectos que el modelo recomendado aún no ve bien')
st.markdown('<p class="nota">Tipos de defecto con menor AUROC, cada uno frente a las piezas normales de su categoría.</p>', unsafe_allow_html=True)
d = tip[tip.modelo == rec.modelo].nsmallest(7, 'auroc')[['categoria', 'tipo', 'n', 'auroc', 'recall']]
st.dataframe(d, hide_index=True, width="stretch",
             column_config={'categoria': 'Categoría', 'tipo': 'Tipo de defecto', 'n': 'Imágenes',
                            'auroc': st.column_config.NumberColumn('AUROC', format='%.3f'),
                            'recall': st.column_config.NumberColumn('Detectadas', format='%.2f')})

# --- Fase 3: simulación de mantenimiento (salidas de fase3_simulacion/simulacion_mantenimiento.py) ---
st.header('Mantenimiento predictivo: qué cambia en la planta (simulación, Fase 3)')
F3 = DATOS / 'fase3_simulacion' / 'salidas'
if not (F3 / 'resumen_escenarios.csv').exists():
    st.info('Falta correr la simulación: python Comparativa_Grupo/fase3_simulacion/simulacion_mantenimiento.py')
else:
    esc = pd.read_csv(F3 / 'resumen_escenarios.csv')
    st.markdown('<p class="nota">Planta simulada de 15 líneas durante un año, 30 réplicas. La detección y las falsas alarmas de cada '
                'sistema salen de sus resultados; la producción, los costos y el desgaste son supuestos (abajo). Cifras de '
                f'{rec.corto} con mantenimiento predictivo frente a hoy (inspección manual y mantenimiento reactivo).</p>',
                unsafe_allow_html=True)
    ix = esc.set_index(['sistema', 'politica'])
    hoy, pc = ix.loc[('manual', 'reactivo')], ix.loc[(rec.modelo, 'predictivo')]
    c = st.columns(5)
    c[0].metric('Costo anual de la planta', f'{pc.costo_total / 1e6:.2f} M USD', f'{(pc.costo_total - hoy.costo_total) / 1e6:+.2f} M vs. hoy', delta_color='inverse')
    c[1].metric('Fallas de máquina al año', f'{pc.fallas:.0f}', f'{pc.fallas - hoy.fallas:+.0f} vs. hoy', delta_color='inverse')
    c[2].metric('Defectos al cliente (ppm)', f'{pc.escapes_ppm:,.0f}', f'{pc.escapes_ppm - hoy.escapes_ppm:+,.0f} vs. hoy', delta_color='inverse')
    c[3].metric('Vida útil de las máquinas', f'{pc.vida_util_anios:.1f} años', f'{pc.vida_util_anios - hoy.vida_util_anios:+.1f} vs. hoy')
    c[4].metric('Disponibilidad', f'{pc.disponibilidad:.1%}', f'{(pc.disponibilidad - hoy.disponibilidad) * 100:+.1f} pts vs. hoy')

    st.markdown('<p class="nota">Costo anual por sistema de inspección: cada fila es un sistema y los tres puntos son sus políticas de '
                'mantenimiento. Línea punteada: costo de hoy. Barras finas: intervalo de confianza del 95 %.</p>', unsafe_allow_html=True)
    e = esc.assign(sis=esc.escenario.str.split(' · ').str[0], costo=esc.costo_total / 1e6,
                   lo=(esc.costo_total - esc.costo_total_ic95) / 1e6, hi=(esc.costo_total + esc.costo_total_ic95) / 1e6,
                   mejor=lambda d: d.costo == d.costo.min())
    orden = e.groupby('sis').costo.min().sort_values().index.tolist()
    y = alt.Y('sis:N', sort=orden, title=None, axis=alt.Axis(domain=False, ticks=False, labelLimit=200, labelOverlap=False))
    xs = alt.Scale(domain=[1, float(e.hi.max()) * 1.02], zero=False)
    rango = alt.Chart(e).mark_rule(color='#e3e3e1', strokeWidth=2).encode(x=alt.X('min(costo):Q', scale=xs), x2='max(costo):Q', y=y)
    ic = alt.Chart(e).mark_rule(color='#c4c4c4').encode(x=alt.X('lo:Q', scale=xs, title='millones de USD al año'), x2='hi:Q', y=y)
    pts = alt.Chart(e).mark_point(filled=True, size=90, stroke='#6b6b6b', strokeWidth=1.2, opacity=1).encode(
        x='costo:Q', y=y,
        fill=alt.Fill('politica:N', scale=alt.Scale(domain=['reactivo', 'calendario', 'predictivo'], range=['#ffffff', '#b5b5b5', '#6b6b6b']),
                      legend=alt.Legend(title=None, orient='top')),
        tooltip=['escenario', alt.Tooltip('costo_total', format=',.0f', title='costo (USD)'), alt.Tooltip('fallas', format='.0f'),
                 alt.Tooltip('escapes_ppm', format=',.0f', title='defectos al cliente (ppm)')])
    mejor = alt.Chart(e[e.mejor]).mark_point(filled=True, size=130, color=NEGRO, opacity=1).encode(x='costo:Q', y=y)
    # a la izquierda del punto: el más barato de su fila no tiene línea a ese lado
    etq_mejor = alt.Chart(e[e.mejor]).mark_text(align='right', dx=-10, fontSize=11, color=NEGRO).encode(
        x='costo:Q', y=y, text=alt.Text('costo:Q', format='.2f'))
    linea_hoy = alt.Chart(pd.DataFrame({'x': [hoy.costo_total / 1e6]})).mark_rule(strokeDash=[2, 3], color='#9a9a9a').encode(x='x:Q')
    mostrar((linea_hoy + rango + ic + pts + mejor + etq_mejor).properties(height=300))

    if (F3 / 'sensibilidad_todos_los_sistemas.csv').exists():
        s = pd.read_csv(F3 / 'sensibilidad_todos_los_sistemas.csv')
        p = s[s.sistema == rec.modelo]
        st.markdown(f'Al mover cada supuesto a la mitad o al 150 %, {rec.corto} con mantenimiento predictivo es el escenario más barato en '
                    f'**{(p.puesto == 1).sum()} de {len(p)}** variaciones, con un ahorro frente a hoy de '
                    f'{p.ahorro_vs_actual.min():,.0f} a {p.ahorro_vs_actual.max():,.0f} USD al año.')
    with st.expander('Ver los supuestos de la simulación'):
        sup = pd.read_csv(F3 / 'supuestos.csv')
        st.dataframe(sup[['descripcion', 'valor', 'unidad', 'justificacion']], hide_index=True, width='stretch',
                     column_config={'descripcion': 'Supuesto', 'valor': 'Valor', 'unidad': 'Unidad', 'justificacion': 'Justificación'})

st.markdown('<p class="nota">Validación: todos los archivos traen las mismas categorías, las mismas imágenes de prueba con las mismas '
            'etiquetas y un umbral fijo por categoría; AUROC, precisión, recall y F1 recalculados desde las predicciones coinciden '
            'con los reportados. Fuente: CSV de los cinco integrantes consolidados en Comparativa_Grupo/.</p>', unsafe_allow_html=True)
