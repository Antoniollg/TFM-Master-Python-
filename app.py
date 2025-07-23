import streamlit as st

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Análisis Sevilla FC", layout="wide")

# --- USUARIOS PERMITIDOS ---
USUARIOS = {
    "admin": "admin1",
    "usuario": "usuario1"
}

# --- LOG IN STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- LOG IN ---
if not st.session_state.logged_in:
    st.title("Iniciar sesión")
    usuario = st.text_input("Usuario")
    contraseña = st.text_input("Contraseña", type="password")
    if st.button("Acceder"):
        if USUARIOS.get(usuario) == contraseña:
            st.session_state.logged_in = True
            st.success("Acceso concedido. Cargando aplicación...") 
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos")
    st.stop()

# --- LOGOUT ---
if st.session_state.get("logged_in"):
    with st.sidebar:
        if st.button("🔒 Log out"):
            st.session_state.clear() 
            st.rerun() 

# --- BARRA DE NAVEGACIÓN CON TABS ---
tab1, tab2, tab3 = st.tabs(["Comparación externa", "Comparación interna", "Jugadores 20-21 vs 24-25"])

# --- Cargar y procesar los datos ---

import pandas as pd

# Cargar el archivo de datos
partidos = pd.read_excel("partidos_sevilla.xlsx")

# Extraer y limpiar jornadas
partidos['Jornada'] = partidos['Ronda'].str.extract(r'Semana (\d+)')
partidos['Jornada'] = pd.to_numeric(partidos['Jornada'], errors='coerce').dropna().astype(int)

# Calcular puntos por resultado
def calcular_puntos(resultado):
    if resultado == 'V': return 3
    elif resultado == 'E': return 1
    elif resultado == 'D': return 0
    return None

partidos['Puntos'] = partidos['Resultado'].apply(calcular_puntos)

# Asignar entrenador según jornada
partidos['Entrenador'] = partidos['Jornada'].apply(lambda j: 'Pimienta' if j < 31 else 'Caparrós')

# Filtrar por entrenador
pimienta = partidos[partidos['Jornada'] <= 31]
caparros = partidos[partidos['Jornada'] >= 32]

# Función para estadística extendida
def calcular_estadisticas_extendidas(df):
    total_partidos = len(df)
    pts = df['Puntos'].sum()
    victorias = len(df[df['Resultado'] == 'V'])
    empates = len(df[df['Resultado'] == 'E'])
    derrotas = len(df[df['Resultado'] == 'D'])
    
    gf = df['GF'].sum()
    gc = df['GC'].sum()
    xg = df['xG'].sum()
    xga = df['xGA'].sum()
    
    asistencia_prom = df['Asistencia'].mean()
    
    pct_v = victorias / total_partidos if total_partidos else 0
    pct_e = empates / total_partidos if total_partidos else 0
    pct_d = derrotas / total_partidos if total_partidos else 0
    xg_mp = xg / total_partidos if total_partidos else 0
    xga_mp = xga / total_partidos if total_partidos else 0
    dg = gf - gc
    d_xg = xg - xga
    dg_mp = dg / total_partidos if total_partidos else 0
    d_xg_mp = d_xg / total_partidos if total_partidos else 0
    formacion_mas = df['Formación'].mode().iloc[0] if not df['Formación'].mode().empty else None
    local = len(df[df['Sedes'] == 'Local'])
    visitante = len(df[df['Sedes'] == 'Visitante'])

    return {
        'Partidos': total_partidos,
        'Puntos totales': round(pts, 2),
        'Victorias (%)': round(pct_v * 100, 2),
        'Empates (%)': round(pct_e * 100, 2),
        'Derrotas (%)': round(pct_d * 100, 2),
        'GF totales': round(gf, 2),
        'GC totales': round(gc, 2),
        'Diferencia goles': round(dg, 2),
        'Diferencia goles/MP': round(dg_mp, 2),
        'xG totales': round(xg, 2),
        'xGA totales': round(xga, 2),
        'Diferencia xG': round(d_xg, 2),
        'Diferencia xG/MP': round(d_xg_mp, 2),
        'xG/MP': round(xg_mp, 2),
        'xGA/MP': round(xga_mp, 2),
        'Asistencia promedio': round(asistencia_prom, 0),
        'Formación más usada': formacion_mas,
        'Partidos Local': local,
        'Partidos Visitante': visitante,
    }

# Generar stats
stats_pimienta = calcular_estadisticas_extendidas(pimienta)
stats_caparros = calcular_estadisticas_extendidas(caparros)

# --- Grupos de métricas para gráficas ---

generales = ['Puntos totales', 'Partidos', 'Victorias (%)', 'Empates (%)', 'Derrotas (%)', 'Asistencia promedio']
ofensivas = ['GF totales', 'xG totales', 'Diferencia goles', 'xG/MP']
defensivas = ['GC totales', 'xGA totales', 'Diferencia xG', 'xGA/MP']

# Función para convertir a dataframe long-form
def crear_df_grupo(medidas, stats1, stats2, nombre1='García Pimienta', nombre2='Joaquín Caparrós'):
    df = pd.DataFrame({
        'Métrica': medidas,
        nombre1: [stats1[m] for m in medidas],
        nombre2: [stats2[m] for m in medidas]
    })
    return df.melt(id_vars='Métrica', var_name='Entrenador', value_name='Valor')

# Crear DataFrames
df_generales = crear_df_grupo(generales, stats_pimienta, stats_caparros)
df_ofensivas = crear_df_grupo(ofensivas, stats_pimienta, stats_caparros)
df_defensivas = crear_df_grupo(defensivas, stats_pimienta, stats_caparros)

# Función para graficar
import plotly.express as px
def graficar_grupo(df, titulo):
    fig = px.bar(
        df,
        y='Métrica',
        x='Valor',
        color='Entrenador',
        barmode='group',
        orientation='h',
        title=titulo,
        labels={'Valor': 'Valor', 'Métrica': 'Métrica', 'Entrenador': 'Entrenador'},
        hover_data={'Valor': ':.2f'}
    )
    return fig

with tab1:
    st.header("Comparación externa")

    texto_intro_tab1 = """
    **Estudio externo:**  
    El bloque de análisis externo se centra en situar el rendimiento del Sevilla FC en el contexto del resto de equipos de LaLiga durante la temporada 2024-2025.

    El objetivo principal es identificar en qué aspectos el Sevilla ha estado por encima, por debajo o en la media de la competición, para entender mejor su posición en la clasificación y su comportamiento global.

    Para ello, se han seleccionado dos subconjuntos de equipos como referencia:

    Los equipos que han descendido, como representación del rendimiento más bajo de la competición.

    Los equipos que han terminado en los puestos 4º y 5º, como ejemplo de rendimiento competitivo alto, clasificados a la Champions League.

    La comparación se lleva a cabo utilizando variables ofensivas, defensivas y de contexto (como la asistencia al estadio).
    """

    texto_descendidos_1 = """
        En este apartado se comparan los datos estadísticos del Sevilla FC con la media de los equipos que han descendido esta temporada.

        El propósito es identificar si existen similitudes en métricas clave que puedan explicar un rendimiento bajo del Sevilla, o si, por el contrario, el club se ha mantenido alejado de ese nivel en variables significativas.

        Esta comparación resulta especialmente relevante para evaluar si el Sevilla ha coqueteado con la zona de descenso por motivos estructurales (ofensivos, defensivos, de puntos...) o si su posición final es consecuencia de otros factores puntuales o contextuales.
    """

    texto_descendidos_2 = """
        Este gráfico refleja perfectamente los datos anteriores, así como las diferencias en ciertos parámetros que han podido ser esenciales para la permanencia del Sevilla en primera división. Hay que tener en cuenta que, al ser la media de los 3 equipos que descendieron pueden desvirtuarse un poco los datos ya que el Valladolid (equipo que quedó último en la clasificación) tiene valores muy negativos en comparación con los otros 2 equipos que desdcendieron, por ejemplo la diferencia de goles (GD) fue de -64 lo que hace que la media descienda drásticamente.
                    
        En general, comparado con la media de los tres equipos descendidos en la temporada 2024-2025, el Sevilla FC presenta mejores registros en casi todaslas métricas clave: puntos, goles a favor, goles en contra, estadísticas avanzadas (xG, xGA) y asistencia. Sin embargo, su rendimiento sigue siendo discreto, con un diferencia de goles negativo y una media de solo 1.08 puntos por partido, lo que lo sitúa más cerca del descenso que de competiciones europeas. Ahora veremos qué tan lejos de las competiciones europeas se ha quedado en lo que a estadísticas se refiere.
    """

    texto_champions_1 = """
        Este análisis pone el foco en comparar al Sevilla con la media de los equipos que han finalizado en las posiciones 4ª y 5ª, es decir, justo por debajo de los tres grandes, y que suelen representar un nivel alto de competitividad en la liga.

        El objetivo es analizar en qué aspectos concretos el Sevilla se ha distanciado de estos equipos con buen rendimiento, tanto si hablamos de eficaciaofensiva, consistencia defensiva o acumulación de puntos.
                    
        Esta comparación permite también valorar si el Sevilla se encuentra estructuralmente lejos de los puestos europeos, o si se trata de una cuestión de detalles o eficiencia en momentos clave de la temporada.
    """

    texto_champions_2 = """
        En el ámbito defensivo vemos que el Sevilla es claramente peor que los equipos de champions, no solo por los goles recibidos, sino por la diferencia total de goles a lo largo de la temporada, cuya diferencia respecto a los equipos de champions es de unos 35.
    """

    texto_radar = """
        Aquí se evidencia la falta de acierto de cara a gol, ya no solo por los goles anotados si no por los goles esperados (que también tiene un valor bajo comparado a los equipos de la champions), el problema no parece limitarse solo a mala suerte en algunos partidos sino que parece que hay pocas opciones esperables de que se marquen goles, lo que repercute claramente en los partidos ganados y los puntos promediados en la temporada.
    """
    
    texto_resumen_champions = """  
        Con estos datos y gráficos podemos llegar a la conclusión de que el Sevilla esta temporada ha estado muy lejos del nivel de equipos de champions perdiendo, con notable diferencia, en la mayoría de los apartados estudiados. Siendo poco contuntente en el área rival a la hora de convertir goles y concediendo muchos más goles que la media estudiada.
    """

    # Texto fijo arriba
    st.markdown(texto_intro_tab1)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
    import time
    import io
    import os
    import tempfile
    from fpdf import FPDF
    import base64

    temp_files_to_delete = []


    # --- INICIALIZAR ESTADO ---
    if "show_descendidos" not in st.session_state:
        st.session_state.show_descendidos = False
    if "show_champions" not in st.session_state:
        st.session_state.show_champions = False

    # --- FUNCIONES PARA TOGGLE ---
    def toggle_descendidos():
        st.session_state.show_descendidos = not st.session_state.show_descendidos

    def toggle_champions():
        st.session_state.show_champions = not st.session_state.show_champions

    def exportar_pdf(fig1, fig2, fig_radar):
        import tempfile
        import os
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        texto_intro = """
        Análisis del rendimiento del Sevilla FC
        Temporada 2024-2025
        ----------------------------------------

        El documento recoge un análisis comparativo entre el Sevilla FC y los equipos descendidos, así
        como los clasificados en posiciones de Champions League.
        """
    
        for linea in texto_intro.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        temp_files_to_delete = []

        # Guardar fig1 (matplotlib)
        for linea in texto_descendidos_1.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        if fig1 is not None:
            tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig1.savefig(tmp1.name, bbox_inches='tight')
            tmp1.close()
            pdf.add_page()
            pdf.image(tmp1.name, x=10, w=190)
            temp_files_to_delete.append(tmp1.name)

        for linea in texto_descendidos_2.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        # Guardar fig2 (plotly)
        for linea in texto_champions_1.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        if fig2 is not None:
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig2.write_image(tmp2.name)
            tmp2.close()
            pdf.add_page()
            pdf.image(tmp2.name, x=10, w=190)
            temp_files_to_delete.append(tmp2.name)

        for linea in texto_champions_1.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        # Guardar fig_radar (plotly)
        if fig_radar is not None:
            tmp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig_radar.write_image(tmp3.name)
            tmp3.close()
            pdf.add_page()
            pdf.image(tmp3.name, x=10, w=190)
            temp_files_to_delete.append(tmp3.name)
        
        for linea in texto_radar.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())
        
        for linea in texto_resumen_champions.strip().split('\n'):
            pdf.multi_cell(0, 10, linea.strip())

        # Guardar el PDF final en archivo temporal
        pdf_output = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(pdf_output.name)
        pdf_output.close()

        # Eliminar archivos temporales de imagen
        for path in temp_files_to_delete:
            try:
                os.remove(path)
            except Exception as e:
                print(f"No se pudo eliminar {path}: {e}")

        return pdf_output.name


    
    # --- EXTRACCIÓN DE DATOS ---
    options = Options()
    options.add_argument("--headless")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    url = "https://fbref.com/en/comps/12/2024-2025/2024-2025-La-Liga-Stats"
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    driver.quit()

    tablas = pd.read_html(html)
    for tabla in tablas:
        if "Squad" in tabla.columns:
            df = tabla
            break

    descendidos = ["Leganés", "Las Palmas", "Valladolid"]
    champions = ["Athletic Club", "Villarreal"]

    sevilla_df = df[df["Squad"] == "Sevilla"]
    descendidos_df = df[df["Squad"].isin(descendidos)]
    champions_df = df[df["Squad"].isin(champions)]

    cols_num = df.select_dtypes(include=["number"]).columns
    media_descendidos = descendidos_df[cols_num].mean().round(2)
    media_champions = champions_df[cols_num].mean().round(2)
    sevilla_stats = sevilla_df[cols_num].iloc[0].round(2)

    # Variables globales para las figuras (inicialmente None)
    fig1 = None
    fig2 = None
    fig_radar = None

# --- CONTENIDO DINÁMICO ---

    if st.session_state.show_descendidos:
        st.markdown(texto_descendidos_1)  
        labels = ["GF", "GA", "GD", "Pts", "xG", "xGA", "xGD"]
        sev_values = sevilla_stats[labels].values
        desc_values = media_descendidos[labels].values
        x = np.arange(len(labels))
        width = 0.35
        fig1, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, sev_values, width, label="Sevilla", color='royalblue')
        ax.bar(x + width/2, desc_values, width, label="Descendidos (Media)", color='orangered')
        ax.set_ylabel("Valor")
        ax.set_title("Comparación Sevilla vs Descendidos")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        st.pyplot(fig1)
        st.markdown(texto_descendidos_2) 
    else:
        fig1 = None

    if st.session_state.show_champions:
        st.markdown(texto_champions_1)  
        comparacion_df = pd.DataFrame({
            "Estadística": cols_num,
            "Sevilla": sevilla_stats.values.flatten(),
            "Top 4-5": media_champions.values.flatten()
        })
        cols_def = ["GA", "xGA", "GD", "xGD", "xGD/90"]
        df_graf2 = comparacion_df[comparacion_df["Estadística"].isin(cols_def)]
        color_discrete_map = {
        "Sevilla": "royalblue",
        "Top 4-5": "lightblue"
        }
        fig2 = px.bar(
            df_graf2.melt(id_vars="Estadística", var_name="Equipo", value_name="Valor"),
            x="Estadística", y="Valor", color="Equipo", barmode="group",
            title="Estadísticas defensivas: Sevilla vs equipos Champions",
            color_discrete_map=color_discrete_map
        )
        st.plotly_chart(fig2)
        st.markdown(texto_champions_2)  

        # Radar
        cols_radar = ["GF", "xG", "Pts/MP", "W", "Attendance"]
        sev_vals_raw = sevilla_stats[cols_radar].values.flatten()
        top_vals_raw = media_champions[cols_radar].values.flatten()
        raw_data = np.vstack([sev_vals_raw, top_vals_raw])
        df_raw = pd.DataFrame(raw_data, columns=cols_radar)
        df_normalized = df_raw / df_raw.max()
        sev_radar = df_normalized.iloc[0].tolist() + [df_normalized.iloc[0].tolist()[0]]
        top_radar = df_normalized.iloc[1].tolist() + [df_normalized.iloc[1].tolist()[0]]
        theta = cols_radar + [cols_radar[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=sev_radar, theta=theta, fill='toself', name='Sevilla', line=dict(color='red')))
        fig_radar.add_trace(go.Scatterpolar(r=top_radar, theta=theta, fill='toself', name='Media Top 4-5', line=dict(color='blue')))
        fig_radar.update_layout(
            title="Comparación ofensiva proporcional: Sevilla vs Media Top 4-5",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig_radar)
        st.markdown(texto_radar)
        st.markdown(texto_resumen_champions)
    else:
        fig2 = None
        fig_radar = None


# --- BOTONES --- (SIN INDENTACIÓN, FUERA DE LOS if/else)
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            label_descendidos = "Mostrar menos detalles (descenso)" if st.session_state.show_descendidos else "Comparación con los equipos descendidos"
            st.button(label_descendidos, key="btn_descendidos", on_click=toggle_descendidos)

        with col2:
            label_champions = "Mostrar menos detalles (champions)" if st.session_state.show_champions else "Comparación con los equipos de champions"
            st.button(label_champions, key="btn_champions", on_click=toggle_champions)

        with col3:
            if st.button("Exportar a PDF"):
                print(f"fig1 is {fig1}")
                print(f"fig2 is {fig2}")
                print(f"fig_radar is {fig_radar}")

                if fig1 is None and fig2 is None and fig_radar is None:
                    st.error("No hay figuras para exportar")
                else:
                    pdf_path = exportar_pdf(fig1, fig2, fig_radar)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📄 Descargar PDF",
                            data=f,
                            file_name="analisis_sevilla_fc.pdf",
                            mime="application/pdf"
                        )
                    os.remove(pdf_path)


def graficar_grupo(df, titulo):
    color_map = {
        "García Pimienta": "#1f77b4",
        "Joaquín Caparrós": "#ff7f0e"
    }
    fig = px.bar(
        df,
        x='Métrica',
        y='Valor',
        color='Entrenador',
        barmode='group',
        color_discrete_map=color_map
    )
    fig.update_layout(title=titulo)
    return fig

with tab2:
    opcion = st.radio("Selecciona tipo de comparación interna:", ["Entrenadores", "Temporada 20-21"], horizontal=True)
    
    st.markdown("""
    ### 🧠 Comparación interna del Sevilla FC

    La segunda parte del estudio se centra en una comparación interna del propio Sevilla FC, buscando entender su rendimiento desde dos ángulos: el impacto del cambio de entrenador durante la temporada, y la evolución del equipo en relación con su propio pasado reciente.

    Este enfoque introspectivo permite detectar cambios estructurales, tácticos o de rendimiento que no dependen de factores externos, sino de decisiones propias del club, como el cuerpo técnico o la gestión deportiva.

    En definitiva, el análisis interno busca explicar hasta qué punto la temporada del Sevilla ha estado condicionada por factores internos y cómo ha evolucionado el equipo a lo largo del tiempo y de los diferentes liderazgos técnicos.
    """)

    if opcion == "Entrenadores":

        st.header("📊 Comparación interna - Entrenadores del Sevilla 24/25")

        st.markdown("""
        Este apartado se centra en analizar el rendimiento del Sevilla bajo la dirección de los dos entrenadores que han estado al frente del equipo durante la temporada 2024-2025: Francisco García Pimienta y Joaquín Caparrós.

        Se comparan métricas clave durante sus respectivos periodos para determinar si hubo una mejora o empeoramiento tras el cambio de técnico, y en qué áreas concretas (ofensiva, defensiva, resultados, etc.) se notaron más diferencias.

        Este tipo de análisis permite valorar el impacto real de los entrenadores y la eficacia de las decisiones tomadas por el club en cuanto a su dirección técnica.
        """)

        # Métricas clave
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Puntos por partido", round(stats_pimienta['Puntos totales']/stats_pimienta['Partidos'], 2),
                    delta=f"{round((stats_pimienta['Puntos totales']/stats_pimienta['Partidos']) - (stats_caparros['Puntos totales']/stats_caparros['Partidos']), 2)} vs Caparrós")

        col2.metric("⚽️ xG por partido", round(stats_pimienta['xG/MP'], 2),
                    delta=f"{round(stats_pimienta['xG/MP'] - stats_caparros['xG/MP'], 2)} vs Caparrós")

        col3.metric("🛡️ GC por partido", round(stats_pimienta['GC totales']/stats_pimienta['Partidos'], 2),
                    delta=f"{round((stats_pimienta['GC totales']/stats_pimienta['Partidos']) - (stats_caparros['GC totales']/stats_caparros['Partidos']), 2)} vs Caparrós")

        # Gráficas

        st.subheader("📌 Estadísticas Generales")
        fig_generales = graficar_grupo(df_generales, "Estadísticas Generales")
        st.plotly_chart(fig_generales, use_container_width=True)

        st.subheader("⚔️ Estadísticas Ofensivas")
        fig_ofensivas = graficar_grupo(df_ofensivas, "Estadísticas Ofensivas")
        st.plotly_chart(fig_ofensivas, use_container_width=True)

        st.subheader("🛡️ Estadísticas Defensivas")
        fig_defensivas = graficar_grupo(df_defensivas, "Estadísticas Defensivas")
        st.plotly_chart(fig_defensivas, use_container_width=True)

        # Comentario del analista
        show_comentario = st.toggle("🧠 Mostrar comentario del analista")
        comentario = """
1. **Estadísticas Generales**

En el primer gráfico vemos claramente que García Pimienta y Joaquín Caparrós tienen diferencias notables en el rendimiento general. Pimienta ha jugado más partidos (31 vs 7 para Caparrós), con una diferencia lógica en puntos totales acumulados. Sin embargo, Caparrós presenta un porcentaje bastante peor en cuanto a derrotas se refiere ya que perdió 4 de los 7 encuentros que disputó. Puede deberse también al número tan reducido de partidos el hecho de que sea tan notorio el porcentaje. La asistencia promedio muestra un ligero aumento en los partidos dirigidos por Caparrós (probablemente por ser un entrenador que ya estuvo en el pasado en el club, al cual se le tiene en alta estima por su compromiso), lo que podría indicar un mayor interés o expectación en su etapa.

2. **Estadísticas Ofensivas**

En el apartado ofensivo, García Pimienta destaca con un mayor total de goles a favor y goles esperados (xG) así como la diferencia de goles más favorable, indicando un ataque más productivo o con más oportunidades generadas. Algo importante a tener en cuenta son los goles esperados por partido, en ese apartado sale más favorecido Joaquín Caparrós ya que tiene un valor de 1.51 goles esperados por partido.

3. **Estadísticas Defensivas**

En defensa, la balanza se inclina hacia Caparrós, que registra menos goles en contra y menor xGA (goles esperados en contra), sugiriendo una defensa más sólida o un enfoque táctico más conservador. La diferencia en xG defensiva (xG - xGA) muestra un mejor control defensivo para Caparrós, así como un menor promedio de goles encajados por partido. Esto puede reflejar una mejora en la organización defensiva durante su mandato, aunque no pareció ser suficiente como para tener buenos resultados en los partidos.

4. **Conclusión rápida**

Pimienta muestra un perfil más ofensivo, con más goles y mayor generación de oportunidades aunque Caparrós tiene mejores proporciones en este aspecto.

Caparrós fortalece la defensa y consigue mejores resultados en cuanto a la proporción de goles encajados por partido.
"""
        if show_comentario:
            st.markdown("### 🧠 Comentario del analista")
            st.markdown(comentario)

        # Exportar todo a PDF
        if st.button("⬇️ Exportar análisis a PDF"):
            from fpdf import FPDF
            from datetime import datetime
            import tempfile
            import os
            import plotly.io as pio

            tmp_dir = tempfile.gettempdir()

            # Exportar gráficos a imagen
            images = []
            for fig, name in zip([fig_generales, fig_ofensivas, fig_defensivas], ["generales", "ofensivas", "defensivas"]):
                path = os.path.join(tmp_dir, f"{name}.png")
                fig.write_image(path, format="png", width=800, height=500)
                images.append(path)

            # Crear PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            intro = """Comparación Interna del Sevilla FC - Análisis por Entrenador

La segunda parte del estudio se centra en una comparación interna del propio Sevilla FC, buscando entender su rendimiento desde dos ángulos: el impacto del cambio de entrenador durante la temporada, y la evolución del equipo en relación con su propio pasado reciente.

Este apartado se centra en analizar el rendimiento del Sevilla bajo la dirección de Francisco García Pimienta y Joaquín Caparrós.
"""
            for line in intro.split("\n"):
                pdf.multi_cell(0, 8, line)
            pdf.ln(4)

            for img in images:
                pdf.image(img, x=10, w=190)
                pdf.ln(5)

            if show_comentario:
                pdf.add_page()
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Comentario del analista", ln=True)
                pdf.set_font("Arial", size=12)
                for line in comentario.strip().split("\n"):
                    pdf.multi_cell(0, 8, line)

            output_path = os.path.join(tmp_dir, f"analisis_sevilla_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            pdf.output(output_path)

            with open(output_path, "rb") as f:
                st.download_button("📄 Descargar PDF", data=f, file_name="analisis_entrenadores_sevilla.pdf", mime="application/pdf")


    elif opcion == "Temporada 20-21":
        st.subheader("📊 Comparación con Temporada 20-21")

        st.markdown("""
        En esta sección se analiza la evolución del Sevilla comparando su desempeño actual con el de la temporada 2020-2021, una campaña que se toma como referencia reciente de rendimiento sólido del club.

        El objetivo es observar las diferencias estadísticas significativas que han tenido lugar en estos años, e identificar posibles retrocesos o cambios de enfoque en el juego del equipo.

        Esta comparación longitudinal ayuda a contextualizar si la temporada 2024-2025 representa una anomalía puntual o una tendencia descendente sostenida.
        """)

        metricas = ['Victorias', 'Empates', 'Derrotas', 'Goles a Favor', 'Goles en Contra',
                    'Diferencia de Gol', 'Puntos Totales', 'Puntos por Partido', 'xG', 'xGA', 'xGD']

        temporada_2020 = [24, 5, 9, 53, 33, 20, 77, 2.03, 54.4, 34.8, 19.6]
        temporada_2024 = [10, 11, 17, 42, 55, -13, 41, 1.08, 42.7, 47.4, -4.7]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=metricas,
            y=temporada_2020,
            name='Temporada 2020-21',
            marker_color='rgb(26, 118, 255)',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            x=metricas,
            y=temporada_2024,
            name='Temporada 2024-25',
            marker_color='rgb(255, 99, 71)',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title='Comparación Sevilla FC – Temporadas 2020-21 vs 2024-25',
            xaxis_title='Estadísticas',
            yaxis_title='Valor',
            barmode='group',
            template='plotly_white',
            autosize=False,
            width=1100,
            height=600,
            legend=dict(x=0.02, y=1.08, orientation='h'),
        )

        st.plotly_chart(fig, use_container_width=True)

        show_comentario_21 = st.checkbox("🧠 Mostrar comentario del analista")

        comentario_21 = """
    La comparación entre ambas temporadas muestra una caída significativa en el rendimiento global del equipo.

    **Resultados**: El Sevilla 2020-21 logró 24 victorias y 77 puntos, frente a solo 10 victorias y 41 puntos en 2024-25. La media de puntos por partido se redujo de 2.03 a 1.08, reflejo de una pérdida clara de competitividad.

    **Producción ofensiva y defensiva**: El equipo de 2020-21 marcó 11 goles más y encajó 22 goles menos que en 2024-25, lo que se traduce en una diferencia de gol de +20 frente a -13, es notoria la diferencia.

    **Estadísticas esperadas (xG y xGA)**: Las métricas avanzadas también respaldan este bajón. El xG cayó de 54.4 a 42.7, y el xGA aumentó de 34.8 a 47.4, lo que indica una menor generación de ocasiones de calidad y una mayor exposición defensiva.

    **Balance general (xGD)**: El diferencial esperado de goles (xGD) pasó de +19.6 en 2020-21 a -4.7 en 2024-25, señalando un empeoramiento preocupante en el rendimiento estructural del equipo en todas las líneas.

    **Conclusión**: La temporada 2024-25 muestra un retroceso en prácticamente todas las métricas clave, tanto en resultados reales como en expectativas estadísticas. El equipo ha sido menos eficaz, menos competitivo y más vulnerable en defensa, lo cual marca una diferencia muy clara con respecto a la solidez y regularidad que tuvo en la 2020-21, eso ha hecho que el equipo pase de una sólida cuarta posición y asegurando la champions vía liga a una decimoséptima al borde del descenso.
    """

        if show_comentario_21:
            st.markdown("### 🧠 Comentario del analista")
            st.markdown(comentario_21)

        if st.button("⬇️ Exportar análisis a PDF"):
            from fpdf import FPDF
            from datetime import datetime
            import tempfile
            import os

            def limpiar_texto(texto):
                reemplazos = {
                    "–": "-",
                    "’": "'",
                    "“": '"',
                    "”": '"',
                    "…": "...",
                }
                for viejo, nuevo in reemplazos.items():
                    texto = texto.replace(viejo, nuevo)
                return texto

            tmp_dir = tempfile.gettempdir()
            img_path = os.path.join(tmp_dir, "comparacion_temporadas.png")

            fig.write_image(img_path, format="png", width=1000, height=500)

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            intro_text = """Comparación Sevilla FC – Temporadas 2020-21 vs 2024-25

    En esta sección se analiza la evolución del Sevilla comparando su desempeño actual con el de la temporada 2020-2021, una campaña que se toma como referencia reciente de rendimiento sólido del club.

    El objetivo es observar las diferencias estadísticas significativas que han tenido lugar en estos años, e identificar posibles retrocesos o cambios de enfoque en el juego del equipo.

    Esta comparación longitudinal ayuda a contextualizar si la temporada 2024-2025 representa una anomalía puntual o una tendencia descendente sostenida.
    """

            for line in limpiar_texto(intro_text).strip().split("\n"):
                pdf.multi_cell(0, 8, line)
            pdf.ln(5)

            pdf.image(img_path, x=10, w=190)
            pdf.ln(5)

            if show_comentario_21:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Comentario del analista", ln=True)
                pdf.set_font("Arial", size=12)
                for line in limpiar_texto(comentario_21).strip().split("\n"):
                    pdf.multi_cell(0, 8, line)

            output_path = os.path.join(tmp_dir, f"analisis_temporadas_sevilla_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            pdf.output(output_path)

            with open(output_path, "rb") as f:
                st.download_button("📄 Descargar PDF", data=f, file_name="analisis_temporadas_sevilla.pdf", mime="application/pdf")




with tab3:
    st.subheader("Comparativa de Jugadores por Posición")

    # ========================
    # DATOS PORTEROS
    # ========================
    cols_porteros = ['Player', 'Nation', 'Pos', 'Age', 'MP', 'Starts', 'Min', '90s', 'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PKatt', 'PKA', 'PKsv', 'PKm', 'Save%.1']

    datos_2020_porteros = [
        ['Yassine Bounou', 'ma MAR', 'GK', 29, 33, 33, 2970, 33.0, 28, 0.85, 106, 78, 77.4, 21, 5, 7, 15, 45.5, 6, 4, 2, 0, 33.3],
        ['Tomáš Vaclík', 'cz CZE', 'GK', 31, 5, 5, 450, 5.0, 5, 1.00, 14, 9, 64.3, 3, 0, 2, 2, 40.0, 0, 0, 0, 0, None]
    ]

    datos_2024_porteros = [
        ['Ørjan Nyland', 'no NOR', 'GK', 33, 30, 30, 2673, 29.7, 39, 1.31, 112, 73, 67.9, 8, 10, 12, 7, 23.3, 4, 3, 1, 0, 25.0],
        ['Álvaro Fernández', 'es ESP', 'GK', 26, 9, 8, 747, 8.3, 16, 1.93, 36, 20, 61.1, 2, 1, 5, 3, 37.5, 2, 2, 0, 0, 0.0]
    ]

    df_2020_porteros = pd.DataFrame(datos_2020_porteros, columns=cols_porteros)
    df_2024_porteros = pd.DataFrame(datos_2024_porteros, columns=cols_porteros)

    st.markdown("### Comparar Porteros")
    col1, col2 = st.columns(2)
    with col1:
        portero_2020 = st.selectbox("Portero 2020-21", df_2020_porteros['Player'].tolist(), key="por2020")
    with col2:
        portero_2024 = st.selectbox("Portero 2024-25", df_2024_porteros['Player'].tolist(), key="por2024")

    with col1:
        if st.button("Comparar Porteros"):
            row1 = df_2020_porteros[df_2020_porteros['Player'] == portero_2020].iloc[0]
            row2 = df_2024_porteros[df_2024_porteros['Player'] == portero_2024].iloc[0]

            datos = {
                'Estadística': cols_porteros[4:], 
                portero_2020: [row1[col] for col in cols_porteros[4:]],
                portero_2024: [row2[col] for col in cols_porteros[4:]]
            }
            df_comp = pd.DataFrame(datos)

            # ========== AÑADIR NOTA FINAL ==========
            def calcular_nota_portero(fila):
                try:
                    ga90 = fila['GA90']
                    save_pct = fila['Save%']
                    cs_pct = fila['CS%']
                    pksv = fila['PKsv']

                    if None in (ga90, save_pct, cs_pct, pksv):
                        return None

                    # Escalo Save%, CS%, PKsv
                    save_score = save_pct / 100
                    cs_score = cs_pct / 100
                    pk_score = pksv / 5 if pksv <= 5 else 1  # normalizar sobre 5

                    # Inverso para GA90 (porque menos es mejor)
                    ga_score = max(0, 1 - ga90 / 2)  # si GA90 = 0 -> 1; si GA90 = 2 -> 0

                    # Pondero
                    nota = (ga_score * 0.30 + save_score * 0.30 + cs_score * 0.25 + pk_score * 0.15) * 10
                    return round(nota, 1)
                except:
                    return None

            nota_2020 = calcular_nota_portero(row1)
            nota_2024 = calcular_nota_portero(row2)

            nueva_fila = pd.DataFrame({
                'Estadística': ['Nota final'],
                portero_2020: [nota_2020],
                portero_2024: [nota_2024]
            })

            df_comp = pd.concat([nueva_fila, df_comp], ignore_index=True)

            st.session_state.comparacion_porteros = df_comp


    # ========================
    # DATOS DEFENSAS
    # ========================
    cols_defensas = [
        'Player', 'Nation', 'Pos', 'Age', '90s',
        'Tkl_total', 'Tkl_won', 'Def_3rd', 'Mid_3rd', 'Att_3rd',
        'Challenges_Tkl', 'Challenges_Att', 'Tkl_pct', 'Challenges_Lost',
        'Blocks_total', 'Blocks_Sh', 'Blocks_Pass',
        'Interceptions', 'Tkl+Int', 'Clearances', 'Errors'
    ]

    datos_2020_defensas = [
        ['Jesús Navas', 'es ESP', 'DF', 34, 31.4, 27, 21, 15, 11, 1, 15, 22, 68.2, 7, 10, 3, 7, 37, 64, 62, 1],
        ['Jules Koundé', 'fr FRA', 'DF', 21, 33.1, 27, 15, 14, 11, 2, 10, 24, 41.7, 14, 24, 16, 8, 40, 67, 104, 0],
        ['Diego Carlos', 'br BRA', 'DF', 27, 31.0, 22, 13, 17, 4, 1, 8, 16, 50.0, 8, 35, 27, 8, 31, 53, 114, 1],
        ['Marcos Acuña', 'ar ARG', 'DF', 28, 25.9, 62, 42, 34, 20, 8, 23, 41, 56.1, 18, 23, 2, 21, 23, 85, 30, 1],
        ['Sergi Gómez', 'es ESP', 'DF', 28, 8.0, 4, 3, 2, 2, 0, 1, 1, 100.0, 0, 5, 4, 1, 6, 10, 34, 0],
        ['Sergio Escudero', 'es ESP', 'DF', 30, 6.3, 12, 8, 6, 5, 1, 6, 12, 50.0, 6, 5, 2, 3, 9, 21, 7, 0],
        ['Karim Rekik', 'nl NED', 'DF', 25, 7.3, 7, 5, 2, 5, 0, 1, 4, 25.0, 3, 8, 3, 5, 6, 13, 17, 0],
        ['Aleix Vidal', 'es ESP', 'DF', 30, 6.6, 9, 5, 5, 3, 1, 5, 12, 41.7, 7, 10, 1, 9, 12, 21, 9, 0],
    ]

    datos_2024_defensas = [
        ['Carmona', 'es ESP', 'DF', 22, 32.7, 83, 59, 53, 21, 9, 41, 77, 53.2, 36, 37, 11, 26, 68, 151, 92, 3],
        ['Loïc Bade', 'fr FRA', 'DF', 24, 29.8, 47, 27, 37, 10, 0, 23, 31, 74.2, 8, 22, 13, 9, 29, 76, 162, 2],
        ['Adrià Pedrosa', 'es ESP', 'DF', 26, 25.5, 56, 32, 34, 16, 6, 31, 54, 57.4, 23, 43, 11, 32, 24, 80, 85, 0],
        ['Kike Salas', 'es ESP', 'DF', 22, 24.9, 47, 28, 24, 16, 7, 27, 46, 58.7, 19, 26, 14, 12, 32, 79, 146, 0],
        ['Juanlu Sánchez', 'es ESP', 'DF', 20, 19.1, 50, 34, 28, 15, 7, 35, 59, 59.3, 24, 27, 2, 25, 17, 67, 19, 0],
        ['Marcão', 'br BRA', 'DF', 28, 6.1, 10, 7, 6, 4, 0, 3, 5, 60.0, 2, 9, 6, 3, 7, 17, 29, 1],
        ['Tanguy Nianzou', 'fr FRA', 'DF', 22, 5.8, 7, 5, 6, 0, 1, 5, 6, 83.3, 1, 1, 0, 1, 8, 15, 27, 0],
        ['Jesús Navas', 'es ESP', 'DF', 38, 4.7, 5, 2, 3, 0, 2, 0, 3, 0.0, 3, 1, 0, 1, 4, 9, 6, 0],
        ['Ramón Martínez', 'es ESP', 'DF', 21, 2.9, 6, 3, 5, 1, 0, 3, 5, 60.0, 2, 2, 1, 1, 2, 8, 23, 0],
    ]

    df_2020_defensas = pd.DataFrame(datos_2020_defensas, columns=cols_defensas)
    df_2024_defensas = pd.DataFrame(datos_2024_defensas, columns=cols_defensas)

    st.markdown("### Comparar Defensas")
    col3, col4 = st.columns(2)
    with col3:
        defensa_2020 = st.selectbox("Defensa 2020-21", df_2020_defensas['Player'].tolist(), key="def2020")
    with col4:
        defensa_2024 = st.selectbox("Defensa 2024-25", df_2024_defensas['Player'].tolist(), key="def2024")

    if st.button("Comparar Defensas"):
        r1 = df_2020_defensas[df_2020_defensas['Player'] == defensa_2020].iloc[0]
        r2 = df_2024_defensas[df_2024_defensas['Player'] == defensa_2024].iloc[0]

        datos_def = {
            'Estadística': cols_defensas[4:],
            defensa_2020: [r1[c] for c in cols_defensas[4:]],
            defensa_2024: [r2[c] for c in cols_defensas[4:]],
        }
        df_def = pd.DataFrame(datos_def)

        def calcular_nota_defensa(fila):
            try:
                tkl_total = fila['Tkl_total']
                tkl_won = fila['Tkl_won']
                tkl_pct = fila['Tkl_pct'] / 100 if fila['Tkl_pct'] is not None else 0
                interceptions = fila['Interceptions']
                clearances = fila['Clearances']
                errors = fila['Errors']
                mins_jugados = fila['90s'] * 90

                if mins_jugados == 0:
                    return None

                # Penalizar notas con poco tiempo jugado

                tkl_ratio = (tkl_won / tkl_total) if tkl_total else 0
                interceptions_90 = (interceptions / mins_jugados) * 90
                clearances_90 = (clearances / mins_jugados) * 90
                error_score = max(0, 1 - errors * 0.1)

                interceptions_score = min(interceptions_90 / 5, 1)   # 5 intercepciones por 90' = puntuación máxima
                clearances_score = min(clearances_90 / 8, 1)         # 8 despejes por 90' = puntuación máxima


                nota = (
                    tkl_ratio * 0.25 +
                    tkl_pct * 0.25 +
                    interceptions_score * 0.25 +
                    clearances_score * 0.2 +
                    error_score * 0.05
                )* 10 


                return round(min(nota, 10), 1)
            except:
                return None

        nota_2020 = calcular_nota_defensa(r1)
        nota_2024 = calcular_nota_defensa(r2)

        nueva_fila = pd.DataFrame({
            'Estadística': ['Nota final'],
            defensa_2020: [nota_2020],
            defensa_2024: [nota_2024]
        })

        df_def = pd.concat([nueva_fila, df_def], ignore_index=True)

        st.session_state.comparacion_defensas = df_def
 

    # ========================
    # DATOS CENTROCAMPISTAS
    # ========================
    cols_centrocampistas = [
        'Player', 'Nation', 'Pos', 'Age', '90s', 'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd',
        'Att', 'Succ', 'Succ%', 'Tkl/Att', 'Carries', 'TotDist', 'PrgDist',
        'ProgCarries', 'Carries3rd', 'CarriesCPA', 'Mis', 'Dis', 'Touches', 'DefPen', 'Mid3rd', 'Att3rd'
    ]

    datos_2020_centrocampistas = [
        ['Fernando', 'br BRA', 'MF', 33, 29.7, 63, 43, 28, 26, 9, 13, 4, 30.8, 1.0, 1246, 14264, 3201, 78, 15, 5, 22, 18, 1814, 56, 676, 1082],
        ['Joan Jordán', 'es ESP', 'MF', 26, 29.6, 50, 33, 27, 14, 9, 40, 18, 45.0, 1.3, 1016, 13606, 4117, 67, 16, 8, 28, 21, 1683, 47, 678, 946],
        ['Ivan Rakitić', 'hr CRO', 'MF', 32, 27.8, 37, 25, 15, 17, 5, 34, 18, 52.9, 1.1, 1010, 11745, 3520, 67, 17, 11, 22, 30, 1569, 36, 578, 932],
        ['Óscar Rodríguez', 'es ESP', 'MF', 22, 9.6, 15, 11, 7, 5, 3, 15, 7, 46.7, 1.0, 381, 4540, 1589, 22, 5, 4, 9, 12, 551, 12, 194, 340],
        ['Oliver Torres', 'es ESP', 'MF', 26, 17.5, 22, 14, 7, 12, 3, 13, 7, 53.8, 1.7, 566, 7541, 1724, 28, 6, 2, 10, 11, 894, 17, 354, 523],
        ['Nemanja Gudelj', 'rs SRB', 'MF', 29, 11.6, 17, 11, 8, 7, 2, 6, 1, 16.7, 0.5, 338, 5096, 1382, 16, 2, 1, 10, 5, 613, 21, 260, 329],
    ]

    datos_2024_centrocampistas = [
        ['Soumaré', 'fr FRA', 'MF', 25, 30.6, 55, 37, 24, 21, 10, 40, 25, 62.5, 1.6, 951, 11100, 3126, 74, 23, 6, 19, 16, 1778, 54, 639, 1022],
        ['Joan Jordán', 'es ESP', 'MF', 29, 17.9, 30, 20, 11, 15, 4, 14, 8, 57.1, 0.9, 501, 6811, 1832, 33, 8, 2, 7, 8, 936, 19, 371, 546],
        ['Óliver Torres', 'es ESP', 'MF', 29, 11.7, 15, 12, 7, 5, 3, 13, 6, 46.2, 1.1, 419, 5346, 1327, 20, 4, 1, 6, 8, 645, 17, 231, 397],
        ['Hannibal Mejbri', 'tn TUN', 'MF', 21, 6.3, 10, 6, 5, 2, 3, 7, 2, 28.6, 1.1, 208, 2597, 636, 13, 2, 1, 2, 5, 271, 6, 94, 168],
        ['Manu Bueno', 'es ESP', 'MF', 20, 5.4, 12, 10, 7, 4, 1, 5, 2, 40.0, 0.9, 166, 2202, 619, 11, 1, 1, 5, 3, 272, 5, 106, 144],
    ]

    df_2020_centrocampistas = pd.DataFrame(datos_2020_centrocampistas, columns=cols_centrocampistas)
    df_2024_centrocampistas = pd.DataFrame(datos_2024_centrocampistas, columns=cols_centrocampistas)

    st.markdown("### Comparar Centrocampistas")
    col5, col6 = st.columns(2)
    with col5:
        centro_2020 = st.selectbox("Centrocampista 2020-21", df_2020_centrocampistas['Player'].tolist(), key="cen2020")
    with col6:
        centro_2024 = st.selectbox("Centrocampista 2024-25", df_2024_centrocampistas['Player'].tolist(), key="cen2024")

    if st.button("Comparar Centrocampistas"):
        r1 = df_2020_centrocampistas[df_2020_centrocampistas['Player'] == centro_2020].iloc[0]
        r2 = df_2024_centrocampistas[df_2024_centrocampistas['Player'] == centro_2024].iloc[0]

        def calcular_nota_centrocampista(fila):
            try:
                # Extraer datos y normalizarlos (dividir por 90 para tasa por partido)
                tkl_per_90 = fila['Tkl'] / fila['90s'] if fila['90s'] > 0 else 0
                tkl_won_per_90 = fila['TklW'] / fila['90s'] if fila['90s'] > 0 else 0
                interceptions_per_90 = fila['Dis'] / fila['90s'] if fila['90s'] > 0 else 0
                mis_per_90 = fila['Mis'] / fila['90s'] if fila['90s'] > 0 else 0
                succ_pct = fila['Succ%'] / 100  # porcentaje a decimal
                tkl_att_ratio = fila['Tkl/Att']  # ya es ratio
                carries_per_90 = fila['Carries'] / fila['90s'] if fila['90s'] > 0 else 0
                prg_dist_per_90 = fila['PrgDist'] / fila['90s'] if fila['90s'] > 0 else 0

                # Normalizar variables con límites máximos razonables para escala 0-1
                tkl_score = min(tkl_per_90 / 5, 1)          # 5 entradas por partido = max
                tkl_won_score = min(tkl_won_per_90 / 3, 1) # 3 entradas ganadas = max
                interceptions_score = min(interceptions_per_90 / 5, 1) # 5 intercepciones = max
                mis_score = max(0, 1 - mis_per_90 / 2)     # menos pérdidas mejor
                succ_score = succ_pct                       # ya en 0-1, porcentaje éxito
                tkl_att_score = min(tkl_att_ratio / 4, 1)  # ratio entradas/intentamos, max 4
                carries_score = min(carries_per_90 / 30, 1) # 30 carries por partido max
                prg_dist_score = min(prg_dist_per_90 / 120, 1) # 120 metros de progresión max

                # Ponderaciones (ajustables según importancia)
                nota = (
                    tkl_score * 0.2 +
                    tkl_won_score * 0.10 +
                    interceptions_score * 0.15 +
                    mis_score * 0.10 +
                    succ_score * 0.15 +
                    tkl_att_score * 0.10 +
                    carries_score * 0.10 +
                    prg_dist_score * 0.10
                )

                # Penalizador por minutos (menos minutos = menos nota)
                penalizador_minutos = min(fila['90s'] / 20, 1)  # menos de 20 partidos baja nota

                nota_final = round(nota * 10, 1)
                return nota_final

            except:
                return None

        nota_2020 = calcular_nota_centrocampista(r1)
        nota_2024 = calcular_nota_centrocampista(r2)

        datos_cent = {
            'Estadística': ['Nota final'] + cols_centrocampistas[5:],
            centro_2020: [nota_2020] + [r1[c] for c in cols_centrocampistas[5:]],
            centro_2024: [nota_2024] +[r2[c] for c in cols_centrocampistas[5:]],
        }
        st.session_state.comparacion_centrocampistas = pd.DataFrame(datos_cent)


    # ========================
    # DATOS DELANTEROS
    # ========================
    cols_delanteros = [
        'Player', 'Nation', 'Pos', 'Age', '90s', 
        'SCA', 'SCA90', 'PassLive', 'PassDead', 'TO', 'Sh', 'Fld', 'Def',
        'GCA', 'GCA90', 'GCA_PassLive', 'GCA_PassDead', 'GCA_TO', 'GCA_Sh', 'GCA_Fld', 'GCA_Def'
    ]

    datos_2020_delanteros = [
        ['Lucas Ocampos', 'ar ARG', 'FW', 26, 29.2, 81, 2.77, 50, 2, 10, 6, 11, 2, 7, 0.24, 5, 0, 0, 1, 1, 0],
        ['Suso', 'es ESP', 'FW', 26, 24.6, 108, 4.40, 82, 6, 14, 3, 3, 0, 14, 0.57, 12, 1, 0, 1, 0, 0],
        ['Youssef En-Nesyri', 'ma MAR', 'FW', 23, 25.7, 36, 1.40, 19, 0, 4, 6, 4, 3, 8, 0.31, 1, 0, 4, 1, 2, 0],
        ['Luuk de Jong', 'nl NED', 'FW', 29, 14.1, 25, 1.77, 16, 0, 1, 4, 4, 0, 0, 0.00, 0, 0, 0, 0, 0, 0],
        ['Munir El Haddadi', 'ma MAR', 'FW', 24, 9.4, 25, 2.66, 16, 3, 1, 2, 3, 0, 3, 0.32, 3, 0, 0, 0, 0, 0],
        ['Carlos Fernández', 'es ESP', 'FW', 24, 1.6, 4, 2.52, 2, 0, 0, 2, 0, 0, 0, 0.00, 0, 0, 0, 0, 0, 0],
    ]

    datos_2024_delanteros = [
        ['Dodi Lukebakio', 'be BEL', 'FW', 26, 34.4, 119, 3.46, 65, 18, 21, 5, 8, 2, 9, 0.26, 6, 0, 2, 0, 1, 0],
        ['Isaac Romero', 'es ESP', 'FW', 24, 24.0, 44, 1.84, 30, 1, 5, 3, 5, 0, 7, 0.29, 4, 0, 1, 2, 0, 0],
        ['Chidera Ejuke', 'ng NGA', 'FW', 26, 10.7, 45, 4.19, 28, 5, 9, 0, 3, 0, 2, 0.19, 2, 0, 0, 0, 0, 0],
        ['Suso', 'es ESP', 'FW,MF', 30, 6.7, 45, 6.68, 25, 8, 8, 3, 1, 0, 3, 0.45, 2, 1, 0, 0, 0, 0],
    ]

    df_2020_delanteros = pd.DataFrame(datos_2020_delanteros, columns=cols_delanteros)
    df_2024_delanteros = pd.DataFrame(datos_2024_delanteros, columns=cols_delanteros)

    st.markdown("### Comparar Delanteros")
    col7, col8 = st.columns(2)
    with col7:
        delantero_2020 = st.selectbox("Delantero 2020-21", df_2020_delanteros['Player'].tolist(), key="del2020")
    with col8:
        delantero_2024 = st.selectbox("Delantero 2024-25", df_2024_delanteros['Player'].tolist(), key="del2024")

    if st.button("Comparar Delanteros"):
        d1 = df_2020_delanteros[df_2020_delanteros['Player'] == delantero_2020].iloc[0]
        d2 = df_2024_delanteros[df_2024_delanteros['Player'] == delantero_2024].iloc[0]

        def calcular_nota_delantero(row):
            # Variables clave
            sca90 = row['SCA90']           # oportunidades creadas por 90'
            gca90 = row['GCA90']           # asistencias de gol por 90'
            sh = row['Sh']                 # disparos
            pass_live = row['PassLive']    # pases en juego
            fld = row['Fld']               # faltas recibidas
            def_ = row['Def']              # acciones defensivas, poco peso para delantero
            
            # Normalizamos y ponderamos (ajusta según lo que consideres importante)
            nota = (
                min(sca90 / 5, 1) * 0.25 +    # max 5 SCA90
                min(gca90 / 2, 1) * 0.25 +    # max 2 GCA90
                min(sh / 4, 1) * 0.20 +       # max 4 disparos por 90
                min(pass_live / 80, 1) * 0.15 +  # max 80 pases en juego
                min(fld / 5, 1) * 0.10        # max 5 faltas recibidas
                # puedes añadir defensa si quieres, pero poca importancia para delantero
            )
            return round(nota * 10, 2)  # Escalamos a nota sobre 10

        nota_2020 = calcular_nota_delantero(d1)
        nota_2024 = calcular_nota_delantero(d2)

        datos_del = {
            'Estadística': ['Nota final'] + cols_delanteros[5:],
            delantero_2020: [nota_2020] + [d1[c] for c in cols_delanteros[5:]],
            delantero_2024: [nota_2024] + [d2[c] for c in cols_delanteros[5:]],
        }

        st.session_state.comparacion_delanteros = pd.DataFrame(datos_del)


    # ========================
    # MOSTRAR TODAS LAS TABLAS
    # ========================

    # Comparación de porteros
    if "comparacion_porteros" in st.session_state:
        st.write("🔍 Comparación: Porteros")
        st.dataframe(st.session_state["comparacion_porteros"])
        if st.button("❌ Borrar comparación de porteros"):
            del st.session_state["comparacion_porteros"]
            st.rerun()

    # Comparación de defensas
    if "comparacion_defensas" in st.session_state:
        st.write("🛡️ Comparación: Defensas")
        st.dataframe(st.session_state["comparacion_defensas"])
        if st.button("❌ Borrar comparación de defensas"):
            del st.session_state["comparacion_defensas"]
            st.rerun()

    # Comparación de centrocampistas
    if "comparacion_centrocampistas" in st.session_state:
        st.write("🎯 Comparación: Centrocampistas")
        st.dataframe(st.session_state["comparacion_centrocampistas"])
        if st.button("❌ Borrar comparación de centrocampistas"):
            del st.session_state["comparacion_centrocampistas"]
            st.rerun()

    # Comparación de delanteros
    if "comparacion_delanteros" in st.session_state:
        st.write("⚽ Comparación: Delanteros")
        st.dataframe(st.session_state["comparacion_delanteros"])
        if st.button("❌ Borrar comparación de delanteros"):
            del st.session_state["comparacion_delanteros"]
            st.rerun()


    # ========================
    # EXPORTAR A PDF
    # ========================
    from fpdf import FPDF
    import tempfile
    import base64

    if st.button("📄 Exportar Comparaciones a PDF"):

        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Comparativa de Jugadores', ln=True, align='C')
                self.ln(5)

            def tabla(self, titulo, dataframe):
                self.set_font("Arial", 'B', 12)
                self.cell(0, 10, titulo, ln=True)
                self.set_font("Arial", '', 9)

                col_widths = [40] + [75] * (len(dataframe.columns) - 1)
                headers = dataframe.columns.tolist()
                for i, h in enumerate(headers):
                    self.cell(col_widths[i], 8, str(h), border=1)
                self.ln()

                for index, row in dataframe.iterrows():
                    for i, item in enumerate(row):
                        self.cell(col_widths[i], 8, str(item), border=1)
                    self.ln()
                self.ln(5) 

        pdf = PDF()
        pdf.add_page()

        if "comparacion_porteros" in st.session_state:
            pdf.tabla("Comparativa: Porteros", st.session_state["comparacion_porteros"])
        if "comparacion_defensas" in st.session_state:
            pdf.tabla("Comparativa: Defensas", st.session_state["comparacion_defensas"])
        if "comparacion_centrocampistas" in st.session_state:
            pdf.tabla("Comparativa: Centrocampistas", st.session_state["comparacion_centrocampistas"])
        if "comparacion_delanteros" in st.session_state:
            pdf.tabla("Comparativa: Delanteros", st.session_state["comparacion_delanteros"])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            pdf.output(tmpfile.name)
            tmpfile.seek(0)
            b64 = base64.b64encode(tmpfile.read()).decode()

        href = f'<a href="data:application/pdf;base64,{b64}" download="comparativas_jugadores.pdf">📥 Descargar PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
