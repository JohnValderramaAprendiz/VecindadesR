# --- Importaciones Esenciales ---
import pandas as pd
from shiny import App, ui, reactive, render
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import pathlib
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import io
import faicons as fa

# --- Carga de Datos ---
def cargar_datos(file_name: str):
    """Carga datos desde un archivo parquet en la carpeta 'data'."""
    try:
        app_dir = pathlib.Path(__file__).parent.resolve()
        file_path = app_dir / "data" / file_name
        df = pd.read_parquet(file_path)
        print(f"Datos cargados desde '{file_name}': {df.shape}")
        return df
    except Exception as e:
        print(f"Error al cargar '{file_name}': {e}")
        return pd.DataFrame()

DATOS_PROGRAMAS = cargar_datos("datos_preprocesados.parquet")
DATOS_DESCRIPTIVOS = cargar_datos("datos_snies_consolidados_con_descriptivos.parquet")

# --- Variables Globales y de Configuración ---
ANIOS_DISPONIBLES = sorted(DATOS_PROGRAMAS['ANNO'].unique(), reverse=True) if not DATOS_PROGRAMAS.empty else []
COLUMNAS_FEATURES = sorted([
    'PRO_GEN_ESTU_EDAD', 'PRO_GEN_FAMI_ESTRATOVIVIENDA', 'PRO_GEN_ESTU_GENERO', 'PRO_GEN_ESTU_HORASSEMANATRABAJA'
])

NOMBRES_FEATURES = {
    'PRO_GEN_ESTU_EDAD': 'EDAD',
    'PRO_GEN_FAMI_ESTRATOVIVIENDA': 'ESTRATO',
    'PRO_GEN_ESTU_GENERO': 'GENERO',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA': 'HORAS DE TRABAJO X SEMANA'
}

NOMBRES_AMIGABLES_MODELAR = {
    'CODIGO_SNIES_DEL_PROGRAMA': 'SNIES', 'PRO_GEN_ESTU_EDAD_RANGO_15-24': 'Edad [15-24]',
    'PRO_GEN_ESTU_EDAD_RANGO_25-34': 'Edad [25-34]', 'PRO_GEN_ESTU_EDAD_RANGO_35-44': 'Edad [35-44]',
    'PRO_GEN_ESTU_EDAD_RANGO_45-54': 'Edad [45-54]', 'PRO_GEN_ESTU_EDAD_RANGO_55 o mas': 'Edad [55-100]',
    'PRO_GEN_FAMI_ESTRATOVIVIENDA_-1': 'Sin Estrato', 'PRO_GEN_FAMI_ESTRATOVIVIENDA_0': 'Estrato 0',
    'PRO_GEN_FAMI_ESTRATOVIVIENDA_1': 'Estrato 1', 'PRO_GEN_FAMI_ESTRATOVIVIENDA_2': 'Estrato 2',
    'PRO_GEN_FAMI_ESTRATOVIVIENDA_3': 'Estrato 3', 'PRO_GEN_FAMI_ESTRATOVIVIENDA_4': 'Estrato 4',
    'PRO_GEN_FAMI_ESTRATOVIVIENDA_5': 'Estrato 5', 'PRO_GEN_FAMI_ESTRATOVIVIENDA_6': 'Estrato 6',
    'PRO_GEN_ESTU_GENERO_FEMENINO': 'Femenino', 'PRO_GEN_ESTU_GENERO_MASCULINO': 'Masculino',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_': 'No reporta horas',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_Entre 11 y 20 horas': '[11-20] horas',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_Entre 21 y 30 horas': '[21-30] horas',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_Menos de 10 horas': '[1-10] horas',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_Más de 30 horas': 'Más de 30 horas',
    'PRO_GEN_ESTU_HORASSEMANATRABAJA_No trabaja': 'No trabaja'
}

NOMBRES_AMIGABLES_TABLAS = {
    'TOTAL_MATRICULADOS_ANNO': 'Matriculados (Año)',
    'TOTAL_PRIMER_CURSO_ANNO': 'Inscritos Primer Curso (Año)',
    'MEAN_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT': 'Media Razonamiento Cuantitativo',
    'STD_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT': 'Desv. Razonamiento Cuantitativo',
    'MEAN_PRO_GEN_MOD_LECTURA_CRITICA_PUNT': 'Media Lectura Crítica',
    'STD_PRO_GEN_MOD_LECTURA_CRITICA_PUNT': 'Desv. Lectura Crítica',
    'MEAN_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT': 'Media Competencia Ciudadana',
    'STD_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT': 'Desv. Competencia Ciudadana',
    'MEAN_PRO_GEN_MOD_INGLES_PUNT': 'Media Inglés',
    'STD_PRO_GEN_MOD_INGLES_PUNT': 'Desv. Inglés',
    'MEAN_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT': 'Media Comunicación Escrita',
    'STD_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT': 'Desv. Comunicación Escrita',
    'MEAN_PRO_GEN_PUNT_GLOBAL': 'Media Puntaje Global',
    'STD_PRO_GEN_PUNT_GLOBAL': 'Desv. Puntaje Global',
    'ANNO': 'Año',
    'CODIGO_SNIES_DEL_PROGRAMA': 'SNIES',
    'PROGRAMA_ACADEMICO': 'Programa Académico',
    'NOMBRE_INSTITUCION': 'Institución',
    'NIVEL_ACADEMICO': 'Nivel Académico',
    'NIVEL_DE_FORMACION': 'Nivel de Formación',
    'MODALIDAD': 'Modalidad',
    'NUMERO_CREDITOS': 'Créditos',
    'COSTO_MATRICULA_ESTUD_NUEVOS': 'Costo Matrícula',
    'TOTAL_GRADUADOS_ANNO': 'Graduados (Año)',
    'TOTAL_MATRICULADOS_ANNO': 'Matriculados (Año)',
    'TOTAL_PRIMER_CURSO_ANNO': 'Inscritos Primer Curso (Año)',
    'CODIGO_INSTITUCION': 'Código Institución',
    'CARACTER_ACADEMICO': 'Carácter Académico',
    'SECTOR': 'Sector',
    'ESTADO_PROGRAMA': 'Estado Programa',
    'FECHA_DE_RESOLUCION': 'Fecha Resolución',
    'FECHA_DE_REGISTRO_EN_SNIES': 'Fecha Registro SNIES',
    'RECONOCIMIENTO_DEL_MINISTERIO': 'Reconocimiento Ministerio',
    'CINE_F_2013_AC_CAMPO_ESPECIFIC': 'Campo Específico (CINE F 2013)',
    'DEPARTAMENTO_PRINCIPAL': 'Departamento Principal'
}

# --- Interfaz de Usuario (UI) ---
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.title("Cálculo de Distancias entre Programas"),
        ui.include_css(pathlib.Path(__file__).parent / "styles.css"),
        ui.tags.style("""
            .container-fluid {
                min-height: 95vh;
            }
            /* Aumentar ancho de columnas específicas por su ID de columna (sin espacios) */
            .ag-header-cell[col-id="ProgramadeReferencia"],
            .ag-cell[col-id="ProgramadeReferencia"],
            .ag-header-cell[col-id="ProgramaAcadémico"],
            .ag-cell[col-id="ProgramaAcadémico"],
            .ag-header-cell[col-id="Institución"],
            .ag-cell[col-id="Institución"] {
                min-width: 400px !important;
            }
            .banner {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 20px;
                background-color: #ffffff;
                border-bottom: 2px solid #f0f0f0;
                margin-bottom: 1rem;
            }
            .banner-left {
                background-color: #044d35;
                padding: 5px 10px;
                border-radius: 5px;
            }
            .banner-center {
                flex-grow: 1;
                text-align: center;
                color: #162644;
            }
            .banner-right {
                font-weight: bold;
            }
        """)
    ),
    # --- Banner Superior ---
    ui.div(
        {"class": "banner"},
        ui.div(
            {"class": "banner-left"},
            ui.tags.a(
                ui.tags.img(src="https://www.uniminuto.edu/sites/default/files/logo-para-web-2024.png", height="60px"),
                href="https://www.uniminuto.edu/",
                target="_blank"
            )
        ),
        ui.div({"class": "banner-center"}, ui.h2("Análisis de Vecindades Socio-Económicas")),
        ui.div(
            {"class": "banner-right"},
            ui.tags.a(
                ui.tags.img(src="https://victorious-pond-055c69910.1.azurestaticapps.net/assets/images/logo.png", height="80px"),
                href="https://victorious-pond-055c69910.1.azurestaticapps.net/",
                target="_blank"
            )
        )
    ),
    # --- Layout Principal (Sidebar + Contenido) ---
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Controles del Análisis"),
            ui.input_select("anio_seleccionado", "Año de Análisis:", {str(a): str(a) for a in ANIOS_DISPONIBLES}, selected=str(ANIOS_DISPONIBLES[0]) if ANIOS_DISPONIBLES else None),
            ui.input_checkbox_group("variables_seleccionadas", "Variables para Cálculo de Distancia:", {k: NOMBRES_FEATURES.get(k, k) for k in COLUMNAS_FEATURES}),
            ui.input_select("metrica_distancia", "Métrica de Distancia:", {"euclidean": "Euclidiana", "manhattan": "Manhattan", "cosine": "Coseno"}, selected="euclidean"),
            ui.input_selectize("programa_referencia", "Programa de Referencia (digite código o nombre):", {"ninguno": "(Ninguno seleccionado)"}),
            ui.input_checkbox_group(
                "filtros_contexto",
                "Filtro de Contexto (opcional):",
                {
                    "SECTOR": "Sector",
                    "RECONOCIMIENTO_DEL_MINISTERIO": "Reconocimiento del Ministerio",
                    "CINE_F_2013_AC_CAMPO_ESPECIFIC": "Campo Específico (CINE F 2013)",
                    "MODALIDAD": "Modalidad",
                    "DEPARTAMENTO_PRINCIPAL": "Departamento Principal"
                }
            ),
            ui.input_slider("n_vecinos", "Número de vecinos a resaltar:", 0, 25, 15),
            ui.input_action_button("ejecutar_analisis", "Calcular Distancias", icon=fa.icon_svg("play"), class_="btn-primary w-100 mt-3"),
            ui.hr(),
            ui.download_button("descargar_excel", "Descargar Excel", icon=fa.icon_svg("download"), class_="btn-success w-100 mt-2"),
            ui.tags.a(
                "Metodología",
                href="https://victorious-pond-055c69910.1.azurestaticapps.net/metodologia.html",
                target="_blank",
                class_="btn btn-info w-100 mt-2"
            ),
            width=525
        ),
        ui.div(
            ui.navset_tab(
                ui.nav_panel(
                    "Análisis de Distancias",
                    ui.h4("Tabla de Distancias"),
                    ui.p("La tabla muestra las distancias desde el programa de referencia a los demás programas, basado en las variables seleccionadas."),
                    ui.output_data_frame("tabla_distancias_output")
                ),
                ui.nav_panel(
                    "Comparación de Características",
                    ui.h4("Comparación de Características"),
                    ui.p("La tabla muestra las características de los programas vecinos en comparación con el programa de referencia."),
                    ui.output_data_frame("tabla_comparacion_caracteristicas_output")
                ),
                ui.nav_panel(
                    "Visualización PCA",
                    ui.card(ui.card_header("PCA 2D"), ui.output_ui("pca_2d_plot"), full_screen=True),
                    ui.card(ui.card_header("PCA 3D"), ui.output_ui("pca_3d_plot"), full_screen=True)
                ),
                ui.nav_panel(
                    "Visualización t-SNE",
                    ui.card(ui.card_header("t-SNE 2D"), ui.output_ui("tsne_2d_plot"), full_screen=True),
                    ui.card(ui.card_header("t-SNE 3D"), ui.output_ui("tsne_3d_plot"), full_screen=True)
                ),
                # Pestañas de trabajo ocultas para producción
                # ui.nav_panel(
                #     "Datos Generales",
                #     ui.h4("Datos Consolidados de Programas"),
                #     ui.p("Vista completa de los datos descriptivos de los programas cargados en la aplicación."),
                #     ui.output_data_frame("tabla_datos_generales")
                # ),
                # ui.nav_panel(
                #     "Datos para Modelar",
                #     ui.h4("Datos Seleccionados para el Cálculo"),
                #     ui.p("Esta tabla muestra los datos exactos (variables seleccionadas para el año elegido) que se usarán para calcular las distancias."),
                #     ui.output_data_frame("tabla_datos_para_modelar")
                # ),
                ui.nav_panel(
                    "Comparación en la Vecindad",
                    ui.card(
                        ui.card_header("Comparación de Indicadores en la Vecindad"),
                        ui.output_ui("grafica_vecindad_matriculados"),
                        full_screen=True
                    )
                ),
                ui.nav_panel(
                    "Saber PRO en la Vecindad",
                    ui.p("Gráficas de puntajes de las pruebas Saber PRO para los programas en la vecindad, mostrando medias y desviaciones estándar."),
                    ui.output_ui("grafica_saber_pro_vecindad")
                )
            ),
        )
    )
)

# --- Lógica del Servidor ---
def server(input, output, session):

    resultados_distancias = reactive.Value(pd.DataFrame())
    pca_resultados = reactive.Value(pd.DataFrame())
    tsne_resultados = reactive.Value(pd.DataFrame())

    @reactive.Calc
    def datos_filtrados_por_anio():
        anio = int(input.anio_seleccionado())
        return DATOS_PROGRAMAS[DATOS_PROGRAMAS['ANNO'] == anio].copy()

    @reactive.Effect
    def _actualizar_opciones_programa():
        df_programas = datos_filtrados_por_anio()
        df_desc = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
        df_opciones = pd.merge(
            df_programas[['CODIGO_SNIES_DEL_PROGRAMA']].drop_duplicates(),
            df_desc[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO', 'NOMBRE_INSTITUCION']],
            on='CODIGO_SNIES_DEL_PROGRAMA', how='left'
        ).sort_values(by=['NOMBRE_INSTITUCION', 'PROGRAMA_ACADEMICO'])
        opciones = {
            str(row['CODIGO_SNIES_DEL_PROGRAMA']): f"{row['CODIGO_SNIES_DEL_PROGRAMA']} - {row['PROGRAMA_ACADEMICO']} ({row['NOMBRE_INSTITUCION']})"
            for _, row in df_opciones.iterrows()
        }
        choices = {"ninguno": "(Ninguno seleccionado)"}
        choices.update(opciones)
        ui.update_selectize("programa_referencia", choices=choices, selected="ninguno")

    @reactive.Calc
    def features_df_reactive():
        variables = input.variables_seleccionadas()
        if not variables:
            return pd.DataFrame({"Mensaje": ["Seleccione variables para empezar."]})
        df_modelar = datos_filtrados_por_anio()
        columnas_features_a_usar = []
        variables_categoricas_base = ['PRO_GEN_ESTU_GENERO', 'PRO_GEN_FAMI_ESTRATOVIVIENDA', 'PRO_GEN_ESTU_HORASSEMANATRABAJA', 'PRO_GEN_ESTU_EDAD']
        df_modelar_cols = df_modelar.columns
        for var_base in variables:
            if var_base in variables_categoricas_base:
                for col in df_modelar_cols:
                    if col.startswith(var_base):
                        columnas_features_a_usar.append(col)
            else:
                columnas_features_a_usar.append(f"MEAN_{var_base}")
                columnas_features_a_usar.append(f"STD_{var_base}")
        columnas_features_a_usar = list(dict.fromkeys(columnas_features_a_usar))
        columnas_existentes = [col for col in columnas_features_a_usar if col in df_modelar.columns]
        if not columnas_existentes:
            return pd.DataFrame({"Error": ["No se encontraron columnas de datos para las variables seleccionadas."]})
        features_df = df_modelar[['CODIGO_SNIES_DEL_PROGRAMA'] + columnas_existentes].set_index('CODIGO_SNIES_DEL_PROGRAMA')
        return features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    @reactive.Effect
    @reactive.event(input.ejecutar_analisis)
    def _ejecutar_calculo_distancias():
        features_df = features_df_reactive()
        if features_df.empty or 'Mensaje' in features_df.columns or 'Error' in features_df.columns:
            ui.notification_show("No hay datos para calcular. Verifique las selecciones.", type="warning")
            resultados_distancias.set(features_df)
            return

        programa_ref_snies = input.programa_referencia()
        if not programa_ref_snies or programa_ref_snies == "ninguno":
            ui.notification_show("Por favor, seleccione un programa de referencia.", type="warning")
            resultados_distancias.set(pd.DataFrame({"Mensaje": ["Seleccione un programa de referencia."]}))
            return

        # Filtrar por contexto si el usuario seleccionó filtros
        filtros_contexto = input.filtros_contexto()
        if filtros_contexto:
            df_desc = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
            ref_row = df_desc[df_desc['CODIGO_SNIES_DEL_PROGRAMA'] == int(programa_ref_snies)]
            if not ref_row.empty:
                for filtro in filtros_contexto:
                    valor = ref_row.iloc[0][filtro]
                    # Filtrar features_df por los SNIES que cumplen el contexto
                    snies_validos = df_desc[df_desc[filtro] == valor]['CODIGO_SNIES_DEL_PROGRAMA'].unique()
                    features_df = features_df[features_df.index.isin(snies_validos)]
            else:
                ui.notification_show("No se encontró el programa de referencia en los datos descriptivos.", type="error")
                resultados_distancias.set(pd.DataFrame({"Mensaje": ["No se encontró el programa de referencia en los datos descriptivos."]}))
                return

        with ui.Progress(min=0, max=4) as p:
            p.set(value=1, message="Preparando datos...")
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df)
            p.set(value=2, message="Calculando matriz de distancias...")
            dist_matrix = pairwise_distances(features_scaled, metric=input.metrica_distancia())
            dist_df = pd.DataFrame(dist_matrix, index=features_df.index, columns=features_df.index)
            p.set(value=3, message="Calculando PCA y t-SNE...")
            n_features = features_scaled.shape[1]
            n_components_pca = min(3, n_features)
            if n_components_pca > 0:
                pca = PCA(n_components=n_components_pca)
                pca_result = pca.fit_transform(features_scaled)
                pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components_pca)], index=features_df.index).reset_index()
                pca_resultados.set(pca_df)
            else:
                pca_resultados.set(pd.DataFrame())

            n_components_tsne = min(3, n_features)
            if n_components_tsne > 1:
                tsne_params = {
                    "n_components": n_components_tsne, "random_state": 42,
                    "perplexity": min(30, len(features_scaled) - 1)
                }
                # Siempre cálculo rápido t-SNE
                tsne_params["max_iter"] = 300
                tsne_params["init"] = "pca"
                tsne = TSNE(**tsne_params)
                tsne_result = tsne.fit_transform(features_scaled)
                tsne_df = pd.DataFrame(tsne_result, columns=[f'TSNE{i+1}' for i in range(n_components_tsne)], index=features_df.index).reset_index()
                tsne_resultados.set(tsne_df)
            else:
                tsne_resultados.set(pd.DataFrame())

            programa_ref_snies_int = int(programa_ref_snies)
            if programa_ref_snies_int not in dist_df.index:
                ui.notification_show("El programa de referencia no tiene datos para las variables seleccionadas.", type="error")
                return

            distancias_serie = dist_df[programa_ref_snies_int].sort_values()
            dist_filtrada = distancias_serie.reset_index()
            dist_filtrada.columns = ['SNIES_Vecino', 'Distancia']
            dist_filtrada['SNIES_Referencia'] = programa_ref_snies_int

            df_desc_filtrado = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())].drop_duplicates(subset=['CODIGO_SNIES_DEL_PROGRAMA'])

            detalles_vecino = df_desc_filtrado.rename(columns={'CODIGO_SNIES_DEL_PROGRAMA': 'SNIES_Vecino'})
            detalles_referencia = df_desc_filtrado[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO']].rename(columns={
                'CODIGO_SNIES_DEL_PROGRAMA': 'SNIES_Referencia', 'PROGRAMA_ACADEMICO': 'Programa de Referencia'
            })

            dist_con_vecinos = pd.merge(dist_filtrada, detalles_vecino, on='SNIES_Vecino', how='left')
            dist_completa = pd.merge(dist_con_vecinos, detalles_referencia, on='SNIES_Referencia', how='left')

            dist_completa.insert(0, 'Posición', range(1, len(dist_completa) + 1))

            resultados_distancias.set(dist_completa)

            ui.notification_show("Cálculo de distancias completado.", type="success")
            p.set(value=4, message="Finalizado.")

    def check_programa_referencia():
        """Función auxiliar para validar el programa de referencia."""
        programa_ref = input.programa_referencia()
        if not programa_ref or programa_ref == "ninguno":
            return None
        return int(programa_ref)

    @output
    @render.ui
    def tsne_3d_plot():
        programa_ref_snies = check_programa_referencia()
        if programa_ref_snies is None: return
        df_tsne = tsne_resultados.get()
        if df_tsne.empty or 'TSNE3' not in df_tsne.columns:
            return ui.div(ui.h5("Gráfico 3D no disponible"), ui.p("Se requieren 3 o más variables para el cálculo."), style="text-align: center; padding: 20px; height: 1040px; display: grid; place-content: center; border: 1px dashed #ccc;")
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns: return
        n_vecinos = input.n_vecinos()
        
        vecinos = dist_df.nsmallest(n_vecinos + 1, 'Distancia')['SNIES_Vecino'].tolist()
        df_tsne['Categoria'] = 'Otro'
        df_tsne.loc[df_tsne['CODIGO_SNIES_DEL_PROGRAMA'] == programa_ref_snies, 'Categoria'] = 'Referencia'
        vecinos_sin_ref = [v for v in vecinos if v != programa_ref_snies]
        df_tsne.loc[df_tsne['CODIGO_SNIES_DEL_PROGRAMA'].isin(vecinos_sin_ref), 'Categoria'] = 'Vecino'
        df_desc_anio = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
        df_to_plot = pd.merge(df_tsne, df_desc_anio[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO', 'NOMBRE_INSTITUCION']], on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        df_to_plot['hover_text'] = df_to_plot.apply(lambda row: f"<b>SNIES:</b> {row['CODIGO_SNIES_DEL_PROGRAMA']}<br><b>Programa:</b> {row['PROGRAMA_ACADEMICO']}<br><b>IES:</b> {row['NOMBRE_INSTITUCION']}", axis=1)
        fig = go.Figure()
        marker_sizes = {'Referencia': 8, 'Vecino': 6, 'Otro': 2}
        for categoria in ['Otro', 'Vecino', 'Referencia']:
            df_cat = df_to_plot[df_to_plot['Categoria'] == categoria]
            fig.add_trace(go.Scatter3d(x=df_cat['TSNE1'], y=df_cat['TSNE2'], z=df_cat['TSNE3'], mode='markers', name=categoria, text=df_cat['hover_text'], hoverinfo='text', marker=dict(size=marker_sizes.get(categoria, 3))))
        # Vectores negros con "flecha" (línea negra + marcador en el extremo)
        if n_vecinos > 0:
            ref_point = df_to_plot[df_to_plot['Categoria'] == 'Referencia']
            if not ref_point.empty:
                ref_point = ref_point.iloc[0]
                vecinos_points = df_to_plot[df_to_plot['Categoria'] == 'Vecino']
                for i, row in vecinos_points.iterrows():
                    # Línea negra
                    fig.add_trace(go.Scatter3d(x=[ref_point['TSNE1'], row['TSNE1']], y=[ref_point['TSNE2'], row['TSNE2']], z=[ref_point['TSNE3'], row['TSNE3']], mode='lines', name='Vector', line=dict(color='black', width=2), showlegend=False))
                    # "Flecha": marcador rojo en el extremo del vecino
                    fig.add_trace(go.Scatter3d(x=[row['TSNE1']], y=[row['TSNE2']], z=[row['TSNE3']], mode='markers', marker=dict(color='red', size=4, symbol='circle'), showlegend=False, hoverinfo='skip'))
        fig.update_layout(title="t-SNE 3D", height=1040)
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='80vh'))

    @output
    @render.ui
    def tsne_2d_plot():
        programa_ref_snies = check_programa_referencia()
        if programa_ref_snies is None: return
        df_tsne = tsne_resultados.get()
        if df_tsne.empty or 'TSNE2' not in df_tsne.columns:
            return ui.div(ui.h5("Gráfico 2D no disponible"), ui.p("Se requieren 2 o más variables para el cálculo."), style="text-align: center; padding: 20px; height: 800px; display: grid; place-content: center; border: 1px dashed #ccc;")
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns: return
        n_vecinos = input.n_vecinos()
        
        vecinos = dist_df.nsmallest(n_vecinos + 1, 'Distancia')['SNIES_Vecino'].tolist()
        df_tsne['Categoria'] = 'Otro'
        df_tsne.loc[df_tsne['CODIGO_SNIES_DEL_PROGRAMA'] == programa_ref_snies, 'Categoria'] = 'Referencia'
        vecinos_sin_ref = [v for v in vecinos if v != programa_ref_snies]
        df_tsne.loc[df_tsne['CODIGO_SNIES_DEL_PROGRAMA'].isin(vecinos_sin_ref), 'Categoria'] = 'Vecino'
        df_desc_anio = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
        df_to_plot = pd.merge(df_tsne, df_desc_anio[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO', 'NOMBRE_INSTITUCION']], on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        df_to_plot['hover_text'] = df_to_plot.apply(lambda row: f"<b>SNIES:</b> {row['CODIGO_SNIES_DEL_PROGRAMA']}<br><b>Programa:</b> {row['PROGRAMA_ACADEMICO']}<br><b>IES:</b> {row['NOMBRE_INSTITUCION']}", axis=1)
        fig = go.Figure()
        marker_sizes = {'Referencia': 12, 'Vecino': 8, 'Otro': 5}
        for categoria in ['Otro', 'Vecino', 'Referencia']:
            df_cat = df_to_plot[df_to_plot['Categoria'] == categoria]
            fig.add_trace(go.Scatter(x=df_cat['TSNE1'], y=df_cat['TSNE2'], mode='markers', name=categoria, text=df_cat['hover_text'], hoverinfo='text', marker=dict(size=marker_sizes.get(categoria, 6))))
        if n_vecinos > 0:
            ref_point = df_to_plot[df_to_plot['Categoria'] == 'Referencia']
            vecinos_points = df_to_plot[df_to_plot['Categoria'] == 'Vecino']
            if not ref_point.empty and not vecinos_points.empty:
                ref_x, ref_y = ref_point.iloc[0]['TSNE1'], ref_point.iloc[0]['TSNE2']
                for _, row in vecinos_points.iterrows():
                    vec_x, vec_y = row['TSNE1'], row['TSNE2']
                    fig.add_annotation(x=vec_x, y=vec_y, ax=ref_x, ay=ref_y, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#636363")
        fig.update_layout(title="t-SNE 2D", height=800)
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='80vh'))

    @output
    @render.ui
    def pca_2d_plot():
        programa_ref_snies = check_programa_referencia()
        if programa_ref_snies is None: return
        df_pca = pca_resultados.get()
        if df_pca.empty: return
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns: return
        n_vecinos = input.n_vecinos()
        
        vecinos = dist_df.nsmallest(n_vecinos + 1, 'Distancia')['SNIES_Vecino'].tolist()
        df_pca['Categoria'] = 'Otro'
        df_pca.loc[df_pca['CODIGO_SNIES_DEL_PROGRAMA'] == programa_ref_snies, 'Categoria'] = 'Referencia'
        vecinos_sin_ref = [v for v in vecinos if v != programa_ref_snies]
        df_pca.loc[df_pca['CODIGO_SNIES_DEL_PROGRAMA'].isin(vecinos_sin_ref), 'Categoria'] = 'Vecino'
        df_desc_anio = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
        df_to_plot = pd.merge(df_pca, df_desc_anio[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO', 'NOMBRE_INSTITUCION']], on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        df_to_plot['hover_text'] = df_to_plot.apply(lambda row: f"<b>SNIES:</b> {row['CODIGO_SNIES_DEL_PROGRAMA']}<br><b>Programa:</b> {row['PROGRAMA_ACADEMICO']}<br><b>IES:</b> {row['NOMBRE_INSTITUCION']}", axis=1)
        fig = go.Figure()
        marker_sizes = {'Referencia': 12, 'Vecino': 8, 'Otro': 5}
        for categoria in ['Otro', 'Vecino', 'Referencia']:
            df_cat = df_to_plot[df_to_plot['Categoria'] == categoria]
            fig.add_trace(go.Scatter(x=df_cat['PC1'], y=df_cat['PC2'], mode='markers', name=categoria, text=df_cat['hover_text'], hoverinfo='text', marker=dict(size=marker_sizes.get(categoria, 6))))
        if n_vecinos > 0:
            ref_point = df_to_plot[df_to_plot['Categoria'] == 'Referencia']
            vecinos_points = df_to_plot[df_to_plot['Categoria'] == 'Vecino']
            if not ref_point.empty and not vecinos_points.empty:
                ref_x, ref_y = ref_point.iloc[0]['PC1'], ref_point.iloc[0]['PC2']
                for _, row in vecinos_points.iterrows():
                    vec_x, vec_y = row['PC1'], row['PC2']
                    fig.add_annotation(x=vec_x, y=vec_y, ax=ref_x, ay=ref_y, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#636363")
        fig.update_layout(title="PCA 2D", height=800)
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='80vh'))

    @output
    @render.ui
    def pca_3d_plot():
        programa_ref_snies = check_programa_referencia()
        if programa_ref_snies is None: return
        df_pca = pca_resultados.get()
        if df_pca.empty or "PC3" not in df_pca.columns:
            return ui.div(ui.h5("Gráfico 3D no disponible"), ui.p("Se requieren 3 o más variables para el cálculo."), style="text-align: center; padding: 20px; height: 1040px; display: grid; place-content: center; border: 1px dashed #ccc;")
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns: return
        n_vecinos = input.n_vecinos()
        
        vecinos = dist_df.nsmallest(n_vecinos + 1, 'Distancia')['SNIES_Vecino'].tolist()
        df_pca['Categoria'] = 'Otro'
        df_pca.loc[df_pca['CODIGO_SNIES_DEL_PROGRAMA'] == programa_ref_snies, 'Categoria'] = 'Referencia'
        vecinos_sin_ref = [v for v in vecinos if v != programa_ref_snies]
        df_pca.loc[df_pca['CODIGO_SNIES_DEL_PROGRAMA'].isin(vecinos_sin_ref), 'Categoria'] = 'Vecino'
        df_desc_anio = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == int(input.anio_seleccionado())]
        df_to_plot = pd.merge(df_pca, df_desc_anio[['CODIGO_SNIES_DEL_PROGRAMA', 'PROGRAMA_ACADEMICO', 'NOMBRE_INSTITUCION']], on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        df_to_plot['hover_text'] = df_to_plot.apply(lambda row: f"<b>SNIES:</b> {row['CODIGO_SNIES_DEL_PROGRAMA']}<br><b>Programa:</b> {row['PROGRAMA_ACADEMICO']}<br><b>IES:</b> {row['NOMBRE_INSTITUCION']}", axis=1)
        fig = go.Figure()
        marker_sizes = {'Referencia': 6, 'Vecino': 4, 'Otro': 2}
        for categoria in ['Otro', 'Vecino', 'Referencia']:
            df_cat = df_to_plot[df_to_plot['Categoria'] == categoria]
            fig.add_trace(go.Scatter3d(x=df_cat['PC1'], y=df_cat['PC2'], z=df_cat['PC3'], mode='markers', name=categoria, text=df_cat['hover_text'], hoverinfo='text', marker=dict(size=marker_sizes.get(categoria, 3))))
        # Vectores negros con "flecha" (línea negra + marcador en el extremo)
        if n_vecinos > 0:
            ref_point = df_to_plot[df_to_plot['Categoria'] == 'Referencia']
            if not ref_point.empty:
                ref_point = ref_point.iloc[0]
                vecinos_points = df_to_plot[df_to_plot['Categoria'] == 'Vecino']
                for i, row in vecinos_points.iterrows():
                    # Línea negra
                    fig.add_trace(go.Scatter3d(x=[ref_point['PC1'], row['PC1']], y=[ref_point['PC2'], row['PC2']], z=[ref_point['PC3'], row['PC3']], mode='lines', name='Vector', line=dict(color='black', width=2), showlegend=False))
                    # "Flecha": marcador rojo en el extremo del vecino
                    fig.add_trace(go.Scatter3d(x=[row['PC1']], y=[row['PC2']], z=[row['PC3']], mode='markers', marker=dict(color='red', size=4, symbol='circle'), showlegend=False, hoverinfo='skip'))
        fig.update_layout(title="PCA 3D", height=1000)
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='80vh'))

    @output
    @render.data_frame
    def tabla_distancias_output():
        df_resultados = resultados_distancias.get()
        if df_resultados.empty or "Mensaje" in df_resultados.columns:
            return render.DataGrid(pd.DataFrame({"Mensaje": ["Aún no se ha ejecutado el análisis."]}))

        n_vecinos = input.n_vecinos()
        df_display = df_resultados.head(n_vecinos + 1).copy()
        # Enriquecer con columnas de datos descriptivos y preprocesados
        snies_vecinos = df_display['SNIES_Vecino'].astype('Int64') if 'SNIES_Vecino' in df_display.columns else []
        anio = int(input.anio_seleccionado())
        df_desc = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == anio]
        df_pre = DATOS_PROGRAMAS[DATOS_PROGRAMAS['ANNO'] == anio]
        # Unir datos descriptivos (agregar columnas crudas)
        columnas_crudas = ['TOTAL_MATRICULADOS_ANNO', 'TOTAL_PRIMER_CURSO_ANNO', 'NUMERO_CREDITOS', 'COSTO_MATRICULA_ESTUD_NUEVOS']
        df_display = pd.merge(
            df_display,
            df_desc[['CODIGO_SNIES_DEL_PROGRAMA'] + columnas_crudas],
            left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left', suffixes=('', '_desc')
        )
        # Unir datos preprocesados
        pre_cols = [
            'MEAN_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT', 'STD_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT',
            'MEAN_PRO_GEN_MOD_LECTURA_CRITICA_PUNT', 'STD_PRO_GEN_MOD_LECTURA_CRITICA_PUNT',
            'MEAN_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT', 'STD_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT',
            'MEAN_PRO_GEN_MOD_INGLES_PUNT', 'STD_PRO_GEN_MOD_INGLES_PUNT',
            'MEAN_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT', 'STD_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT',
            'MEAN_PRO_GEN_PUNT_GLOBAL', 'STD_PRO_GEN_PUNT_GLOBAL'
        ]
        df_display = pd.merge(df_display, df_pre[['CODIGO_SNIES_DEL_PROGRAMA'] + pre_cols], left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        # Renombrar columnas crudas a amigables
        df_renamed = df_display.rename(columns=NOMBRES_AMIGABLES_TABLAS)
        # Formateo y selección de columnas
        if 'SNIES_Vecino' in df_renamed.columns:
            df_renamed['SNIES_Vecino'] = df_renamed['SNIES_Vecino'].astype('Int64').astype(str)
        if 'SNIES_Referencia' in df_renamed.columns:
            df_renamed['SNIES_Referencia'] = df_renamed['SNIES_Referencia'].astype('Int64').astype(str)
        if 'Distancia' in df_renamed.columns:
            df_renamed['Distancia'] = pd.to_numeric(df_renamed['Distancia'], errors='coerce').map('{:.2f}'.format)
        if 'Costo Matrícula' in df_renamed.columns:
            df_renamed['Costo Matrícula'] = pd.to_numeric(df_renamed['Costo Matrícula'], errors='coerce').apply(
                lambda x: f"$ {x:,.0f}".replace(",", ".") if pd.notna(x) else ''
            )
        
        # Formatear columnas numéricas de puntajes con una cifra decimal
        columnas_puntajes = [
            'Media Razonamiento Cuantitativo', 'Desv. Razonamiento Cuantitativo',
            'Media Lectura Crítica', 'Desv. Lectura Crítica',
            'Media Competencia Ciudadana', 'Desv. Competencia Ciudadana',
            'Media Inglés', 'Desv. Inglés',
            'Media Comunicación Escrita', 'Desv. Comunicación Escrita',
            'Media Puntaje Global', 'Desv. Puntaje Global'
        ]
        
        for col in columnas_puntajes:
            if col in df_renamed.columns:
                df_renamed[col] = pd.to_numeric(df_renamed[col], errors='coerce').apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else ''
                )
        # Filtrar filas sin Programa Académico
        if 'Programa Académico' in df_renamed.columns:
            df_renamed = df_renamed[df_renamed['Programa Académico'].notnull() & (df_renamed['Programa Académico'].astype(str).str.strip() != '')]

        columnas_base = [
            'Posición', 'SNIES_Referencia', 'Programa de Referencia',
            'SNIES_Vecino', 'Programa Académico', 'Institución', 'Distancia',
            'Sector', 'Reconocimiento Ministerio', 'Campo Específico (CINE F 2013)', 'Modalidad', 'Departamento Principal',
            'Matriculados (Año)', 'Inscritos Primer Curso (Año)', 'Créditos', 'Costo Matrícula',
            'Media Razonamiento Cuantitativo', 'Desv. Razonamiento Cuantitativo',
            'Media Lectura Crítica', 'Desv. Lectura Crítica',
            'Media Competencia Ciudadana', 'Desv. Competencia Ciudadana',
            'Media Inglés', 'Desv. Inglés',
            'Media Comunicación Escrita', 'Desv. Comunicación Escrita',
            'Media Puntaje Global', 'Desv. Puntaje Global'
        ]
        final_cols = [col for col in columnas_base if col in df_renamed.columns]
        final_cols += [val for key, val in NOMBRES_AMIGABLES_TABLAS.items() if val not in final_cols and key in df_display.columns]
        return render.DataGrid(df_renamed[final_cols], width="100%", height="600px", selection_mode="rows", filters=True)


    @output
    @render.data_frame
    def tabla_comparacion_caracteristicas_output():
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns:
            return render.DataGrid(pd.DataFrame({"Mensaje": ["Aún no se ha ejecutado el análisis."]}))

        n_vecinos = input.n_vecinos()
        programas_a_comparar = dist_df.head(n_vecinos + 1)
        
        features_df = features_df_reactive().reset_index()
        if features_df.empty or 'CODIGO_SNIES_DEL_PROGRAMA' not in features_df.columns:
            return render.DataGrid(programas_a_comparar)

        tabla_combinada = pd.merge(
            programas_a_comparar,
            features_df,
            left_on='SNIES_Vecino',
            right_on='CODIGO_SNIES_DEL_PROGRAMA',
            how='left'
        )

        tabla_formateada = tabla_combinada.rename(columns=NOMBRES_AMIGABLES_MODELAR)
        tabla_formateada = tabla_formateada.rename(columns=NOMBRES_AMIGABLES_TABLAS)
        
        if 'Distancia' in tabla_formateada.columns:
            tabla_formateada['Distancia'] = pd.to_numeric(tabla_formateada['Distancia'], errors='coerce').map('{:.2f}'.format)
        
        if 'Costo Matrícula' in tabla_formateada.columns:
            tabla_formateada['Costo Matrícula'] = pd.to_numeric(tabla_formateada['Costo Matrícula'], errors='coerce').apply(
                lambda x: f"$ {x:,.0f}".replace(",", ".") if pd.notna(x) else ''
            )

        columnas_id = [
            'Posición', 'SNIES_Referencia', 'Programa de Referencia',
            'SNIES_Vecino', 'Programa Académico', 'Institución', 'Distancia'
        ]
        
        nombres_amigables_features = list(NOMBRES_AMIGABLES_MODELAR.values())
        columnas_features_presentes = sorted([col for col in nombres_amigables_features if col in tabla_formateada.columns])
        
        # Re-aplicar formato de porcentaje después de la unión
        for col in columnas_features_presentes:
            if col != 'SNIES':
                 tabla_formateada[col] = pd.to_numeric(tabla_formateada[col], errors='coerce').apply(lambda x: f"{x:.1%}" if pd.notna(x) else '')

        columnas_a_mostrar = columnas_id + columnas_features_presentes
        
        columnas_finales = [col for col in columnas_a_mostrar if col in tabla_formateada.columns]

        return render.DataGrid(tabla_formateada[columnas_finales], width="100%", height="750px", selection_mode="rows", filters=True)


    @output
    @render.data_frame
    def tabla_datos_para_modelar():
        df = features_df_reactive()
        if df.empty or "Mensaje" in df.columns or "Error" in df.columns:
            return render.DataGrid(df)
        
        df_renamed = df.rename(columns=NOMBRES_AMIGABLES_MODELAR)
        df_formatted = df_renamed.reset_index().rename(columns={'CODIGO_SNIES_DEL_PROGRAMA': 'SNIES'})
        columnas_a_formatear = [col for col in df_formatted.columns if col not in ['SNIES']]
        for col in columnas_a_formatear:
            if pd.api.types.is_numeric_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.1%}")
        return render.DataGrid(df_formatted, width="100%", height="500px", selection_mode="rows", filters=True)


    @output
    @render.data_frame
    def tabla_datos_generales():
        columnas_seleccionadas = list(NOMBRES_AMIGABLES_TABLAS.keys())
        columnas_existentes = [col for col in columnas_seleccionadas if col in DATOS_DESCRIPTIVOS.columns]
        df_display = DATOS_DESCRIPTIVOS[columnas_existentes].rename(columns=NOMBRES_AMIGABLES_TABLAS)

        if 'Costo Matrícula' in df_display.columns:
            df_display['Costo Matrícula'] = pd.to_numeric(df_display['Costo Matrícula'], errors='coerce').apply(
                lambda x: f"$ {x:,.0f}".replace(",", ".") if pd.notna(x) else ''
            )

        return render.DataGrid(df_display, width="100%", height="900px", selection_mode="rows", filters=True)


    @output
    @render.ui
    def grafica_vecindad_matriculados():
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns:
            return ui.HTML("<div style='text-align:center;padding:40px;'>Ejecute el análisis para ver la gráfica.</div>")
        n_vecinos = input.n_vecinos()
        df_display = dist_df.head(n_vecinos + 1).copy()
        
        # Enriquecer con columnas de datos descriptivos y preprocesados (igual que tabla_distancias_output)
        snies_vecinos = df_display['SNIES_Vecino'].astype('Int64') if 'SNIES_Vecino' in df_display.columns else []
        anio = int(input.anio_seleccionado())
        df_desc = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == anio]
        df_pre = DATOS_PROGRAMAS[DATOS_PROGRAMAS['ANNO'] == anio]
        
        # Unir datos descriptivos (agregar columnas crudas)
        columnas_crudas = ['TOTAL_MATRICULADOS_ANNO', 'TOTAL_PRIMER_CURSO_ANNO', 'NUMERO_CREDITOS', 'COSTO_MATRICULA_ESTUD_NUEVOS']
        df_display = pd.merge(
            df_display,
            df_desc[['CODIGO_SNIES_DEL_PROGRAMA'] + columnas_crudas],
            left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left', suffixes=('', '_desc')
        )
        
        # Unir datos preprocesados
        pre_cols = [
            'MEAN_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT', 'STD_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT',
            'MEAN_PRO_GEN_MOD_LECTURA_CRITICA_PUNT', 'STD_PRO_GEN_MOD_LECTURA_CRITICA_PUNT',
            'MEAN_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT', 'STD_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT',
            'MEAN_PRO_GEN_MOD_INGLES_PUNT', 'STD_PRO_GEN_MOD_INGLES_PUNT',
            'MEAN_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT', 'STD_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT',
            'MEAN_PRO_GEN_PUNT_GLOBAL', 'STD_PRO_GEN_PUNT_GLOBAL'
        ]
        df_display = pd.merge(df_display, df_pre[['CODIGO_SNIES_DEL_PROGRAMA'] + pre_cols], left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        
        # Renombrar columnas a amigables
        df_display = df_display.rename(columns=NOMBRES_AMIGABLES_TABLAS)
        # Graficas
        graficas = []
        graficas.append(grafica_barra_variable(df_display.copy(), 'Matriculados (Año)', 'Total Matriculados en la Vecindad', 'Matriculados (Año)'))
        graficas.append(grafica_barra_variable(df_display.copy(), 'Inscritos Primer Curso (Año)', 'Total Inscritos Primer Curso en la Vecindad', 'Inscritos Primer Curso (Año)'))
        graficas.append(grafica_barra_variable(df_display.copy(), 'Costo Matrícula', 'Costo Matrícula en la Vecindad', 'Costo Matrícula'))
        graficas.append(grafica_barra_variable(df_display.copy(), 'Créditos', 'Número de Créditos en la Vecindad', 'Créditos'))
        graficas.append(grafica_puntaje_global_errorbar(df_display.copy()))
        return ui.div(*graficas)

    @output
    @render.ui
    def grafica_saber_pro_vecindad():
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns:
            return ui.HTML("<div style='text-align:center;padding:40px;'>Ejecute el análisis para ver las gráficas de Saber PRO.</div>")
        n_vecinos = input.n_vecinos()
        df_display = dist_df.head(n_vecinos + 1).copy()
        
        # Enriquecer con columnas de datos preprocesados (igual que tabla_distancias_output)
        anio = int(input.anio_seleccionado())
        df_pre = DATOS_PROGRAMAS[DATOS_PROGRAMAS['ANNO'] == anio]
        
        # Unir datos preprocesados
        pre_cols = [
            'MEAN_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT', 'STD_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT',
            'MEAN_PRO_GEN_MOD_LECTURA_CRITICA_PUNT', 'STD_PRO_GEN_MOD_LECTURA_CRITICA_PUNT',
            'MEAN_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT', 'STD_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT',
            'MEAN_PRO_GEN_MOD_INGLES_PUNT', 'STD_PRO_GEN_MOD_INGLES_PUNT',
            'MEAN_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT', 'STD_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT',
            'MEAN_PRO_GEN_PUNT_GLOBAL', 'STD_PRO_GEN_PUNT_GLOBAL'
        ]
        df_display = pd.merge(df_display, df_pre[['CODIGO_SNIES_DEL_PROGRAMA'] + pre_cols], left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
        
        # Renombrar columnas a amigables
        df_display = df_display.rename(columns=NOMBRES_AMIGABLES_TABLAS)
        
        # Crear gráficas para cada puntaje Saber PRO
        tarjetas_graficas = []
        
        # Definir las gráficas a crear
        graficas_info = [
            ('Media Puntaje Global', 'Desv. Puntaje Global', 'Puntaje Global en la Vecindad', 'Media Puntaje Global'),
            ('Media Razonamiento Cuantitativo', 'Desv. Razonamiento Cuantitativo', 'Razonamiento Cuantitativo en la Vecindad', 'Media Razonamiento Cuantitativo'),
            ('Media Lectura Crítica', 'Desv. Lectura Crítica', 'Lectura Crítica en la Vecindad', 'Media Lectura Crítica'),
            ('Media Competencia Ciudadana', 'Desv. Competencia Ciudadana', 'Competencia Ciudadana en la Vecindad', 'Media Competencia Ciudadana'),
            ('Media Inglés', 'Desv. Inglés', 'Inglés en la Vecindad', 'Media Inglés'),
            ('Media Comunicación Escrita', 'Desv. Comunicación Escrita', 'Comunicación Escrita en la Vecindad', 'Media Comunicación Escrita')
        ]
        
        for col_media, col_std, titulo, x_label in graficas_info:
            grafica_html = grafica_puntaje_saber_pro(df_display.copy(), col_media, col_std, titulo, x_label)
            tarjeta = ui.card(ui.card_header(titulo), grafica_html, full_screen=True)
            tarjetas_graficas.append(tarjeta)
        
        return ui.div(*tarjetas_graficas)

    @output
    @render.download(filename=lambda: f"Analisis_Vecindad_{input.programa_referencia()}_{input.anio_seleccionado()}.xlsx")
    def descargar_excel():
        dist_df = resultados_distancias.get()
        if dist_df.empty or 'SNIES_Vecino' not in dist_df.columns:
            yield "No hay datos para descargar. Por favor, ejecute el análisis primero."
            return
        
        # Obtener datos para la tabla de distancias (igual que tabla_distancias_output)
        n_vecinos = input.n_vecinos()
        df_display = dist_df.head(n_vecinos + 1).copy()
        
        with io.BytesIO() as output:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Obtener datos para la tabla de distancias (igual que tabla_distancias_output)
                df_display = dist_df.head(n_vecinos + 1).copy()
                
                # Enriquecer con columnas de datos descriptivos y preprocesados
                anio = int(input.anio_seleccionado())
                df_desc = DATOS_DESCRIPTIVOS[DATOS_DESCRIPTIVOS['ANNO'] == anio]
                df_pre = DATOS_PROGRAMAS[DATOS_PROGRAMAS['ANNO'] == anio]
                
                # Unir datos descriptivos
                columnas_crudas = ['TOTAL_MATRICULADOS_ANNO', 'TOTAL_PRIMER_CURSO_ANNO', 'NUMERO_CREDITOS', 'COSTO_MATRICULA_ESTUD_NUEVOS']
                df_display = pd.merge(
                    df_display,
                    df_desc[['CODIGO_SNIES_DEL_PROGRAMA'] + columnas_crudas],
                    left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left', suffixes=('', '_desc')
                )
                
                # Unir datos preprocesados
                pre_cols = [
                    'MEAN_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT', 'STD_PRO_GEN_MOD_RAZONA_CUANTITAT_PUNT',
                    'MEAN_PRO_GEN_MOD_LECTURA_CRITICA_PUNT', 'STD_PRO_GEN_MOD_LECTURA_CRITICA_PUNT',
                    'MEAN_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT', 'STD_PRO_GEN_MOD_COMPETEN_CIUDADA_PUNT',
                    'MEAN_PRO_GEN_MOD_INGLES_PUNT', 'STD_PRO_GEN_MOD_INGLES_PUNT',
                    'MEAN_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT', 'STD_PRO_GEN_MOD_COMUNI_ESCRITA_PUNT',
                    'MEAN_PRO_GEN_PUNT_GLOBAL', 'STD_PRO_GEN_PUNT_GLOBAL'
                ]
                df_display = pd.merge(df_display, df_pre[['CODIGO_SNIES_DEL_PROGRAMA'] + pre_cols], left_on='SNIES_Vecino', right_on='CODIGO_SNIES_DEL_PROGRAMA', how='left')
                
                # Renombrar columnas a amigables
                df_renamed = df_display.rename(columns=NOMBRES_AMIGABLES_TABLAS)
                
                # Filtrar filas sin Programa Académico
                if 'Programa Académico' in df_renamed.columns:
                    df_renamed = df_renamed[df_renamed['Programa Académico'].notnull() & (df_renamed['Programa Académico'].astype(str).str.strip() != '')]
                
                # Seleccionar columnas para Excel
                columnas_base = [
                    'Posición', 'SNIES_Referencia', 'Programa de Referencia',
                    'SNIES_Vecino', 'Programa Académico', 'Institución', 'Distancia',
                    'Sector', 'Reconocimiento Ministerio', 'Campo Específico (CINE F 2013)', 'Modalidad', 'Departamento Principal',
                    'Matriculados (Año)', 'Inscritos Primer Curso (Año)', 'Créditos', 'Costo Matrícula',
                    'Media Razonamiento Cuantitativo', 'Desv. Razonamiento Cuantitativo',
                    'Media Lectura Crítica', 'Desv. Lectura Crítica',
                    'Media Competencia Ciudadana', 'Desv. Competencia Ciudadana',
                    'Media Inglés', 'Desv. Inglés',
                    'Media Comunicación Escrita', 'Desv. Comunicación Escrita',
                    'Media Puntaje Global', 'Desv. Puntaje Global'
                ]
                final_cols = [col for col in columnas_base if col in df_renamed.columns]
                final_cols += [val for key, val in NOMBRES_AMIGABLES_TABLAS.items() if val not in final_cols and key in df_display.columns]
                
                # Tabla 1: Análisis de Distancias
                tabla_distancias = df_renamed[final_cols]
                tabla_distancias.to_excel(writer, sheet_name='Análisis de Distancias', index=False)
                
                # Tabla 2: Comparación de Características
                features_df = features_df_reactive().reset_index()
                if not features_df.empty and 'CODIGO_SNIES_DEL_PROGRAMA' in features_df.columns:
                    tabla_comparacion = pd.merge(
                        dist_df.head(n_vecinos + 1),
                        features_df,
                        left_on='SNIES_Vecino',
                        right_on='CODIGO_SNIES_DEL_PROGRAMA',
                        how='left'
                    )
                    tabla_comparacion = tabla_comparacion.rename(columns=NOMBRES_AMIGABLES_MODELAR)
                    tabla_comparacion = tabla_comparacion.rename(columns=NOMBRES_AMIGABLES_TABLAS)
                    
                    columnas_id = [
                        'Posición', 'SNIES_Referencia', 'Programa de Referencia',
                        'SNIES_Vecino', 'Programa Académico', 'Institución', 'Distancia'
                    ]
                    
                    nombres_amigables_features = list(NOMBRES_AMIGABLES_MODELAR.values())
                    columnas_features_presentes = sorted([col for col in nombres_amigables_features if col in tabla_comparacion.columns])
                    
                    columnas_a_mostrar = columnas_id + columnas_features_presentes
                    columnas_finales = [col for col in columnas_a_mostrar if col in tabla_comparacion.columns]
                    tabla_comparacion = tabla_comparacion[columnas_finales]
                    tabla_comparacion.to_excel(writer, sheet_name='Comparación de Características', index=False)
                else:
                    pd.DataFrame({"Mensaje": ["No hay datos de características disponibles."]}) \
                      .to_excel(writer, sheet_name='Comparación de Características', index=False)
            
            yield output.getvalue()

# --- Funciones Auxiliares ---
def grafica_puntaje_global_errorbar(df_display):
    # Filtrar solo registros con nombre de programa
    if 'Programa Académico' in df_display.columns:
        df_display = df_display[df_display['Programa Académico'].notnull() & (df_display['Programa Académico'].astype(str).str.strip() != '')]
    
    # Buscar nombres crudos y amigables para media y desviación
    col_media = None
    col_std = None
    
    for col in ['Media Puntaje Global', 'MEAN_PRO_GEN_PUNT_GLOBAL']:
        if col in df_display.columns:
            col_media = col
            break
    
    for col in ['Desv. Puntaje Global', 'STD_PRO_GEN_PUNT_GLOBAL']:
        if col in df_display.columns:
            col_std = col
            break
    
    if not col_media:
        return ui.HTML("<div style='text-align:center;padding:40px;'>No se encontró la columna de media de puntaje global.</div>")
    
    if not col_std:
        return ui.HTML("<div style='text-align:center;padding:40px;'>No se encontró la columna de desviación estándar de puntaje global.</div>")
    
    # Convertir a numérico y ordenar por media descendente
    df_display[col_media] = pd.to_numeric(df_display[col_media], errors='coerce')
    df_display[col_std] = pd.to_numeric(df_display[col_std], errors='coerce')
    df_display = df_display.sort_values(col_media, ascending=False)
    
    etiquetas_y = df_display.apply(
        lambda row: f"{row['SNIES_Vecino']} - {row['Programa Académico']} ({row['Institución']})" if 'SNIES_Vecino' in row and 'Programa Académico' in row and 'Institución' in row else str(row.get('SNIES_Vecino', '')), axis=1
    )
    
    x_vals = df_display[col_media]
    error_vals = df_display[col_std]
    programa_ref_snies = df_display['SNIES_Referencia'].iloc[0] if 'SNIES_Referencia' in df_display.columns and len(df_display) > 0 else None
    colores = ['red' if str(row['SNIES_Vecino']) == str(programa_ref_snies) else 'blue' for _, row in df_display.iterrows()]
    
    fig = go.Figure()
    
    # Agregar puntos con barras de error
    fig.add_trace(go.Scatter(
        y=etiquetas_y,
        x=x_vals,
        mode='markers',
        marker=dict(
            color=colores,
            size=8,
            line=dict(width=1, color='black')
        ),
        error_x=dict(
            type='data',
            array=error_vals,
            visible=True,
            color='gray',
            thickness=1,
            width=3
        ),
        name='Media Puntaje Global',
        text=[f"Media: {x:.1f}<br>Desv: {e:.1f}" for x, e in zip(x_vals, error_vals)],
        hovertemplate='%{text}<br>%{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Media Puntaje Global en la Vecindad (con Desviación Estándar)',
        xaxis_title='Media Puntaje Global',
        yaxis_title='SNIES - Programa (Institución)',
        height=600,
        margin=dict(l=250),
        showlegend=False
    )
    
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

def grafica_barra_variable(df_display, variable, titulo, x_label):
    # Filtrar solo registros con nombre de programa
    if 'Programa Académico' in df_display.columns:
        df_display = df_display[df_display['Programa Académico'].notnull() & (df_display['Programa Académico'].astype(str).str.strip() != '')]
    # Ordenar por variable descendente
    if variable in df_display.columns:
        df_display[variable] = pd.to_numeric(df_display[variable], errors='coerce')
        df_display = df_display.sort_values(variable, ascending=False)
    etiquetas_y = df_display.apply(
        lambda row: f"{row['SNIES_Vecino']} - {row['Programa Académico']} ({row['Institución']})" if 'SNIES_Vecino' in row and 'Programa Académico' in row and 'Institución' in row else str(row.get('SNIES_Vecino', '')), axis=1
    )
    x_vals = df_display[variable] if variable in df_display.columns else []
    programa_ref_snies = df_display['SNIES_Referencia'].iloc[0] if 'SNIES_Referencia' in df_display.columns and len(df_display) > 0 else None
    colores = ['red' if str(row['SNIES_Vecino']) == str(programa_ref_snies) else 'blue' for _, row in df_display.iterrows()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=etiquetas_y,
        x=x_vals,
        orientation='h',
        marker=dict(color=colores),
        text=x_vals,
        textposition='auto',
        name=variable
    ))
    fig.update_layout(
        title=titulo,
        xaxis_title=x_label,
        yaxis_title="SNIES - Programa (Institución)",
        height=600,
        margin=dict(l=250)
    )
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

def grafica_puntaje_saber_pro(df_display, col_media, col_std, titulo, x_label):
    # Filtrar solo registros con nombre de programa
    if 'Programa Académico' in df_display.columns:
        df_display = df_display[df_display['Programa Académico'].notnull() & (df_display['Programa Académico'].astype(str).str.strip() != '')]
    
    # Verificar que las columnas existan
    if col_media not in df_display.columns:
        return ui.HTML(f"<div style='text-align:center;padding:40px;'>No se encontró la columna '{col_media}'.</div>")
    
    if col_std not in df_display.columns:
        return ui.HTML(f"<div style='text-align:center;padding:40px;'>No se encontró la columna '{col_std}'.</div>")
    
    # Convertir a numérico y ordenar por media descendente
    df_display[col_media] = pd.to_numeric(df_display[col_media], errors='coerce')
    df_display[col_std] = pd.to_numeric(df_display[col_std], errors='coerce')
    df_display = df_display.sort_values(col_media, ascending=False)
    
    etiquetas_y = df_display.apply(
        lambda row: f"{row['SNIES_Vecino']} - {row['Programa Académico']} ({row['Institución']})" if 'SNIES_Vecino' in row and 'Programa Académico' in row and 'Institución' in row else str(row.get('SNIES_Vecino', '')), axis=1
    )
    
    x_vals = df_display[col_media]
    error_vals = df_display[col_std]
    programa_ref_snies = df_display['SNIES_Referencia'].iloc[0] if 'SNIES_Referencia' in df_display.columns and len(df_display) > 0 else None
    colores = ['red' if str(row['SNIES_Vecino']) == str(programa_ref_snies) else 'blue' for _, row in df_display.iterrows()]
    
    fig = go.Figure()
    
    # Agregar puntos con barras de error
    fig.add_trace(go.Scatter(
        y=etiquetas_y,
        x=x_vals,
        mode='markers',
        marker=dict(
            color=colores,
            size=8,
            line=dict(width=1, color='black')
        ),
        error_x=dict(
            type='data',
            array=error_vals,
            visible=True,
            color='gray',
            thickness=1,
            width=3
        ),
        name=col_media,
        text=[f"Media: {x:.1f}<br>Desv: {e:.1f}" for x, e in zip(x_vals, error_vals)],
        hovertemplate='%{text}<br>%{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title=x_label,
        yaxis_title='SNIES - Programa (Institución)',
        height=600,
        margin=dict(l=250),
        showlegend=False
    )
    
    return ui.HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))


app = App(app_ui, server)
