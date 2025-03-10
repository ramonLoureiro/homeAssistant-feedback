from flask import Blueprint, request, jsonify, render_template, render_template_string

import numpy as np
import pandas as pd

import plotly.express as px
import io
import base64
import os
from dotenv import load_dotenv, dotenv_values # type: ignore
from confort.lib.prepara_data import PreparaData # type: ignore
from confort.lib.smooth_data import SmoothData # type: ignore

# Cargar variables desde .env
load_dotenv(override=True)
config = dotenv_values(".env")


# Priorizar valores de .env sobre variables de entorno
INFLUXDB_URL = config.get('INFLUXDB_URL') or os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN= config.get('INFLUXDB_TOKEN') or os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = config.get('INFLUXDB_ORG') or os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = config.get('INFLUXDB_BUCKET') or os.getenv('INFLUXDB_BUCKET')



bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    return render_template('indice.html')

@bp.route('/about')
def about():
    return render_template('about.html')


@bp.route('/data')
def get_data():
    entrenador = PreparaData(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    data = entrenador.prepara_datos (10,15)
    return data  # Devuelve JSON automáticamente

# Ruta para obtener los datos en formato DataFrame
@bp.route('/dataFrame')
def get_dataFrame():
    entrenador = PreparaData(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    data = entrenador.prepara_datos (10,15)
    df = entrenador.dataFrame

    smoother = SmoothData(df)
    smoother.execute('temperature_smooth','temperature')
    smoother.execute('humidity_smooth','humidity')


    html_table = entrenador.dataFrame_total.tail(20).to_html(classes='table table-bordered table-striped')

    # Crear el gráfico de temperatura
    fig_temperatura = px.line(
        df, 
        x='timestamp', 
        y='temperature_smooth', 
        title='Temperatura a lo largo del tiempo', 
        labels={'temperatura': 'Temperatura (°C)'},
        line_shape='linear')
    fig_temperatura.add_scatter(
        x=entrenador.prediccion['timestamp'], 
        y=entrenador.prediccion['temperature'], 
        mode='markers', 
        name='Predicción', 
        marker=dict(color='red', size=10))

    # Crear el gráfico de humedad
    fig_humedad = px.line(
        df, 
        x='timestamp', 
        y='humidity_smooth', 
        title='Humedad a lo largo del tiempo', 
        labels={'humedad': 'Humedad (%)'},
        line_shape='linear')
    fig_humedad.add_scatter(
        x=entrenador.prediccion['timestamp'], 
        y=entrenador.prediccion['humidity'], 
        mode='markers', 
        name='Predicción', 
        marker=dict(color='blue', size=10))
    
    # Convertir los gráficos a formato HTML para insertar en la plantilla
    graph_html_temperatura = fig_temperatura.to_html(full_html=False)
    graph_html_humedad  = fig_humedad.to_html(full_html=True)

    return render_template(
        'grafica.html', 
        table=html_table, 
        graph_temperatura=graph_html_temperatura, 
        graph_humedad=graph_html_humedad)


@bp.route('/correlation', methods=['GET'])
def get_correlation():
    dias  = request.args.get('dias' , type=int)
    media = request.args.get('media', type=int)
    entrenador = PreparaData(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    data = entrenador.prepara_datos (dias,media)
    df = entrenador.dataFrame_total

    smoother = SmoothData(df)
    smoother.execute('temperature_smooth','temperature')
    smoother.execute('humidity_smooth','humidity')


    # Eliminar nulos
    df_clean = df[['temperature', 'temperature_ext']].dropna()

    # Convertir a arrays
    x = df_clean['temperature'].values
    y = df_clean['temperature_ext'].values

    # Calcular la correlación cruzada
    correlation = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    lags = np.arange(-len(x) + 1, len(x))

    # Crear dataframe para plotly
    df_corr = pd.DataFrame({'lag': lags, 'correlation': correlation})


    # Lag óptimo (máxima correlación)
    best_lag = lags[np.argmax(correlation)]
    # Traducir lag a tiempo real (si los datos están cada 5 minutos, por ejemplo)
    sampling_rate = media  # minutos
    lag_minutes = best_lag * sampling_rate
    print(f"El desfase máximo es de aproximadamente {lag_minutes} minutos")

    title = f'Correlación cruzada entre temperatura exterior e interior ({lag_minutes:.2f} minutos)'

    # Crear la gráfica con el título dinámico
    fig = px.line(df_corr, x='lag', y='correlation', title=title)

    # Línea vertical en lag=0
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Sin desfase", annotation_position="top")
    fig.add_vline(x=best_lag, line_dash="dot", line_color="green",
                annotation_text=f"Lag óptimo: {best_lag}", annotation_position="top right")

    # Mostrar el gráfico interactivo
#    fig.show()

    
    # Convertir los gráficos a formato HTML para insertar en la plantilla
    graph_html = fig.to_html(full_html=False)

    return render_template(
        'grafica-correlacion.html', 
        graph_correlacion=graph_html)
