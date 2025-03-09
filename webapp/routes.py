from flask import Blueprint, render_template, render_template_string

import plotly.express as px
import io
import base64
import os
from dotenv import load_dotenv, dotenv_values # type: ignore
from confort.lib.prepara_data import PreparaData # type: ignore

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
    data = entrenador.prepara_datos ()
#    return jsonify(ultimo_valor) 
    return data  # Devuelve JSON automáticamente

# Ruta para obtener los datos en formato DataFrame
@bp.route('/dataFrame')
def get_dataFrame():
    entrenador = PreparaData(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    data = entrenador.prepara_datos ()
    df = entrenador.dataFrame
    html_table = df.to_html(classes='table table-bordered table-striped')

    # Crear el gráfico de temperatura
    fig_temperatura = px.line(df, x='timestamp', y='temperature', title='Temperatura a lo largo del tiempo', labels={'temperatura': 'Temperatura (°C)'},line_shape='linear')
    fig_temperatura.add_scatter(x=entrenador.prediccion['timestamp'], y=entrenador.prediccion['temperature'], mode='markers', name='Predicción', marker=dict(color='red', size=10))

    # Crear el gráfico de humedad
    fig_humedad = px.line(df, x='timestamp', y='humidity', title='Humedad a lo largo del tiempo', labels={'humedad': 'Humedad (%)'},line_shape='linear')
    fig_humedad.add_scatter(x=entrenador.prediccion['timestamp'], y=entrenador.prediccion['humidity'], mode='markers', name='Predicción', marker=dict(color='blue', size=10))
    
    # Convertir los gráficos a formato HTML para insertar en la plantilla
    graph_html_temperatura = fig_temperatura.to_html(full_html=False)
    graph_html_humedad  = fig_humedad.to_html(full_html=True)

    return render_template('grafica.html', 
        table=html_table, 
        graph_temperatura=graph_html_temperatura, 
        graph_humedad=graph_html_humedad)

