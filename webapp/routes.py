from flask import Blueprint, render_template
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
    return render_template('index.html')

@bp.route('/about')
def about():
    return "Esta es la página 'About'."


@bp.route('/data')
def get_data():
    preparador = PreparaData(INFLUXDB_URL, INFLUXDB_TOKEN, INFLUXDB_ORG, INFLUXDB_BUCKET)
    data = preparador.prepara_datos ()
#    return jsonify(ultimo_valor) 
    return data  # Devuelve JSON automáticamente
