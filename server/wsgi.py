import sys
import os

# Añadir el directorio raíz al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webapp import create_app

app = create_app()
app.config['DEBUG'] = True  # Habilitar modo de depuración
