# Usa una imagen base oficial de Python
FROM python:3.9

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de tu código a la carpeta /app del contenedor
COPY . /app

# Actualiza el sistema e instala las dependencias necesarias
RUN apt-get update && \
    apt-get install -y \
    python3-pip && \
    pip install --upgrade pip

# Crea un entorno virtual y activa el entorno
RUN python -m venv /app/venv

# Asegúrate de usar el entorno virtual y actualizar pip
RUN /app/venv/bin/pip install --upgrade pip

# Instala las dependencias dentro del entorno virtual
RUN /app/venv/bin/pip install pandas numpy influxdb-client requests scikit-learn python-dotenv flask

# Establece la variable de entorno para usar el entorno virtual de Python
ENV PATH="/app/venv/bin:$PATH"

# Expone el puerto 5000 para que Flask sea accesible fuera del contenedor
EXPOSE 5000

# Establece el contenedor para ejecutar la aplicación web.py en el entorno virtual
CMD ["/app/venv/bin/python", "web.py"]
# Usa una imagen base oficial de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos al contenedor
COPY . .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 5000 para Flask
EXPOSE 5000

# Usa Gunicorn para servir la aplicación desde webapp
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "server.wsgi:app"]
