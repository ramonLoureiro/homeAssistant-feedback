# Creamos imagen my-python-influxdb
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

# Instala las dependencias de Python necesarias
RUN pip install influxdb-client requests numpy pandas scikit-learn python-dotenv


# Establece el contenedor para que no ejecute nada automáticamente
CMD ["/bin/bash"]
