# Usar la imagen oficial ligera de Python 3.12
FROM python:3.12-slim

# Evitar que Python escriba archivos .pyc y forzar salida limpia en consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# Actualizar pip e instalar dependencias si existe requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copiar el resto del código del proyecto al contenedor
COPY . .

# Comando por defecto para abrir una terminal interactiva
CMD ["bash"]
