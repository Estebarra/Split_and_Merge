# Split and Merge - Segmentación de Imágenes

Aplicación web interactiva para segmentación de imágenes utilizando el algoritmo **Split and Merge** con estructura Quad-Tree.

## 📋 Descripción

Esta aplicación implementa el algoritmo de segmentación Split and Merge, que divide recursivamente una imagen en regiones homogéneas utilizando una estructura de datos tipo Quad-Tree, y posteriormente fusiona regiones similares adyacentes. El resultado es una segmentación efectiva de la imagen basada en características de homogeneidad.

### Características principales:
- Segmentación basada en Quad-Tree
- Preprocesamiento con filtros Gaussiano o Mediana
- Postprocesamiento para fusionar regiones adyacentes similares
- Interfaz web interactiva con Streamlit
- Visualización de regiones detectadas
- Ajuste de parámetros en tiempo real

## 🏗️ Estructura del Proyecto

```
Split_and_Merge/
├── src/
│   └── main.py              # Aplicación principal con lógica de segmentación
├── test_images/             # Imágenes de prueba
│   ├── image_0.PNG
│   ├── image_1.PNG
│   ├── imagen_1.png
│   └── ...
├── docker-compose.yml       # Configuración de Docker Compose
├── dockerfile               # Definición del contenedor Docker
├── pyproject.toml          # Dependencias y configuración del proyecto
├── .env.example            # Ejemplo de variables de entorno
└── README.md               # Documentación
```

## 🔧 Tecnologías

- **Python 3.12**: Lenguaje de programación
- **Streamlit**: Framework para la interfaz web interactiva
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas y manejo de arrays
- **Pillow**: Carga y manipulación de imágenes
- **UV**: Gestor de paquetes ultrarrápido para Python
- **Docker**: Containerización de la aplicación

## 🚀 Instalación y Ejecución

### Opción 1: Usando Docker Compose (Recomendado)

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/Estebarra/Split_and_Merge.git
   cd Split_and_Merge
   ```

2. **Configurar variables de entorno** (opcional)

   El proyecto incluye un archivo `.env.example` con la configuración por defecto:
   ```env
   STREAMLIT_SERVER_PORT=8501
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

   Para usar una configuración personalizada, copia el archivo de ejemplo:
   ```bash
   cp .env.example .env
   ```

   Y modifica los valores según tus necesidades.

3. **Iniciar la aplicación con Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Acceder a la aplicación**

   Abrir el navegador en: [http://localhost:8501](http://localhost:8501)

5. **Detener la aplicación**
   ```bash
   docker-compose down
   ```

### Opción 2: Usando Docker directamente

1. **Construir la imagen**
   ```bash
   docker build -t split-merge-app .
   ```

2. **Ejecutar el contenedor**
   ```bash
   docker run -p 8501:8501 split-merge-app
   ```

3. **Acceder a la aplicación**

   Abrir el navegador en: [http://localhost:8501](http://localhost:8501)

### Opción 3: Ejecución local con UV (sin Docker)

1. **Instalar UV** (si no lo tienes instalado)
   ```bash
   pip install uv
   ```

2. **Instalar dependencias**
   ```bash
   uv pip install -r pyproject.toml
   ```

3. **Ejecutar la aplicación**
   ```bash
   streamlit run src/main.py
   ```

## 🎯 Uso de la Aplicación

1. **Cargar imagen**: Usa el selector de archivos en el panel lateral para cargar una imagen (PNG, JPG, JPEG, BMP)

2. **Ajustar parámetros**:
   - **Min Size**: Tamaño mínimo de región (4-64 píxeles)
   - **Std Threshold**: Umbral de desviación estándar para determinar homogeneidad (1.0-50.0)
   - **Mean Threshold**: Umbral de diferencia de medias para fusionar regiones (1.0-50.0)

3. **Configurar filtros**:
   - **Tipo de filtro**: Gaussiano o Mediana
   - **Kernel Size**: Tamaño del kernel del filtro (3-11)
   - **Aplicar preprocesamiento**: Activar/desactivar filtrado previo
   - **Aplicar postprocesamiento**: Activar/desactivar fusión de regiones adyacentes

4. **Iniciar segmentación**: Presionar el botón "Iniciar Segmentación"

5. **Visualizar resultados**:
   - Imagen original
   - Imagen preprocesada (si está activado)
   - Imagen segmentada
   - Imagen postprocesada (si está activado)
   - Visualización de regiones con bordes

## 🧮 Algoritmo Split and Merge

### Fase Split (División)
La imagen se divide recursivamente en cuadrantes si:
- El tamaño de la región es mayor que `min_size`
- La desviación estándar de la región es mayor que `std_threshold`

### Fase Merge (Fusión)
Las regiones adyacentes se fusionan si:
- Ambas regiones son hojas del árbol
- La diferencia de sus valores medios es menor que `mean_threshold`

### Métricas calculadas por región:
- Media (mean)
- Desviación estándar (std)
- Valor máximo y mínimo
- Mediana

## 📦 Dependencias

```toml
streamlit>=1.39.0
opencv-python-headless>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

## 🐳 Configuración Docker

El proyecto incluye configuración completa de Docker:

- **Imagen base**: `python:3.12-slim`
- **Puerto expuesto**: `8501`
- **Gestor de paquetes**: UV para instalación rápida
- **Entrypoint**: Streamlit server configurado para acceso externo

## 📝 Notas

- La aplicación convierte automáticamente imágenes a color a escala de grises
- Se incluyen imágenes de prueba en la carpeta `test_images/`
- El algoritmo es más efectivo con imágenes que tienen regiones claramente diferenciadas
- El preprocesamiento ayuda a reducir ruido y mejorar la segmentación
- El postprocesamiento permite fusionar regiones muy similares que quedaron separadas

## 🤝 Contribuciones

Este proyecto es parte de un trabajo académico de Visión por Computadora.

## 📄 Licencia

Proyecto académico - MNA Computer Vision