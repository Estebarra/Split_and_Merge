import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Segmentación Split and Merge", layout="wide")

class quad_image:
    """
    Implementación de una estructura del tipo quad-tree para manejo de imágenes.
    """
    def __init__(self, img: np.ndarray, x: int, y: int, width: int, height: int):
        self.img = img
        self.height = height
        self.width = width
        self.x = x
        self.y = y

        region = self.img[self.y:self.y+self.height, self.x:self.x+self.width]

        self.mean = np.mean(region)
        self.std = np.std(region)
        self.max = np.max(region)
        self.min = np.min(region)
        self.median = np.median(region)

        self.children = None
        self.is_leaf = True

    def _split(self):
        """Divide la región en 4 cuadrantes."""
        new_width = self.width//2
        new_height = self.height//2

        NW_x = self.x
        NW_y = self.y

        NE_x = self.x + new_width
        NE_y = self.y

        SW_x = self.x
        SW_y = self.y + new_height

        SE_x = self.x + new_width
        SE_y = self.y + new_height

        NW = quad_image(self.img, NW_x, NW_y, new_width, new_height)
        NE = quad_image(self.img, NE_x, NE_y, (self.width - new_width), new_height)
        SW = quad_image(self.img, SW_x, SW_y, new_width, (self.height - new_height))
        SE = quad_image(self.img, SE_x, SE_y, (self.width - new_width), (self.height - new_height))

        self.children = [NE, NW, SW, SE]
        self.is_leaf = False

    def _is_homogeneous(self, std_threshold):
        """Determina si la región es homogénea."""
        return self.std < std_threshold

    def build_tree(self, min_size, std_threshold):
        """Construye el quad-tree recursivamente."""
        if self.width <= min_size or self.height <= min_size:
            return
        if self._is_homogeneous(std_threshold):
            return

        self._split()

        for child in self.children:
            child.build_tree(min_size, std_threshold)


def requires_segmentation(func):
    """Decorador para verificar que se ejecutó segment()."""
    def wrapper(self, *args, **kwargs):
        if self.root is None:
            st.error("Error: Primero debe ejecutarse segment()")
            return None
        return func(self, *args, **kwargs)
    return wrapper


class split_and_merge:
    """Implementación del algoritmo split and merge."""
    
    def __init__(self, image, min_size, std_threshold, mean_threshold=10):
        self.image = image
        self.min_size = min_size
        self.std_threshold = std_threshold
        self.mean_threshold = mean_threshold
        self.root = None

    @staticmethod
    def count_leafs(node):
        """Cuenta el número de regiones en un quad-tree."""
        if node.is_leaf:
            return 1
        else:
            return sum(split_and_merge.count_leafs(child) for child in node.children)

    @staticmethod
    def draw_regions(node, image_with_regions):
        """Dibuja los bordes de las regiones."""
        if node.is_leaf:
            cv2.rectangle(image_with_regions,
                        (node.x, node.y),
                        (node.x + node.width, node.y+node.height),
                        color=(0,255,0),
                        thickness=1)
        else:
            for child in node.children:
                split_and_merge.draw_regions(child, image_with_regions)

    def _siblings_are_similar(self, children):
        """Determina si los hijos de un nodo son similares."""
        means = [child.mean for child in children]
        return (max(means)-min(means)) < self.mean_threshold

    def _recursive_merge(self, node):
        """Realiza la operación de fusión recursiva."""
        if node.is_leaf:
            return

        for child in node.children:
            self._recursive_merge(child)

        all_are_leafs = all(child.is_leaf for child in node.children)
        siblings_are_similar = self._siblings_are_similar(node.children)

        if all_are_leafs and siblings_are_similar:
            node.is_leaf = True
            node.children = None
            return True

    def _merge(self):
        """Ejecuta la operación de fusión recursiva."""
        while True:
            before_number_of_regions = self.count_leafs(self.root)
            self._recursive_merge(self.root)
            after_number_of_regions = self.count_leafs(self.root)

            if before_number_of_regions == after_number_of_regions:
                break

    def _fill_regions(self, node, segmented):
        """Rellena las regiones de la imagen segmentada."""
        if node.is_leaf:
            segmented[node.y:node.y+node.height, node.x:node.x+node.width] = node.mean
        else:
            for child in node.children:
                self._fill_regions(child, segmented)

    def pre_process(self, filter_type='gaussian', kernel_size=5):
        """Preprocesamiento con filtro de suavizado."""
        if filter_type == 'gaussian':
            self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            self.image = cv2.medianBlur(self.image, kernel_size)
        else:
            raise ValueError("Tipo de filtro no válido. Use 'gaussian' o 'median'.")
        return self

    def segment(self):
        """Ejecuta el algoritmo split and merge."""
        height, width = self.image.shape
        self.root = quad_image(self.image, x=0, y=0, width=width, height=height)
        self.root.build_tree(self.min_size, self.std_threshold)
        self._merge()

    @requires_segmentation
    def post_process(self):
        """Postprocesamiento de la imagen segmentada."""
        segmented = self.get_segmented_image()

        unique_values = np.unique(segmented)
        labeled_image = np.zeros_like(segmented, dtype=np.int32)

        for idx, value in enumerate(unique_values, start=1):
            labeled_image[segmented == value] = idx

        labels = labeled_image.copy()
        num_labels = len(unique_values) + 1

        region_means = {}
        for label in range(1, num_labels):
            mask = (labels == label)
            region_means[label] = np.mean(segmented[mask])

        kernel = np.ones((3, 3), np.uint8)
        merged = True

        while merged:
            merged = False

            for label in range(1, num_labels):
                if label not in region_means:
                    continue

                mask = (labels == label).astype(np.uint8)
                dilated = cv2.dilate(mask, kernel, iterations=1)
                neighbors = np.unique(labels[dilated == 1])
                neighbors = neighbors[(neighbors != label) & (neighbors != 0)]

                for neighbor in neighbors:
                    if neighbor not in region_means:
                        continue

                    diff = abs(region_means[label] - region_means[neighbor])

                    if diff < self.mean_threshold:
                        labels[labels == neighbor] = label
                        mask_combined = (labels == label)
                        region_means[label] = np.mean(segmented[mask_combined])
                        del region_means[neighbor]
                        merged = True
                        break

                if merged:
                    break

        result = np.zeros_like(segmented)
        for label in region_means:
            mask = (labels == label)
            result[mask] = region_means[label]

        return result

    @requires_segmentation
    def get_segmented_image(self):
        """Obtiene la imagen segmentada."""
        segmented = np.zeros_like(self.image)
        self._fill_regions(self.root, segmented)
        return segmented

    @requires_segmentation
    def visualize_regions(self):
        """Dibuja los bordes de las regiones."""
        image_with_regions = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.draw_regions(self.root, image_with_regions)
        return image_with_regions

    @requires_segmentation
    def count_regions(self):
        """Cuenta el número de regiones."""
        return self.count_leafs(self.root)

# Título y descripción
st.title("🖼️ Segmentación Split and Merge")
st.write("Carga una imagen y ajusta los parámetros para ver las diferentes etapas de segmentación")

# Sidebar con controles
with st.sidebar:
    st.header("Parámetros")
    
    uploaded_file = st.file_uploader("Cargar imagen", type=['png', 'jpg', 'jpeg', 'bmp'])
    
    st.divider()
    
    min_size = st.slider("Min Size", min_value=4, max_value=64, value=16, step=4,
                        help="Tamaño mínimo de región")
    
    std_threshold = st.slider("Std Threshold", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                             help="Umbral de desviación estándar para homogeneidad")
    
    mean_threshold = st.slider("Mean Threshold", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                              help="Umbral de diferencia de medias para fusión")
    
    st.divider()
    
    filter_type = st.selectbox("Filtro de preprocesamiento", ['gaussian', 'median'])
    kernel_size = st.slider("Kernel Size", min_value=3, max_value=11, value=5, step=2)
    
    apply_preprocessing = st.checkbox("Aplicar preprocesamiento", value=True)
    apply_postprocessing = st.checkbox("Aplicar postprocesamiento", value=True)
    
    st.divider()
    
    # Botón para ejecutar segmentación
    run_segmentation = st.button("Iniciar Segmentación", type="primary", width="stretch")

# Procesamiento principal
if uploaded_file is not None:
    # Cargar imagen
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Convertir a escala de grises si es necesario
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np
    
    # Mostrar imagen original siempre
    st.subheader("Original")
    st.image(image_gray, width="stretch", clamp=True)
    
    # Ejecutar segmentación solo si se presionó el botón
    if run_segmentation:
        # Crear instancia del algoritmo
        sam = split_and_merge(image_gray.copy(), min_size, std_threshold, mean_threshold)
        
        # Preprocesamiento
        if apply_preprocessing:
            sam.pre_process(filter_type=filter_type, kernel_size=kernel_size)
        
        # Segmentación
        with st.spinner("Segmentando imagen..."):
            sam.segment()
        
        # Obtener imágenes en diferentes etapas
        segmented = sam.get_segmented_image()
        regions = sam.visualize_regions()
        num_regions = sam.count_regions()
        
        # Postprocesamiento
        if apply_postprocessing:
            postprocessed = sam.post_process()
        
        # Mostrar resultados
        st.success(f"Segmentación completada - {num_regions} regiones detectadas")
        
        # Layout de columnas
        col1, col2 = st.columns(2)
        
        with col1:
            if apply_preprocessing:
                st.subheader("Preprocesada")
                st.image(sam.image, width="stretch", clamp=True)
            
            st.subheader("Segmentada")
            st.image(segmented, width="stretch", clamp=True)
        
        with col2:
            if apply_postprocessing:
                st.subheader("Postprocesada")
                st.image(postprocessed, width="stretch", clamp=True)
            
            st.subheader("Visualización de Regiones")
            st.image(regions, width="stretch", clamp=True)
    
else:
    st.info("Carga una imagen desde el panel lateral para comenzar")