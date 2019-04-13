import numpy as np                                     # Manejo arreglos.
from pybalu.io import imread                           # Lectura imágenes.
from sklearn.neighbors import KNeighborsClassifier     # Modelo KNN
from pybalu.feature_extraction import (hugeo_features,  # Extracción features.
                                       flusser_features, fourier_features)
from pybalu.feature_selection import sfs               # Selección features.
from pybalu.performance_eval import performance  # Cómputo accuracy.
