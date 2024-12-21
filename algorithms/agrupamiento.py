import numpy as np
from sklearn.cluster import KMeans
#!/usr/bin/env python
# coding: utf-8

# ## Algoritmo de agrupamiento espectral


# Función que normaliza un vector
def normalizarenglones(eg_vector):
    W = np.zeros(eg_vector.shape)
    norms = np.linalg.norm(eg_vector, axis=1)
    for i in range(eg_vector.shape[0]):
        if norms[i] > 0:  # Verifica que la norma no sea cero
            W[i, :] = eg_vector[i, :] / norms[i]
    return W


# Algritmo de agrupamiento espectral permite transformar una matriz  
# de datos al espacio espectral, para segmentar los datos. 
# Paso 1: Construir la matriz de similitud
# Paso 2: Construir la matriz laplaciana Normalizada
# Paso 3: Calcular los valores y vectores propios de la matriz laplaciana
# Paso 4: Reducción de dimensionalidad
# Paso 5: Aplicar algoritmo de agrupamiento K-means
def algoritmo_agrupamiento_espectral(X_matriz,X):
    # Se convierten las unidades de Hz. a kHz.
    X_matriz = X_matriz/1000

    #Normalizamos cada columna de los datos entre 0 y 1.
    maxCols = np.max(X_matriz,axis=0)
    for i in range(X_matriz.shape[1]):
        X_matriz[:,i] /= maxCols[i]
    
    # Calcular el cuadrado de cada elemento en la matriz
    X_cuadrada = X_matriz**2

    # Suma de los cuadrados a lo largo de cada fila
    suma_cuadrados = np.sum(X_cuadrada, axis=1) 

    # Transforma el vector fila a una matriz columna
    suma_cuadrada_col = suma_cuadrados.reshape(-1, 1)

    # Producto punto entre las matrices A, B, se obtiene multiplicando las 
    # filas de A por las columnas de B
    prod_punto = np.dot(X_matriz, X_matriz.T)

    # Calcular la distancia euclidiana al cuadrado
    distancia_cuadrado = suma_cuadrada_col + suma_cuadrados - 2 * prod_punto

    # elimina ceros numéricos en al diagonal
    m = distancia_cuadrado.shape[0]
    distancia_cuadrado[range(m), range(m)] = 0.

    # matriz de distancias (disilmilaridades)
    distancia = np.sqrt(distancia_cuadrado)

    distancia = distancia/np.max(distancia)

    '''
    # El quinto vecino
    r=5
    distancia_ordenada_por_filas=np.sort(distancia)
    sigma=distancia_ordenada_por_filas[:,r]
    '''

    # Calcular la matriz de similitud utilizando el kernel gaussiano
    sigma=np.ones(m)
    S = np.zeros((m,m))

    # Agregué esta linea
    epsilon = 1e-10  # Pequeño valor para evitar log de ceros
    
    for i in range(m):
        for j in range(i,m):
            if (i == j):
                S[i,j] = 0.
            else:
                S[i,j] = np.exp(-(distancia[i,j]**2)/(2*(sigma[i]**2)))
                
                S[j,i] = S[i,j]

    # Calcular la matriz de grados
    D = np.diag(np.sum(S, axis=1))
    
    # Extrae la Diagonal de D (matriz de grados)
    diag_D = np.diag(D)

    # Nos Asegúramos que no haya ceros en la diagonal de D
    diag_D.setflags(write=True)  # Permitir escritura en el array
    diag_D[diag_D == 0] = epsilon  # Reemplaza los ceros por epsilon
    
    Inv_raicescuadradas_D = np.diag(1.0 / np.sqrt(diag_D))
    
    # Calcular la matriz laplaciana normalizada simetrica
    L_sym = np.identity(S.shape[0]) - np.dot(np.dot(Inv_raicescuadradas_D,S), Inv_raicescuadradas_D)
    
    print(L_sym)
    # Calcular los Valores y Vectores propios
    valores_propios, vectores_propios = np.linalg.eig(L_sym)
    # Definir el número de clusters
    k = 2
    k_vectores_propios = vectores_propios[:, 0:k]
    
    # Renormaliza los vectores propios
    k_evectores_filasnorm = normalizarenglones(k_vectores_propios)
    
    # Crear el modelo k-means con k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(k_evectores_filasnorm)
    etiquetas = kmeans.labels_
    centros = kmeans.cluster_centers_

    return etiquetas 

# Algritmo K-means 
def algoritmo_kmeans(X_matriz):
    
    print("entra a algoritmo kmeans")
    # Definir el número de clusters
    k = 2
    
    # Crear el modelo k-means con k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++').fit(X_matriz)
    etiquetas = kmeans.labels_
    centros = kmeans.cluster_centers_
    
    print("centros despues de calcularlos: ",centros)
    print("etiquetas despues de calcularlas: ",etiquetas)
    return etiquetas,centros


# %%
