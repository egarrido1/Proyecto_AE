import numpy as np
#!/usr/bin/env python
# coding: utf-8

# ## Archivo utils.py contiene los calculos para la matriz de confusión y las diferentes nétricas.



# This function computes the Root Mean Square Error (RMSE).
def calcula_RMSE(C,eC):
    return(np.sqrt((1/C.size)*(np.linalg.norm(C-eC)**2)))

# Matriz de confusion
def calcula_matriz_confusion(C,eC):
    
    CM = np.zeros((2,2))
    
    for i in range(C.size):
        
        if (C[i] == 0.):
            if (C[i] == eC[i]):
                CM[0,0] += 1
            if (C[i] != eC[i]):
                CM[0,1] += 1
        
        if (C[i] == 1.):
            if (C[i] == eC[i]):
                CM[1,1] += 1
            if (C[i] != eC[i]):
                CM[1,0] += 1
    
    return CM

# Función porcentajes 
def calcula_porcentajes(cm):
    p = np.zeros(2)
    p[0] = (np.sum(np.diag(cm))/np.sum(cm))*100
    p[1] = (np.sum(np.diag(np.rot90(cm)))/np.sum(cm))*100
    return p

# Función para calcular precisión y recall y F-measure.
def calcula_precision_recall_fmeasure(cm,beta=1):
    tp = cm[0, 0]  # Verdaderos Positivos
    fn = cm[0, 1]  # Falsos Negativos
    tn = cm[1, 1]  # Verdaderos Negativos
    fp = cm[1, 0]  # Falsos positivos
   

    # Calculando precisión y recall para clase 0
    precision_0 = tp / (tp + fp) 
    recall_0 = tp / (tp + fn) 
    
    # Calculando precisión y recall para clase 1
    precision_1 = tn / (tn + fn) 
    recall_1 = tn / (tn + fp) 

    # Calculando F-measure para clase 0
    f_measure_0 = (1 + beta**2) * (precision_0 * recall_0) / (beta**2 * precision_0 + recall_0)

   
    # Calculando F-measure para clase 1
    f_measure_1 = (1 + beta**2) * (precision_1 * recall_1) / (beta**2 * precision_1 + recall_1)


    return precision_0, recall_0, f_measure_0, precision_1, recall_1, f_measure_1


# %%
