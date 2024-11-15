Proyecto para la implementación del método de agrupamiento espectral para segmentación binaria de voces humanas.


Estructura del proyecto y descripción de archivos

Proyecto_AE/
├── app.py                  	     # Archivo principal de Flask (Punto de entrada de la aplicación)
├── config.py                   # Archivo de configuraciones (como bases de datos)
├── requirements.txt        # Lista de dependencias del proyecto (Librerías)
├── templates/               # Carpeta para plantillas HTML
│   ├── register.html      # Plantilla de registro
│   ├	── login.html           # Plantilla de inicio de sesión
│   ├── main.html          # Plantilla principal
│   ├── perfil_usuario.html       # Plantilla para visualizar el perfil del usuario
│   ├── recuperar_contraseña.html       # Plantilla para recuperar contraseña
│   ├── cambiar_contraseña.html       # Plantilla para cambiar contraseña
│   ├── agrupamiento_espectral.html       # Plantilla principal del alg. Agrup.espectral.
│   ├── kmeans.html       # Plantilla para visualizar los resultados
│   ├── selecciona_caracteristicas.html       # Plantilla para visualizar los resultados
│   ├── comparacion.html   # Plantilla para la comparación de algoritmos
│   ├── resultados.html   # Plantilla para desplegar resultados de un algoritmo
│   └── resultados_comparacion.html   # Plantilla desplegar resultados de dos algtms.
├── static/                 	   # Carpeta para archivos estáticos (CSS, JS, imágenes)
│   ├── styles.css           # Archivo CSS para estilos
│   └── script.js                # Archivo JavaScript (si es necesario)
├── data/ 		   # Carpeta para archivos de datos
 │ ├── matriz_datos.ipynb     # Archivo con la matriz de datos (características)
└── algorithms/              # Carpeta para la lógica de los algoritmos
    ├── agrupamiento.py      # Implementación del algoritmo de agrupamiento espectral
    └── kmeans.py           # Implementación del algoritmo K-means

lenguage de programación: Python

Librerias: Numpy, Matplotlib, Librosa, flask_sqlalchemy,  werkzeug.security 

