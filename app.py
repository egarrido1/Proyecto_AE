from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
import sqlite3 
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_required, current_user, UserMixin, logout_user, login_user 
from flask import flash
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
import re  # Expresiones regulares

# Importa el algoritmo de agrupamiento espectral, algoritmo K,means, matriz de Confusión y métricas
from algorithms.agrupamiento import algoritmo_agrupamiento_espectral # Importa el algoritmo de agrupamiento.py
from algorithms.agrupamiento import algoritmo_kmeans # Importa el algoritmo de agrupamiento.py
from utils import calcula_matriz_confusion, calcula_RMSE, calcula_precision_recall_fmeasure, calcula_porcentajes


# Configuración de la aplicación
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bd_usuarios_tesina5.db'
app.config['SECRET_KEY'] = 'tu_llave_secreta'
db = SQLAlchemy(app)

# Genera y asigna la clave secreta de 32 bytes en formato hexadecimal
app.secret_key = secrets.token_hex(32)

# Cargar la matriz de datos desde el archivo .pynb
X = np.load("matriz_ae.npy")
if np.isnan(X).any():
    raise ValueError("La matriz contiene valores NaN. Verifica los datos.")


# Define maximo número de intentos de autenticación
# datetime se utiliza para obtener la fecha y hora actuales (datetime.now()).
# timedelta se utiliza para representar una diferencia de tiempo, como por ejemplo 15 minutos.
LIMITE_INTENTOS = 3
TIEMPO_EXPIRACION = timedelta(minutes=15)

# Ruta para la pagina principal
@app.route('/', methods=['GET', 'POST'])

def home():
    if current_user.is_authenticated:
        return redirect(url_for('main'))  # Redirigir a la página del perfil
    if request.method == 'POST':
        nombre_usuario = request.form['username']
        password = request.form['password']
        
        # Buscar el usuario en la base de datos
        usuario = Usuario.query.filter_by(nombre_usuario=nombre_usuario).first()
        
        if usuario:
            ahora = datetime.now()
            
            # Verificar si los intentos han expirado
            if usuario.ultima_fecha_intento and ahora - usuario.ultima_fecha_intento > TIEMPO_EXPIRACION:
                usuario.intentos_fallidos = 0  # Reinicia intentos si ha pasado el tiempo
                db.session.commit()
            
            if usuario.intentos_fallidos >= LIMITE_INTENTOS:
                flash("Has alcanzado el límite de intentos. Intenta más tarde.")
                return render_template('index.html')

            # Validar contraseña
            if usuario.check_password(password):
                usuario.intentos_fallidos = 0
                usuario.ultima_fecha_intento = None  # Restablecer el intento fallido
                db.session.commit()
                session['username'] = nombre_usuario
                login_user(usuario)

                # Redirigir al perfil del usuario después de iniciar sesión correctamente
                return redirect(url_for('main'))  # Redirigir a la página del perfil
               
            else:
                usuario.intentos_fallidos += 1
                usuario.ultima_fecha_intento = ahora
                db.session.commit()
                flash('Nombre de usuario o contraseña incorrectos.')
        else:
            flash('Nombre de usuario o contraseña incorrectos.')

    return render_template('index.html')


# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/'  # Ruta a la que se redirige si el usuario no está autenticado

# Modelo de Usuario
class Usuario(db.Model, UserMixin):  # UserMixin añade funcionalidades de Flask-Login
    __tablename__ = 'usuarios'
    id = db.Column(db.Integer, primary_key=True)
    nombre_usuario = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    ultima_fecha_intento = db.Column(db.DateTime, nullable=True)
    intentos_fallidos = db.Column(db.Integer,default = 0, nullable=False)
    

    # Hashea y guarda la contraseña.
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

with app.app_context():
    db.create_all()
    
# Cargar usuario para Flask-Login a partir del ID
@login_manager.user_loader
def load_user(user_id):
    return Usuario.query.get(int(user_id))


# db.create_all()  # Crea la base de datos y las tablas

# Ruta para registrar un usuario
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Obtener los datos del formulario
        nombre_usuario = request.form['username']
        password = request.form['password']
        
        # Verificar si el nombre de usuario ya está registrado
        if Usuario.query.filter_by(nombre_usuario=nombre_usuario).first():
            flash('El nombre de usuario ya está registrado. Por favor, elige otro.', 'error')
            return redirect(url_for('register'))
        
        # Validar la contraseña
        if not validar_contraseña(password):
            flash('La contraseña debe tener al menos 8 caracteres, 1 mayuscula, 1 dígito y 1 carácter especial.', 'error')
            return render_template('register.html')
        
        # Crear un nuevo usuario
        nuevo_usuario = Usuario(nombre_usuario=nombre_usuario)
        nuevo_usuario.set_password(password)
        
        # Guardar el nuevo usuario en la base de datos
        db.session.add(nuevo_usuario)
        db.session.commit()
       
        flash('Registro exitoso. Por favor, inicia sesión.')
        return redirect(url_for('home'))

    return render_template('register.html')


# Ruta para el menú principal despues de iniciar sesion
@app.route('/main', methods=['GET', 'POST'])
@login_required
def main():
  #  if 'username' in session:
    if 'username' in session:
        fecha_y_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template('main.html', fecha_y_hora=fecha_y_hora)
    
    else:
        return redirect(url_for('main')) 
   # return redirect(url_for('login'))


CARACTERISTICAS_IDX = {
    0: 'Frecuencia fundamental',
    1: 'Ancho de banda',
    2: 'Centroide espectral',
    3: 'Entropía espectral',
}

@app.route('/perfil_usuario')
@login_required  # Asegúrarse de que el usuario esté autenticado
def perfil_usuario():
    
    return render_template('perfil_usuario.html', user_name=current_user.nombre_usuario)


# Ruta para recuperar contraseña
@app.route('/recuperar_contraseña', methods=['GET', 'POST'])
def recuperar_contraseña():
    if request.method == 'POST':
        nombre_usuario = request.form['username']
        nueva_contraseña = request.form['new_password']
        confirmar_contraseña = request.form['confirm_password']
        
        # Buscar el usuario en la base de datos
        usuario = Usuario.query.filter_by(nombre_usuario=nombre_usuario).first()
        
        if not usuario:
            flash('El usuario no existe.', 'error')
            return redirect(url_for('recuperar_contraseña'))
            
        if nueva_contraseña != confirmar_contraseña:
            flash('Las contraseñas no coinciden.', 'error')
            return redirect(url_for('recuperar_contraseña'))
        
         # Validar la complejidad de la nueva contraseña.
        if not validar_contraseña(nueva_contraseña):
            #flash('La nueva contraseña no cumple con los requisitos.', 'error')
            flash('La contraseña debe tener al menos 8 caracteres,1 letra mayuscula,  1 dígito y 1 carácter especial.', 'error')
            return redirect(url_for('recuperar_contraseña'))
        
        # Establecer la nueva contraseña
        usuario.set_password(nueva_contraseña)
        db.session.commit()
        

        flash('Contraseña cambiada con éxito. Por favor, inicia sesión', 'success')
        return redirect(url_for('home'))  # Redirigir al login después de cambiar la contraseña

    return render_template('recuperar_contraseña.html')



@app.route('/cambiar_contraseña', methods=['GET', 'POST'])
@login_required
def cambiar_contraseña():
    if request.method == 'POST':
        current_password = request.form.get('current_password', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # Verificar que se proporcionen todas las contraseñas requeridas
        if not current_password or not new_password or not confirm_password:
            flash('Todos los campos son obligatorios.', 'error')
            return redirect(url_for('cambiar_contraseña'))

        # Verificar si la contraseña actual proporcionada es incorrecta
        if not current_user.check_password(current_password):
            flash('La contraseña actual es incorrecta.', 'error')
            return redirect(url_for('cambiar_contraseña'))  # Salir temprano

        # Validar que la nueva contraseña y la confirmación coincidan
        if new_password != confirm_password:
            flash('La nueva contraseña y la confirmación no coinciden.', 'error')
            return redirect(url_for('cambiar_contraseña'))

        # Validar la complejidad de la nueva contraseña.
        if not validar_contraseña(new_password):
            #flash('La nueva contraseña no cumple con los requisitos.', 'error')
            flash('La contraseña debe tener al menos 8 caracteres,1 letra mayuscula,  1 dígito y 1 carácter especial.', 'error')
            return redirect(url_for('cambiar_contraseña'))

        # Cambiar la contraseña
        current_user.set_password(new_password)  # Asumimos que set_password maneja el hashing
        db.session.commit()  # Guardar cambios en la base de datos

        flash('Contraseña cambiada con éxito. Por favor inicia sesión', 'success')
        return redirect(url_for('perfil_usuario'))  # Redirigir al perfil del usuario

    return render_template('cambiar_contraseña.html')


def validar_contraseña(password):
    # Expresión regular para verificar los requisitos:
    # - Al menos 8 caracteres
    # - Al menos 1 dígito
    # - Al menos un carácter especial
    # - Al menos 1 letra mayúscula
    # - Al menos 1 letra minúscula
    regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[!@#$%^&*()_+~\-=\[\]{}|;:\'",.<>?/])(?=.{8,})'
    
    # Validar con re.match
    if re.match(regex, password):
        return True
    else:
        return False

# Función que calcula metricas
def calcula_metricas(X,etiquetas):
    # Obtener matriz de confusion
    # Llamar a la funciones que se encuentran en el archvo utils.py 
    confusionM = calcula_matriz_confusion(X[:,4],etiquetas)

    # Calcula RMSE (Root Mean Square Error)
    RMSE = calcula_RMSE(X[:,4],etiquetas)

     # Calcular precisión y recall y F-measure
    precision_0,recall_0,f_measure_0,precision_1,recall_1,f_measure_1 = calcula_precision_recall_fmeasure(confusionM,beta=1)

    # calcula porcentajes
    porcentajes = calcula_porcentajes(confusionM)
        
    return confusionM,RMSE,precision_0,recall_0,f_measure_0,precision_1,recall_1,f_measure_1,porcentajes


# Función para filtrar las características y ejecutar algoritmo de agrupamiento espectral
def ejecutar_agrupamiento_espectral(X, caracteristicas):
    print("Entrando en la ruta ejecutar_agrupamiento_espectral")
    print("Características seleccionadas:", caracteristicas)
     

    # Filtrar la matriz X según las columnas seleccionadas
    X_filtrado = X[:, caracteristicas]
    print("Matriz filtrada (X_filtrado):\n", X_filtrado)

    
    # llamar a mi algoritmo de agrupamiento espectral
    etiquetas = algoritmo_agrupamiento_espectral(X_filtrado,X)
    print("Etiquetas obtenidas por el algoritmo:", etiquetas)


    return etiquetas, X_filtrado
    

# Función para filtrar las características y ejecutar algoritmo K-means.
def ejecutar_k_means(X, caracteristicas):
    
    print("entra a ruta ejecutar_k_means")
     # Filtrar las columnas según las características seleccionadas
    #columnas_seleccionadas = [CARACTERISTICAS_IDX[car] for car in caracteristicas if car in CARACTERISTICAS_IDX]
    X_filtrado = X[:, caracteristicas]
    
    print("Tipo de X_filtrado:", type(X_filtrado))
    print("Dimensiones de X_filtrado:", getattr(X_filtrado, "shape", "No tiene forma"))
    print("Contenido de X_filtrado:\n", X_filtrado)
    
    # llamar al algoritmo K-means
    etiquetas, centros = algoritmo_kmeans(X_filtrado)
    
    print("despues del llamado del algoritmo")

    print("centros despues de calcular kmeans, el valor de centros: ",centros)

    return etiquetas, X_filtrado, centros


@app.route('/agrupamiento_espectral', methods=['GET', 'POST'])
@login_required 
def agrupamiento_espectral():
    
    if request.method == 'POST':
        # Obtener las características seleccionadas
        caracteristicas = request.form.getlist("opciones")  
        caracteristicas = [ int(caracteristica) for caracteristica in caracteristicas  ]
        
        # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            # Si se seleccionan menos de 2 características
             return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona al menos 2 características.", titulo="Agrupamiento Espectral")
            
        elif len(caracteristicas) > 3:
            # Si se seleccionan más de 3 características
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona un máximo de 3 características.", titulo="Agrupamiento Espectral")

        # Ejecutar el algoritmo con las características seleccionadas
        etiquetas, X_filtrado = ejecutar_agrupamiento_espectral(X, caracteristicas)

        # Antes de almacenar en la sesión
        
        X_filtrado = np.array(X_filtrado).reshape(-1, len(caracteristicas))

        # Validar que los datos sean correctos
        if not etiquetas.any() or not X_filtrado.any() or not caracteristicas:
            return render_template('error.html', mensaje="Datos insuficientes o no encontrados en la sesión.")


        # Calcula métricas
        confusionM,RMSE,precision_0,recall_0,f_measure_0,precision_1,recall_1,f_measure_1,porcentajes=calcula_metricas(X,etiquetas)
        
        print("confusionM dentro de agrupamiento_espectral", confusionM)
        
        # Comparar diagonales
        diagonal_principal = np.sum(np.diag(confusionM))
        diagonal_secundaria = np.sum(np.diag(np.fliplr(confusionM)))  # np.fliplr invierte las columnas para acceder a la diagonal secundaria

        if diagonal_principal < diagonal_secundaria:
            # Invertir etiquetas
            etiquetas = [1 - x for x in etiquetas]

            print("etiquetas despues de cambiar", etiquetas)

            # Recalcular métricas con las nuevas etiquetas
            confusionM, RMSE, precision_0, recall_0, f_measure_0, precision_1, recall_1, f_measure_1, porcentajes = calcula_metricas(X, etiquetas)
            print("confusionM despues de cambiar valores", confusionM)

        # Colores personalizados para cada etiqueta
        colores = []    # Inicializa la lista de colores

        # Asignar colores a cada etiqueta
        for etiqueta in etiquetas:
            if etiqueta == 0:
                colores.append('blue')   # Asignar azul para la etiqueta 0
            else:
                colores.append('green')  # Asignar verde para la etiqueta 1
                
        # Graficar los resultados dependiendo de las características seleccionadas
        imagen = ""
        
        if len(caracteristicas) == 2:
            # Graficar en 2D con subgráficas
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 9))

            # Subgráfica 1: Resultados del agrupamiento (con colores personalizados)
            sc1 = ax1.scatter(X_filtrado[:, 0], X_filtrado[:, 1], c=colores, s=90)  # Usar colores personalizados
            ax1.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=16)
            ax1.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=16)
            ax1.set_title("Agrupamiento Espectral 2D", fontsize=18)
            
            # Añadir la leyenda en la subgráfica 1 (Agrupamiento Espectral)
            ax1.scatter([], [], c='blue', label='Mujeres')  # Invisible para leyenda
            ax1.scatter([], [], c='green', label='Hombres')  # Invisible para leyenda
            ax1.legend(fontsize=14)

            # Subgráfica 2: Verdad fundamental
            sc2 = ax2.scatter(X_filtrado[:74, 0], X_filtrado[:74, 1], c='blue', label='Mujeres', s=90)
            sc3 = ax2.scatter(X_filtrado[74:, 0], X_filtrado[74:, 1], c='green', label='Hombres', s=90)
            ax2.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=16)
            ax2.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=16)
            ax2.set_title("Verdad Fundamental 2D", fontsize=18)
            ax2.legend(fontsize=14)

            # Guardar la imagen y cerrar
            imagen = 'static/agrupamiento_vs_verdad_2d.png'
            plt.savefig(imagen)
            plt.close()

        elif len(caracteristicas) == 3:
            # Crear una figura con dos subgráficas
            fig = plt.figure(figsize=(17, 9))  # Tamaño de la figura ajustado
            ax1 = fig.add_subplot(121, projection='3d')  # Subgráfico 1 (Agrupamiento Espectral)
            ax2 = fig.add_subplot(122, projection='3d')  # Subgráfico 2 (Verdad Fundamental)

            # Graficar los puntos del agrupamiento espectral en el primer subgráfico
            sc1 = ax1.scatter(X_filtrado[:, 0], X_filtrado[:, 1], X_filtrado[:, 2], c=colores, s=90, alpha=0.8)
            ax1.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=16)
            ax1.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=16)
            ax1.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]], fontsize=16)
            ax1.set_title("Agrupamiento espectral 3D", fontsize=18)
            
            # Añadir la leyenda en la subgráfica 1 (Agrupamiento Espectral)
            ax1.scatter([], [], c='blue', label='Mujeres', s=65, alpha=1)  # Alpha < 1 puntos con transparencia
            ax1.scatter([], [], c='green', label='Hombres', s=65, alpha=1)  # alpha =,0.6 puntos con transparencia, alpha=1 puntos sin transparencia
            ax1.legend(fontsize=16)
            
            # Graficar la verdad fundamental en el segundo subgráfico con colores específicos
            ax2.scatter(X_filtrado[:74, 0], X_filtrado[:74, 1], X_filtrado[:74, 2], c='blue', label='Mujeres', s=90, alpha=0.8)
            ax2.scatter(X_filtrado[74:, 0], X_filtrado[74:, 1], X_filtrado[74:, 2], c='green', label='Hombres', s=90, alpha=0.8)
            ax2.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=16)
            ax2.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=16)
            ax2.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]], fontsize=16)
            ax2.set_title("Verdad Fundamental 3D", fontsize=18)

            # Añadir leyenda a la verdad fundamental
            ax2.legend(fontsize=14)

            # Ajustar la disposición para evitar solapamiento entre subgráficas
            #plt.tight_layout()

            # Guardar la figura con dos subgráficas
            imagen = 'static/agrupamiento_vs_verdad_3d.png'
            plt.savefig(imagen)  # Guardar la imagen con mayor resolución
            plt.close()


        # Renderizar el template y pasarle la ruta de la imagen
        return render_template('resultadosAE.html', imagen=imagen, etiquetas=etiquetas, confusionM=confusionM, RMSE=RMSE, 
                            precision_0=precision_0, recall_0=recall_0, f_measure_0=f_measure_0, precision_1=precision_1,
                                recall_1=recall_1, f_measure_1=f_measure_1)
        

    # Si es una solicitud GET, muestra la página inicial
    #return render_template('agrupamiento_espectral.html')
    return render_template('selecciona_caracteristicas.html', titulo="Agrupamiento Espectral")

# Ruta para K-means
@app.route('/k_means', methods=['GET', 'POST'])
@login_required 
def k_means():
    if request.method == 'POST':

        # Obtener las características seleccionadas 
        caracteristicas = request.form.getlist('opciones')
        caracteristicas = [ int(caracteristica) for caracteristica in caracteristicas  ]

         # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            # Si se seleccionan menos de 2 características
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona al menos 2 características.", titulo="K-means")
        elif len(caracteristicas) > 3:
            # Si se seleccionan más de 3 características
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona un máximo de 3 características.", titulo="K-means")
         
        # Ejecutar K-means, con las características seleccionadas
        etiquetas, X_filtrado, centros = ejecutar_k_means(X, caracteristicas)

        X_filtrado = np.array(X_filtrado).reshape(-1, len(caracteristicas))

        # Validar que los datos sean correctos
        if not etiquetas.any() or not X_filtrado.any() or not caracteristicas:
            return render_template('error.html', mensaje="Datos insuficientes o no encontrados en la sesión.")

        # Calcula métricas
        confusionM,RMSE,precision_0,recall_0,f_measure_0,precision_1,recall_1,f_measure_1,porcentajes=calcula_metricas(X,etiquetas)
        
        # Comparar diagonales
        diagonal_principal = np.sum(np.diag(confusionM))
        diagonal_secundaria = np.sum(np.diag(np.fliplr(confusionM)))  # np.fliplr invierte las columnas para acceder a la diagonal secundaria

        if diagonal_principal < diagonal_secundaria:
            # Invertir etiquetas
            etiquetas = [1 - x for x in etiquetas]

            # Recalcular métricas con las nuevas etiquetas
            confusionM, RMSE, precision_0, recall_0, f_measure_0, precision_1, recall_1, f_measure_1, porcentajes = calcula_metricas(X, etiquetas)

        
        # Colores personalizados para cada etiqueta
        colores = []  # Inicializa la lista de colores

        # Asignar colores a cada etiqueta
        for etiqueta in etiquetas:
            if etiqueta == 0:
                colores.append('blue')   # Asignar azul para la etiqueta 0
            else:
                colores.append('green')  # Asignar verde para la etiqueta 1
                
        # Graficar los resultados dependiendo de las características seleccionadas
        imagen = ""
        
        if len(caracteristicas) == 2:
            # Graficar en 2D
            fig = plt.figure(figsize=(14, 8))

            # Subgráfica 1: Resultado de K-means
            ax1 = fig.add_subplot(121)
            sc1 = ax1.scatter(X_filtrado[:, 0], X_filtrado[:, 1], c=colores, alpha=0.8)
            ax1.scatter(centros[:, 0], centros[:, 1], c='red', marker='X', s=160, label="Centroides", alpha=1)
            ax1.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=14)
            ax1.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=14)
            ax1.set_title("K-means 2D", fontsize=18)

            # Despliega etiquetas para grafica k-means (parte superior derecha)
            ax1.legend(
                handles=[
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1'),
                    plt.Line2D([0], [0], marker='X', color='red', markersize=12, label='Centroides')
                ],
                fontsize=12, loc='upper right', ncol=1  # Leyenda en la parte superior derecha
            )

            # Subgráfica 2: Verdad Fundamental
            ax2 = fig.add_subplot(122)

            # Graficar verdad fundamental (colores específicos)
            ax2.scatter(X_filtrado[:74, 0], X_filtrado[:74, 1], c='blue', label='Mujeres', alpha=0.8)
            ax2.scatter(X_filtrado[74:, 0], X_filtrado[74:, 1], c='green', label='Hombres', alpha=0.8)

           
            # Despliega etiquetas para la verdad fundamental (parte superior derecha)
            ax2.legend(
                handles=[
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1')
                ],
                fontsize=12, loc='upper right', ncol=1  # Leyenda sin los centroides
            )

            ax2.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=14)
            ax2.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=14)
            ax2.set_title("Verdad Fundamental 2D", fontsize=18)

            # Guardar la imagen
            imagen = 'static/kmeans_vs_truth_2d.png'
            plt.savefig(imagen)
            plt.close()

        elif len(caracteristicas) == 3:
            # Graficar en 3D
            fig = plt.figure(figsize=(14, 8))

            # Subgráfica 1: Resultado de K-means
            ax1 = fig.add_subplot(121, projection='3d')
            sc1 = ax1.scatter(X_filtrado[:, 0], X_filtrado[:, 1], X_filtrado[:, 2], c=colores, alpha=0.8)
            ax1.scatter(centros[:, 0], centros[:, 1], centros[:, 2], c='red', marker='X', s=160, label="Centroides", alpha=1)

            ax1.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=14)
            ax1.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=14)
            ax1.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]], fontsize=14)
            ax1.set_title("K-means 3D", fontsize=18)

            # Despliega etiquetas para grafica K-means (parte superior derecha)
            ax1.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1'),
                plt.Line2D([0], [0], marker='X', color='red', markersize=12, label='Centroides')], 
                fontsize=12, loc='upper right')  # Leyenda de K-means

            # Subgráfica 2: Verdad Fundamental
            ax2 = fig.add_subplot(122, projection='3d')

            # Graficar verdad fundamental (colores específicos)
            ax2.scatter(X_filtrado[:74, 0], X_filtrado[:74, 1], X_filtrado[:74, 2], c='blue', label='Mujeres', alpha=0.8)
            ax2.scatter(X_filtrado[74:, 0], X_filtrado[74:, 1], X_filtrado[74:, 2], c='green', label='Hombres', alpha=0.8)

           # Despliega etiquetas para grafica verdad fundamental (parte superior derecha)
            ax2.legend(
                 handles=[
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1')
                ],
                fontsize=12, loc='upper right', ncol=1  # Leyenda sin los centroides
            )

            ax2.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]], fontsize=14)
            ax2.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]], fontsize=14)
            ax2.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]], fontsize=14)
            ax2.set_title("Verdad Fundamental 3D", fontsize=18)
            
            # Guardar la imagen
            imagen = 'static/kmeans_vs_truth_3d.png'
            plt.savefig(imagen)
            plt.close()



        # Renderizar el template y pasarle la ruta de la imagen
        return render_template('resultadosKM.html', imagen=imagen, etiquetas=etiquetas, confusionM=confusionM, RMSE=RMSE, 
                            precision_0=precision_0, recall_0=recall_0, f_measure_0=f_measure_0, precision_1=precision_1,
                                recall_1=recall_1, f_measure_1=f_measure_1)

        
    # Si es una solicitud GET, muestra el formulario
    return render_template('selecciona_caracteristicas.html', titulo="k-means")
   
@app.route("/compara_algoritmos", methods=['GET', 'POST'])
def compara_algoritmos():
    if request.method == 'POST':
        # Obtener las características seleccionadas
        caracteristicas = request.form.getlist('opciones')
        caracteristicas = [int(caracteristica) for caracteristica in caracteristicas]

        # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona al menos 2 características.", 
                                   titulo="Compara algoritmos")
        elif len(caracteristicas) > 3:
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona un máximo de 3 características.", 
                                   titulo="Compara algoritmos")

        # Ejecutar algoritmos
        etiquetas_ae, X_filtrado_ae = ejecutar_agrupamiento_espectral(X, caracteristicas)
        confusionM_ae,RMSE_ae,precision_ae_0,recall_ae_0,f_measure_ae_0,precision_ae_1,recall_ae_1,f_measure_ae_1,porcentajes_ae=calcula_metricas(X,etiquetas_ae)
        etiquetas_km, X_filtrado_km, centros = ejecutar_k_means(X, caracteristicas)
        confusionM_km,RMSE_km,precision_km_0,recall_km_0,f_measure_km_0,precision_km_1,recall_km_1,f_measure_km_1,porcentajes_km=calcula_metricas(X,etiquetas_km)
        
        # Comparar diagonales agrupamiento espectral, para desplegar graficas solo cuando la matriz de confusion esta en el formato correcto
        diagonal_principal_ae = np.sum(np.diag(confusionM_ae))
        diagonal_secundaria_ae = np.sum(np.diag(np.fliplr(confusionM_ae)))  # np.fliplr invierte las columnas para acceder a la diagonal secundaria

        if diagonal_principal_ae < diagonal_secundaria_ae:
            # Invertir etiquetas
            etiquetas_ae = [1 - x for x in etiquetas_ae]

            # Recalcular métricas con las nuevas etiquetas
            confusionM_ae,RMSE_ae,precision_ae_0,recall_ae_0,f_measure_ae_0,precision_ae_1,recall_ae_1,f_measure_ae_1,porcentajes_ae=calcula_metricas(X,etiquetas_ae)
        


        # Comparar diagonales k-means
        diagonal_principal_km = np.sum(np.diag(confusionM_km))
        diagonal_secundaria_km = np.sum(np.diag(np.fliplr(confusionM_km)))  # np.fliplr invierte las columnas para acceder a la diagonal secundaria

        if diagonal_principal_km < diagonal_secundaria_km:
            # Invertir etiquetas
            etiquetas_km = [1 - x for x in etiquetas_km]

            # Recalcular métricas con las nuevas etiquetas
            confusionM_km,RMSE_km,precision_km_0,recall_km_0,f_measure_km_0,precision_km_1,recall_km_1,f_measure_km_1,porcentajes_km=calcula_metricas(X,etiquetas_km)

        # Colores personalizados para cada etiqueta, para Agrupamiento espectral
        colores_ae = []    # Inicializa la lista de colores

        # Asignar colores a cada etiqueta
        for etiqueta in etiquetas_ae:
            if etiqueta == 0:
                colores_ae.append('blue')   # Asignar azul para la etiqueta 0
            else:
                colores_ae.append('green')  # Asignar verde para la etiqueta 1
                
                
         # Colores personalizados para cada etiqueta, para k-means
        colores_km = []    # Inicializa la lista de colores

        # Asignar colores a cada etiqueta
        for etiqueta in etiquetas_km:
            if etiqueta == 0:
                colores_km.append('blue')   # Asignar azul para la etiqueta 0
            else:
                colores_km.append('green')  # Asignar verde para la etiqueta 1
                
        # Crear subgráficas
        if len(caracteristicas) == 2:
            # Gráficas 2D
            fig, axes = plt.subplots(1, 2, figsize=(25, 10))
            
            # Agrupamiento Espectral
            axes[0].scatter(X_filtrado_ae[:, 0], X_filtrado_ae[:, 1], c=colores_ae, s=65)
            axes[0].set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]])  # Usar el diccionario para los nombres
            axes[0].set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]])  # Usar el diccionario para los nombres
            axes[0].set_title('Agrupamiento Espectral')
            
            # Despliega leyenda en la parte superior derecha
            axes[0].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1')],
               loc='upper right', title='Leyenda')
            
            # K-Means
            axes[1].scatter(X_filtrado_km[:, 0], X_filtrado_km[:, 1], c=colores_km, s=65)
            plt.scatter(centros[:, 0], centros[:, 1], c='red', marker='X', s=150, label="Centroides", alpha=1)  # Centroides
            axes[1].set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]])  # Usar el diccionario para los nombres
            axes[1].set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]])  # Usar el diccionario para los nombres
            axes[1].set_title('K-Means')
            
            # Despliega leyenda en la parte superior derecha
            axes[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1'),
                        plt.Line2D([0], [0], marker='X', color='red', markersize=10, label='Centroides')],
               loc='upper right', title='Leyenda')
            
            
            # Guardar imagen combinada
            imagen_combinada = 'static/comparaAlgoritmos_2d.png'
            plt.savefig(imagen_combinada)
            plt.close()

        elif len(caracteristicas) == 3:
            # Gráficas 3D
            fig = plt.figure(figsize=(14, 6))
            
            # Agrupamiento Espectral
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(X_filtrado_ae[:, 0], X_filtrado_ae[:, 1], X_filtrado_ae[:, 2], c=colores_ae) # Usar colores predeterminados
            ax1.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]])  # Usar el diccionario para los nombres
            ax1.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]])  # Usar el diccionario para los nombres
            ax1.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]])  # Usar el diccionario para los nombres
            ax1.set_title('Agrupamiento Espectral')
            
            # Despliega leyenda en la parte superior derecha.
            ax1.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1')],
                loc='upper right', title='Leyenda')
            
            # K-Means
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(X_filtrado_km[:, 0], X_filtrado_km[:, 1], X_filtrado_km[:, 2], c=colores_km) # Usar colores predeterminados
            ax2.scatter(centros[:, 0], centros[:, 1], centros[:, 2], c='red', marker='X', s=150, label="Centroides", alpha=1)  # Centroides
            ax2.set_xlabel(CARACTERISTICAS_IDX[caracteristicas[0]])  # Usar el diccionario para los nombres
            ax2.set_ylabel(CARACTERISTICAS_IDX[caracteristicas[1]])  # Usar el diccionario para los nombres
            ax2.set_zlabel(CARACTERISTICAS_IDX[caracteristicas[2]])  # Usar el diccionario para los nombres
            ax2.set_title('K-Means')
            
            # Despliega leyenda en la parte superior derecha.
            ax2.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Etiqueta 0'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Etiqueta 1'),
                    plt.Line2D([0], [0], marker='X', color='red', markersize=10, label='Centroides')],
                loc='upper right', title='Leyenda')
            
            # Guardar imagen combinada
            imagen_combinada = 'static/comparaAlgoritmos_3d.png'
            plt.savefig(imagen_combinada)
            plt.close()

        # Renderizar la página de comparación
        return render_template('resultados_comparacion.html', 
                               imagen_combinada=imagen_combinada, 
                               caracteristicas=caracteristicas,confusionM_ae=confusionM_ae,RMSE_ae=RMSE_ae,precision_ae_0=precision_ae_0,
                               recall_ae_0=recall_ae_0,f_measure_ae_0=f_measure_ae_0,precision_ae_1=precision_ae_1,recall_ae_1=recall_ae_1,
                               f_measure_ae_1=f_measure_ae_1,confusionM_km=confusionM_km,RMSE_km=RMSE_km,precision_km_0=precision_km_0,
                               recall_km_0=recall_km_0,f_measure_km_0=f_measure_km_0,precision_km_1=precision_km_1,recall_km_1=recall_km_1,
                               f_measure_km_1=f_measure_km_1)
        
    return render_template('selecciona_caracteristicas.html', titulo="Compara algoritmos")



@app.route('/logout')
#@login_required  # Asegura que solo los usuarios autenticados puedan cerrar sesión
def logout():
    logout_user()  # Cierra la sesión del usuario
    return redirect(url_for('home'))  # Redirige al login


if __name__ == '__main__':
    app.run(debug=False)
    