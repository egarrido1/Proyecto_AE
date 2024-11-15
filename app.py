from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import secrets
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_required, current_user, UserMixin
from datetime import datetime, timedelta
import re  # Expresiones regulares

# Importa el algoritmo de agrupamiento espectral, algoritmo K,means, matriz de Confusión y métricas
from algorithms.agrupamiento import algoritmo_agrupamiento_espectral # Importa el algoritmo de agrupamiento.py
from algorithms.agrupamiento import algoritmo_kmeans # Importa el algoritmo de agrupamiento.py
from utils import calcula_matriz_confusion, calcula_RMSE, calcula_precision_recall_fmeasure, calcula_porcentajes


# Configuración de la aplicación
app = Flask(__name__)

# Genera y asigna la clave secreta de 32 bytes en formato hexadecimal
app.secret_key = secrets.token_hex(32)

# Cargar la matriz de datos desde el archivo .pynb
X = np.load("matriz_ae.npy")

# Ruta para la pagina principal
@app.route('/')
def home():
    return render_template('login.html')
    

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///usuarios.db'
app.config['SECRET_KEY'] = 'tu_llave_secreta'
db = SQLAlchemy(app)

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # Ruta a la que se redirige si el usuario no está autenticado

# Modelo de Usuario
class Usuario(db.Model, UserMixin):  # UserMixin añade funcionalidades de Flask-Login
    __tablename__ = 'usuarios'
    id_usuario = db.Column(db.Integer, primary_key=True)
    nombre_usuario = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

    # Hashea y guarda la contraseña.
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

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
            flash('El nombre de usuario ya está registrado. Por favor, elige otro.')
            return redirect(url_for('register'))
        
        # Validar la contraseña
        if not validar_contraseña(password):
            flash('La contraseña debe tener al menos 8 caracteres, 1 dígito y 1 carácter especial.', 'error')
            return render_template('register.html')
        
        # Crear un nuevo usuario
        nuevo_usuario = Usuario(nombre_usuario=nombre_usuario)
        nuevo_usuario.set_password(password)
        db.session.add(nuevo_usuario)
        db.session.commit()
        
        flash('Registro exitoso. Por favor, inicia sesión.')
        return redirect(url_for('login'))

    return render_template('register.html')




# Ruta para el menú principal despues de iniciar sesion
@app.route('/main')
def main():
  #  if 'username' in session:
         # Obtenemos la fecha y hora actual
    session={
        'username':'edith01'
    }
    fecha_y_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('main.html', user_name=session['username'], fecha_y_hora=fecha_y_hora)
   # return redirect(url_for('login'))

# Diccionario para mapear nombres de características a índices de columnas en la matriz X
CARACTERISTICAS_IDX = {
    'frecuencia': 0,
    'Ancho_banda': 1,
    'Centroide': 2,
    'Entropia': 3
}


# Define maximo número de intentos de autenticación
# datetime se utiliza para obtener la fecha y hora actuales (datetime.now()).
# timedelta se utiliza para representar una diferencia de tiempo, como por ejemplo 15 minutos.
LIMITE_INTENTOS = 3
TIEMPO_EXPIRACION = timedelta(minutes=15)

@app.route('/login', methods=['GET', 'POST'])
def login():
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
                return render_template('login.html')

            # Validar contraseña
            if usuario.check_password(password):
                usuario.intentos_fallidos = 0
                usuario.ultima_fecha_intento = None  # Restablecer el intento fallido
                db.session.commit()
                session['username'] = nombre_usuario

                # Redirigir al perfil del usuario después de iniciar sesión correctamente
                return redirect(url_for('perfil_usuario'))  # Redirigir a la página del perfil
               
            else:
                usuario.intentos_fallidos += 1
                usuario.ultima_fecha_intento = ahora
                db.session.commit()
                flash('Nombre de usuario o contraseña incorrectos.')
        else:
            flash('Nombre de usuario o contraseña incorrectos.')

    return render_template('login.html')


@app.route('/perfil_usuario')
@login_required  # Asegúrate de que el usuario esté autenticado
def perfil_usuario():
    return render_template('perfil_usuario.html')


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
            flash("El usuario no existe.")
            return redirect(url_for('recuperar_contraseña'))
            
        if nueva_contraseña != confirmar_contraseña:
            flash("Las contraseñas no coinciden.")
            return redirect(url_for('recuperar_contraseña'))
        
        # Establecer la nueva contraseña
        usuario.set_password(nueva_contraseña)
        db.session.commit()
        
        flash("Contraseña cambiada con éxito.")
        return redirect(url_for('login'))  # Redirigir al login después de cambiar la contraseña

    return render_template('recuperar_contraseña.html')


# Ruta para cambiar contraseña
@app.route('/cambiar_contraseña', methods=['GET', 'POST'])
@login_required  # Asegúrate de que el usuario esté autenticado
def cambiar_contraseña():
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        
        # Verificar que la contraseña actual es correcta
        if not current_user.check_password(current_password):
            flash('La contraseña actual es incorrecta.', 'error')
            return redirect(url_for('cambiar_contraseña'))
        
        # Llamar a la función de validación para la nueva contraseña
        if not validar_contraseña(new_password):
            flash('La nueva contraseña debe tener al menos 8 caracteres, 1 dígito y 1 carácter especial.', 'error')
            return redirect(url_for('cambiar_contraseña'))
        
        # Establecer la nueva contraseña
        current_user.set_password(new_password)
        db.session.commit()
        
        flash('Contraseña cambiada con éxito.', 'success')
        return redirect(url_for('perfil_usuario'))  # Redirigir al  despperfil después de cambiar la contraseña

    return render_template('cambiar_contraseña.html')

# Función para validar contraseña
def validar_contraseña(password):
    # Expresión regular para verificar los requisitos:
    # - Al menos 8 caracteres
    # - Al menos 1 dígito
    # - Al menos un carácter especial
    regex = r'^(?=.*[0-9])(?=.*[!@#$%^&*])(?=.{8,})'
    
    # Validar con re.match
    if re.match(patron, password):
        return True
    else:
        return False



# Función para filtrar las características y ejecutar algoritmo de agrupamiento espectral
def ejecutar_agrupamiento_espectral(X, caracteristicas):
    
    # Filtrar las columnas según las características seleccionadas
    columnas_seleccionadas = [CARACTERISTICAS_IDX[car] for car in caracteristicas if car in CARACTERISTICAS_IDX]
    X_filtrado = X[:, columnas_seleccionadas]
    
    # llamar a mi algoritmo de agrupamiento espectral
    etiquetas = algoritmo_agrupamiento_espectral(X_filtrado,X)

    return etiquetas, X_filtrado
    

# Función para filtrar las características y ejecutar algoritmo K-means.
def ejecutar_k_means(X, caracteristicas):
    
     # Filtrar las columnas según las características seleccionadas
    columnas_seleccionadas = [CARACTERISTICAS_IDX[car] for car in caracteristicas if car in CARACTERISTICAS_IDX]
    X_filtrado = X[:, columnas_seleccionadas]
    
    # llamar al algoritmo K-means
    etiquetas,centros = algoritmo_k_means(X_filtrado,X)

    return etiquetas, centros


@app.route('/agrupamiento_espectral', methods=['GET', 'POST'])
def agrupamiento_espectral():
    if request.method == 'POST':
        # Obtener las características seleccionadas
        caracteristicas = request.form.getlist("caracteristicas")
        
        # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            # Si se seleccionan menos de 2 características
            return render_template('agrupamiento_espectral.html', 
                                   error="Por favor, selecciona al menos 2 características.")
        elif len(caracteristicas) > 3:
            # Si se seleccionan más de 3 características
            return render_template('agrupamiento_espectral.html', 
                                   error="Por favor, selecciona un máximo de 3 características.")

        # Ejecutar el algoritmo con las características seleccionadas
        etiquetas, X_filtrado = ejecutar_agrupamiento_espectral(X, caracteristicas)

        # Almacenar los resultados en la sesión
        session['etiquetas'] = etiquetas.tolist()
        session['X_filtrado'] = X_filtrado.tolist()
        session['caracteristicas'] = caracteristicas

        # Redirigir a la ruta resultados
        return redirect(url_for('prepara_resultados'))

    # Si es una solicitud GET, muestra la página inicial
    return render_template('agrupamiento_espectral.html')


# Ruta para K-means
@app.route('/k_means', methods=['GET', 'POST'])
def k_means():
    if request.method == 'POST':

        # Renderizar la página inicial para el algoritmo de K-means
        return render_template('kmeans.html')

        # Obtener las características seleccionadas 
        caracteristicas = request.form.getlist('caracteristicas')

         # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            # Si se seleccionan menos de 2 características
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona al menos 2 características.")
        elif len(caracteristicas) > 3:
            # Si se seleccionan más de 3 características
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona un máximo de 3 características.")
            
        # Ejecutar K-means, con las características seleccionadas
        etiquetas, X_filtrado = ejecutar_k_means(X, caracteristicas)

        # Almacenar los resultados en la sesión
        session['etiquetas'] = etiquetas.tolist()
        session['X_filtrado'] = X_filtrado.tolist()
        session['caracteristicas'] = caracteristicas

        # Redirigir a la ruta resultados
        return redirect(url_for('prepara_resultados'))
        
        # Pasar los resultados a la plantilla
        #return render_template('resultados.html', etiquetas=etiquetas, X_filtrado=X_filtrado, caracteristicas=caracteristicas)
       
    # Si es una solicitud GET, muestra el formulario
    return render_template('selecciona_caracteristicas.html')
   
# Ruta para comparación de algoritmos
@app.route("/comparacion", methods=['GET', 'POST'])
def comparacion():
    if request.method == 'POST':
        # Obtener las características seleccionadas
        caracteristicas = request.form.getlist('caracteristicas')

        # Validación de las características seleccionadas
        if len(caracteristicas) < 2:
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona al menos 2 características.")
        elif len(caracteristicas) > 3:
            return render_template('selecciona_caracteristicas.html', 
                                   error="Por favor, selecciona un máximo de 3 características.")

        # Ejecutar agrupamiento espectral
        etiquetas_ae, X_filtrado_ae = ejecutar_agrupamiento_espectral(X, caracteristicas)

        # Ejecutar k-means
        etiquetas_km, X_filtrado_km = ejecutar_k_means(X, caracteristicas)

        # Graficar resultados de agrupamiento espectral
        plt.figure(figsize=(8, 6))
        if len(caracteristicas) == 2:
            plt.scatter(X_filtrado_ae[:, 0], X_filtrado_ae[:, 1], c=etiquetas_ae, cmap='viridis')
            plt.xlabel(caracteristicas[0])
            plt.ylabel(caracteristicas[1])
            plt.title('Agrupamiento Espectral')
            imagen_ae = 'static/agrupamiento_espectral.png'
            plt.savefig(imagen_ae)
            plt.close()
        elif len(caracteristicas) == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_filtrado_ae[:, 0], X_filtrado_ae[:, 1], X_filtrado_ae[:, 2], c=etiquetas_ae, cmap='viridis')
            ax.set_xlabel(caracteristicas[0])
            ax.set_ylabel(caracteristicas[1])
            ax.set_zlabel(caracteristicas[2])
            ax.set_title('Agrupamiento Espectral')
            imagen_ae = 'static/agrupamiento_espectral_3d.png'
            plt.savefig(imagen_ae)
            plt.close()

        # Graficar resultados de k-means
        plt.figure(figsize=(8, 6))
        if len(caracteristicas) == 2:
            plt.scatter(X_filtrado_km[:, 0], X_filtrado_km[:, 1], c=etiquetas_km, cmap='viridis')
            plt.xlabel(caracteristicas[0])
            plt.ylabel(caracteristicas[1])
            plt.title('K-Means')
            imagen_km = 'static/kmeans.png'
            plt.savefig(imagen_km)
            plt.close()
        elif len(caracteristicas) == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_filtrado_km[:, 0], X_filtrado_km[:, 1], X_filtrado_km[:, 2], c=etiquetas_km, cmap='viridis')
            ax.set_xlabel(caracteristicas[0])
            ax.set_ylabel(caracteristicas[1])
            ax.set_zlabel(caracteristicas[2])
            ax.set_title('K-Means')
            imagen_km = 'static/kmeans_3d.png'
            plt.savefig(imagen_km)
            plt.close()

        # Renderizar la página de comparación
        return render_template('resultados_comparacion.html', 
                               imagen_ae=imagen_ae, 
                               imagen_km=imagen_km, 
                               caracteristicas=caracteristicas)

    # Si es una solicitud GET, muestra el formulario
    return render_template('selecciona_caracteristicas.html')


# Ruta para preparar resultados como metricas y graficas de acuerdo a las características elegidas
@app.route('/prepara_resultados')
def prepara_resultados():

    # Recuperar los datos de la sesión
    caracteristicas = session.get('caracteristicas', [])
    etiquetas = np.array(session.get('etiquetas', []))
    X_filtrado = np.array(session.get('X_filtrado', [])).reshape(-1, len(caracteristicas))

    # Validar que los datos sean correctos
    if not etiquetas.any() or not X_filtrado.any() or not caracteristicas:
        return render_template('error.html', mensaje="Datos insuficientes o no encontrados en la sesión.")

    # Obtener matriz de confusion
    # Llamar a la funciones que se encuentran en el archvo utils.py 
    confusionM = calcula_matriz_confusion(X[:,4],etiquetas)

    # Calcula RMSE (Root Mean Square Error)
    RMSE = calcula_RMSE(X[:,4],etiquetas)

    # Calcular precisión y recall y F-measure
    precision_0,recall_0,f_measure_0,precision_1,recall_1,f_measure_1 = calcula_precision_recall_fmeasure(confusionM,beta=1)

    # calcula porcentajes
    porcentajes = calcula_porcentajes(confusionM)

    #------
    # Colores personalizados para cada etiqueta
    # Inicializar la lista de colores
    colores = []

    # Asignar colores a cada etiqueta
    for etiqueta in etiquetas:
        if etiqueta == 0:
            colores.append('blue')   # Asignar azul para la etiqueta 0
        else:
            colores.append('green')  # Asignar verde para la etiqueta 1
    #------
            
    # Graficar los resultados dependiendo de las características seleccionadas
    imagen = ""
    if len(caracteristicas) == 2:
        # Graficar en 2D
        plt.figure(figsize=(8, 6))
        plt.scatter(X_filtrado[:, 0], X_filtrado[:, 1], c=etiquetas, cmap='viridis')
        plt.xlabel(caracteristicas[0])
        plt.ylabel(caracteristicas[1])
        plt.title('Agrupamiento espectral en 2D')
        plt.colorbar(label='Etiqueta del cluster')
        imagen = 'static/agrupamiento_2d.png'
        plt.savefig(imagen)  # Guardar la imagen
        plt.close()

    elif len(caracteristicas) == 3:
        # Graficar en 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_filtrado[:, 0], X_filtrado[:, 1], X_filtrado[:, 2], c=etiquetas, cmap='viridis')
        ax.set_xlabel(caracteristicas[0])
        ax.set_ylabel(caracteristicas[1])
        ax.set_zlabel(caracteristicas[2])
        ax.set_title('Agrupamiento espectral en 3D')
        fig.colorbar(sc, label='Etiqueta del cluster')
        imagen = 'static/agrupamiento_3d.png'
        plt.savefig(imagen)  # Guardar la imagen
        plt.close()

    # Renderizar el template y pasarle la ruta de la imagen
    return render_template('resultados.html', imagen=imagen, etiquetas=etiquetas, confusionM=confusionM, RMSE=RMSE, 
                          precision_0=precision_0, recall_0=recall_0, f_measure_0=f_measure_0, precision_1=precision_1,
                            recall_1=recall_1, f_measure_1=f_measure_1)




@app.route('/logout')
@login_required  # Asegura que solo los usuarios autenticados puedan cerrar sesión
def logout():
    logout_user()  # Cierra la sesión del usuario
    return redirect(url_for('login'))  # Redirige al login


if __name__ == '__main__':
    app.run(debug=True)
    