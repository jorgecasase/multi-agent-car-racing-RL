# Proyecto de IA de RL: Comparación de Algoritmos de Refuerzo

<div>
  <img src="https://github.com/jorgecasase/github-repos-img/blob/main/img/python.png" alt="python" height="100"/>
</div>

## Introducción
Este proyecto tiene como objetivo estudiar y comparar distintos algoritmos de Aprendizaje por Refuerzo (RL), explorando sus características y comportamientos. Implementamos tres tipos de algoritmos:

- **Deep Q-Learning (Valor)**
- **Proximal Policy Optimization (Policy)**
- **Advantage Actor-Critic (Mixed)**

Finalmente, los modelos entrenados compiten en un entorno de carreras multiagente para evaluar su desempeño.

---

## Simulaciones

### Deep Q-Learning
<img src="videos/gifs/DQL.gif" alt="Deep Q-Learning en acción" height="400">

### Proximal Policy Optimization
<img src="videos/gifs/PPO.gif" alt="PPO en acción" height="400">

### Advantage Actor-Critic
<img src="videos/gifs/A2C.gif" alt="A2C en acción" height="400">

### Carrera Multiagente
<img src="videos/gifs/racefinal.gif" alt="Carrera Multiagente" height="400">
---

## Estructura del Proyecto
```
├── A2C/ # Carpeta para el algoritmo A2C 
├── PPO/ # Carpeta para el algoritmo PPO 
├── deepQ/ # Carpeta para el algoritmo Deep Q-Learning 
├── gym_multi_car_racing/ # Librería modificada de MultiCarRacing 
├── race/ # Scripts para competir con los modelos entrenados 
├── videos/ # Contiene gifs y videos de las simulaciones 
├── .gitignore # Archivos ignorados por git 
├── enviroment.yml # Configuración del entorno de Anaconda 
├── setup.py # Configuración de instalación de librería```

---

## Archivos importantes por algoritmo
Cada carpeta de algoritmo incluye los siguientes archivos:

- **`train.py`**: Entrena el modelo.
- **`trainhuman.py`**: Entrena el modelo con render gráfico (más lento pero visualizable).
- **`replay.py`**: Carga un modelo entrenado y lo muestra jugando.
- **`plot_results.py`**: Genera gráficas de episodios vs. recompensas para analizar el aprendizaje.
- **`paralleltrain.py`**: Entrenamiento intensivo con múltiples entornos paralelos.

---

## Ejecución

### Requisitos Previos
- **Anaconda instalado.**
- **GPU compatible con CUDA** (opcional, pero recomendado para acelerar los entrenamientos).

---

### Instrucciones

#### Clonar el Repositorio
```
git clone https://github.com/tu_usuario/multi-agent-car-racing-RL.git```
```cd multi-agent-car-racing-RL```

### Crear el Entorno de Anaconda
´´´
conda env create -f environment.yml```
```conda activate car´´´

### Instalar la Librería gym_multi_car_racing
```python -m pip install -e gym_multi_car_racing```

## Entrenar un Modelo

Ve a la carpeta del algoritmo que quieras entrenar, por ejemplo, PPO:
```cd PPO python train.py```

## Ver una Simulación
Una vez entrenado, puedes cargar el modelo y verlo jugar:
```python replay.py```

## Realizar una Carrera Multiagente
Ejecuta el script de carrera en la carpeta `race`:
```cd race```
```python race.py```

## Resultados 

Ejemplo de Comportamiento de los Modelos
Aquí puedes ver cómo se desempeña cada algoritmo en el entorno:

1. Deep Q-Learning
2. Proximal Policy Optimization
3. Advantage Actor-Critic

---

## Carrera multiagente
En la carrera final, los tres modelos compiten para obtener el mayor puntaje. Aquí está el ranking final basado en 10 carreras:

## Carrera Multiagente

En la carrera final, los tres modelos compiten para obtener el mayor puntaje. Aquí está el ranking final basado en 10 carreras:

| Modelo | Carreras Ganadas |
|--------|------------------|
| PPO    | 6                |
| A2C    | 3                |
| DQN    | 1                |

Cada modelo utiliza una estrategia única derivada de su algoritmo de RL, demostrando cómo estos enfoques diferentes se desempeñan en un entorno competitivo.

---

## Detalles del Entrenamiento

### Configuración de los Algoritmos

- **Deep Q-Learning (DQN)**: Utiliza una tabla de valores Q para decidir la mejor acción en función de un estado observado.
- **Proximal Policy Optimization (PPO)**: Algoritmo basado en políticas que optimiza directamente la probabilidad de tomar una acción en función de un estado.
- **Advantage Actor-Critic (A2C)**: Combina una política (actor) con una función de valor (critic) para mejorar la estabilidad y la eficiencia del entrenamiento.

### Estrategias Utilizadas

- **Entornos paralelos**: Se utilizan múltiples entornos para optimizar el tiempo de entrenamiento y mejorar la eficiencia del hardware disponible.
- **Visualización en tiempo real**: Con `trainhuman.py`, puedes observar el comportamiento del modelo durante el entrenamiento.
- **Gráficas de aprendizaje**: Con `plot_results.py`, puedes analizar la eficiencia y las recompensas obtenidas a lo largo de los episodios.

---

## Ejemplo de Gráfica de Entrenamiento

Las recompensas acumuladas por episodio permiten visualizar si el modelo está aprendiendo de manera efectiva y su velocidad de convergencia:

![Ejemplo de gráfica](videos/plot_rewards_example.png)

