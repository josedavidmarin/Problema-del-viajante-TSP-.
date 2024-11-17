import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance_matrix
import random

class AntColonyOptimizer:
    def __init__(self, locations, num_hormigas=10, num_iterations=100, parametro_parametro_alpha=1.0, parametro_parametro_beta=2.0, tasa_evaporacion=0.5, feromona_deposit=1.0):
        self.locations = locations
        self.num_hormigas = num_hormigas
        self.num_iterations = num_iterations
        self.parametro_parametro_alpha = parametro_parametro_alpha  # Traducción:  Importancia de feromonas
        self.parametro_parametro_beta = parametro_parametro_beta    # Traducción:  Importancia de la distancia
        self.tasa_evaporacion = tasa_evaporacion
        self.feromona_deposit = feromona_deposit
        self.num_locations = len(locations)
        self.distances = distance_matrix(locations, locations)
        self.feromona_matrix = np.ones((self.num_locations, self.num_locations))  # Traducción:  Inicialmente se colocan feromonas de igual nivel en cada camino

    def _initialize_hormigas(self):
        """Inicializa hormigas en posiciones aleatorias."""
        return [random.randint(0, self.num_locations - 1) for _ in range(self.num_hormigas)]

    def _select_next_location(self, current_location, visited):
        """Selecciona el siguiente nodo basado en la probabilidad ponderada por feromonas y la distancia."""
        feromonas = np.copy(self.feromona_matrix[current_location])
        distances = np.copy(self.distances[current_location])
        probabilities = np.zeros(self.num_locations)
        
        # Traducción:  Calcular probabilidad para cada ciudad
        for i in range(self.num_locations):
            if i not in visited:
                probabilities[i] = (feromonas[i] ** self.parametro_parametro_alpha) * ((1.0 / distances[i]) ** self.parametro_parametro_beta)
        
        # Traducción:  Normalización de probabilidades
        total = sum(probabilities)
        if total > 0:
            probabilities = probabilities / total
        else:
            probabilities.fill(1.0 / (self.num_locations - len(visited)))
        
        return np.random.choice(range(self.num_locations), p=probabilities)

    def _evaluate_solution(self, path):
        """Calcula el costo total del recorrido dado."""
        total_cost = sum(self.distances[path[i], path[i + 1]] for i in range(len(path) - 1))
        return total_cost + self.distances[path[-1], path[0]]  # Traducción:  Añadir el regreso al punto inicial

    def _update_feromonas(self, all_paths, all_costs):
        """Evapora y actualiza la matriz de feromonas basada en las rutas de todas las hormigas."""
        self.feromona_matrix *= (1 - self.tasa_evaporacion)  # Traducción:  Evaporación global

        for path, cost in zip(all_paths, all_costs):
            feromona_increase = self.feromona_deposit / cost  # Traducción:  Más corto = más depósito
            for i in range(len(path) - 1):
                self.feromona_matrix[path[i], path[i + 1]] += feromona_increase
                self.feromona_matrix[path[i + 1], path[i]] += feromona_increase  # Traducción:  Actualizar en ambas direcciones
            # Traducción:  Añadir feromonas para el camino de regreso
            self.feromona_matrix[path[-1], path[0]] += feromona_increase
            self.feromona_matrix[path[0], path[-1]] += feromona_increase

    def solve(self):
        """Ejecuta el algoritmo ACO para optimizar el TSP."""
        best_path = None
        best_cost = float('inf')

        for iteration in range(self.num_iterations):
            all_paths = []
            all_costs = []

            # Traducción:  Crear rutas para cada hormiga
            for _ in range(self.num_hormigas):
                current_location = random.randint(0, self.num_locations - 1)
                path = [current_location]
                visited = {current_location}

                # Traducción:  Construir camino
                for _ in range(self.num_locations - 1):
                    next_location = self._select_next_location(current_location, visited)
                    path.append(next_location)
                    visited.add(next_location)
                    current_location = next_location

                # Traducción:  Evaluar la solución (camino completo)
                cost = self._evaluate_solution(path)
                all_paths.append(path)
                all_costs.append(cost)

                # Traducción:  Verificar si es la mejor ruta encontrada
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # Traducción:  Actualizar feromonas basado en todas las rutas encontradas en esta iteración
            self._update_feromonas(all_paths, all_costs)

            print(f"Iteración {iteration + 1}/{self.num_iterations}, Mejor costo encontrado: {best_cost}")

        return best_path, best_cost

    def plot_solution(self, path):
        """Dibuja el recorrido TSP encontrado."""
        plt.figure(figsize=(10, 10))
        x_coords, y_coords = self.locations[:, 0], self.locations[:, 1]

        # Traducción:  Marcar el punto de inicio en rojo y el resto en azul
        plt.scatter(x_coords, y_coords, color='blue', zorder=5, label='Ubicaciones')
        plt.scatter(x_coords[path[0]], y_coords[path[0]], color='red', zorder=6, label='Inicio')

        # Traducción:  Dibujar el camino
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            plt.plot(
                [self.locations[start][0], self.locations[end][0]], 
                [self.locations[start][1], self.locations[end][1]], 
                'b-', lw=2, zorder=2
            )

        # Traducción:  Volver al inicio
        plt.plot(
            [self.locations[path[-1]][0], self.locations[path[0]][0]], 
            [self.locations[path[-1]][1], self.locations[path[0]][1]], 
            'b-', lw=2, zorder=2
        )
        
        plt.title("Ruta del TSP usando ACO")
        plt.xlabel("Coordenada X")
        plt.ylabel("Coordenada Y")
        plt.legend()
        plt.grid(True)
        plt.show()

# Traducción:  Configuración de ubicaciones aleatorias en un plano 2D
num_locations = 500
locations = np.random.rand(num_locations, 2) * 100  # Traducción:  Coordenadas aleatorias en un espacio 2D

# Traducción:  Crear y resolver TSP usando ACO
aco_solver = AntColonyOptimizer(locations)
best_path, best_cost = aco_solver.solve()

# Traducción:  Mostrar resultados y visualizar el recorrido
print("Mejor ruta encontrada:", best_path)
for i, location in enumerate(best_path):
    print(f"{i+1}: Ciudad {location}")
print(f"Total de ciudades visitadas: {len(best_path)}")
print("Costo total de la mejor ruta:", best_cost)
aco_solver.plot_solution(best_path)
