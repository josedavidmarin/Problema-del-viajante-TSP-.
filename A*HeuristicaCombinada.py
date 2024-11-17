import heapq
from collections import defaultdict
from itertools import combinations
import random

class Grafo:
    def __init__(self):
        self.connections = defaultdict(list)
        self.distances = {}

    def add_connection(self, origin, destination, distance):
        self.connections[origin].append(destination)
        self.connections[destination].append(origin)
        self.distances[(origin, destination)] = distance
        self.distances[(destination, origin)] = distance

    def calculate_mst(self, nodes):
        nodes = frozenset(nodes)
        if len(nodes) <= 1:
            return 0

        cost_mst = 0
        explored = set()
        start_node = next(iter(nodes))
        priority_queue = [(0, start_node)]

        while len(explored) < len(nodes):
            cost, current = heapq.heappop(priority_queue)
            if current in explored:
                continue

            explored.add(current)
            cost_mst += cost

            for neighbor in self.connections[current]:
                if neighbor not in explored and neighbor in nodes:
                    heapq.heappush(priority_queue, (self.distances[(current, neighbor)], neighbor))

        if len(explored) < len(nodes):
            return float('inf')

        return cost_mst

    def a_star_with_combined_heuristics(self, start, parametro_alpha=1.0, parametro_beta=2.0):
        open_set = []
        heapq.heappush(open_set, (0, start, frozenset([start]), 0))
        best_cost = {(start, frozenset([start])): 0}
        feromonas = defaultdict(float)
        total_nodes = len(self.connections)

        while open_set:
            f, current, visited, g = heapq.heappop(open_set)

            if len(visited) == total_nodes and current == start:
                return g

            for neighbor in self.connections[current]:
                if neighbor not in visited or (len(visited) == total_nodes and neighbor == start):
                    new_visited = visited | frozenset([neighbor])
                    g_new = g + self.distances[(current, neighbor)]

                    remaining_nodes = set(self.connections.keys()) - new_visited
                    mst_cost = self.calculate_mst(remaining_nodes)

                    aco_heuristic = feromonas[(current, neighbor)] + self.distances[(current, neighbor)]
                    h = parametro_alpha * mst_cost + parametro_beta * aco_heuristic

                    f_new = g_new + h
                    state = (neighbor, new_visited)
                    if g_new < best_cost.get(state, float('inf')):
                        best_cost[state] = g_new
                        heapq.heappush(open_set, (f_new, neighbor, new_visited, g_new))

        return None

def generate_tsp_instance(size, max_distance=100):
    nodes = [chr(i) for i in range(65, 65 + size)]
    graph = Grafo()
    for (node1, node2) in combinations(nodes, 2):
        graph.add_connection(node1, node2, random.randint(1, max_distance))
    return graph, nodes

# Traducción:  Example Usage
if __name__ == "__main__":
    graph, nodes = generate_tsp_instance(5)
    start_node = nodes[0]
    cost = graph.a_star_with_combined_heuristics(start_node)
    print(f"Optimal cost using combined heuristics: {cost}")
