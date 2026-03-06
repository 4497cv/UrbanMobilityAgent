import osmnx as ox
from math import inf, sqrt
import time

# ======================================================
# ===== TU CÓDIGO (SIN CAMBIOS) =========================
# ======================================================

def getLetra(num):
    return chr(num + ord('A'))

def getNum(letra):
    return ord(letra) - ord('A')

def matrix_from_edge_list(edge_list, nodes):
    matrix = [[None]*nodes for _ in range(nodes)]
    for u, v, w in edge_list:
        matrix[u][v] = w
    return matrix

def distance_manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def distance_euclidean(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def letters_edges_list_to_numeric(numeric_letters, weighted_graph=True):
    result = []
    for u, v, w in numeric_letters:
        result.append((getNum(u), getNum(v), w))
    return result

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, val, priority):
        self.queue.append((priority, val))

    def pop(self):
        min_idx = 0
        for i in range(len(self.queue)):
            if self.queue[i][0] < self.queue[min_idx][0]:
                min_idx = i
        priority, value = self.queue.pop(min_idx)
        return value, priority

    def empty(self):
        return len(self.queue) == 0

class Dijkstra:
    def __init__(self, edges_list, n_vertex, node_positions={(0,0)}):
        edges_list_numeric = letters_edges_list_to_numeric(edges_list, True)
        self.adj_matrix = matrix_from_edge_list(edges_list_numeric, n_vertex)
        self.positions = node_positions

    def dijkstra_PriorityQueue(self, start):
        graph = self.adj_matrix
        MAX_V = len(graph)
        distance = [[inf, i] for i in range(MAX_V)]
        prev = [None] * MAX_V
        distance[start][0] = 0

        pq = PriorityQueue()
        pq.push(start, 0)

        while not pq.empty():
            node, dist = pq.pop()
            if dist > distance[node][0]:
                continue

            for i in range(MAX_V):
                if graph[node][i] is not None:
                    new_dist = dist + graph[node][i]
                    if new_dist < distance[i][0]:
                        distance[i][0] = new_dist
                        prev[i] = node
                        pq.push(i, new_dist)

        self.distances = distance
        self.prev = prev

    def dijkstra_Heuristic(self, start, end, heuristic):
        graph = self.adj_matrix
        MAX_V = len(graph)
        distance = [inf]*MAX_V
        prev = [None]*MAX_V
        distance[start] = 0

        pq = PriorityQueue()
        h = heuristic(self.positions[getLetra(start)], self.positions[getLetra(end)])
        pq.push(start, h)

        while not pq.empty():
            node, _ = pq.pop()
            for i in range(MAX_V):
                if graph[node][i] is not None:
                    g = distance[node] + graph[node][i]
                    h = heuristic(self.positions[getLetra(i)], self.positions[getLetra(end)])
                    f = g + h
                    if g < distance[i]:
                        distance[i] = g
                        prev[i] = node
                        pq.push(i, f)

        self.distances = [[d, i] for i, d in enumerate(distance)]
        self.prev = prev

    def ReconstructPath(self, end):
        path = []
        current = end
        while current is not None:
            path.append(getLetra(current))
            current = self.prev[current]
        path.reverse()
        print(" -> ".join(path))
        print("Distancia:", self.distances[end][0])
        return path

# ======================================================
# ===== ADAPTACIÓN OSMnx → TU DIJKSTRA ==================
# ======================================================

def main():

    import networkx as nx
    import osmnx as ox

    print("Loading graph...")
    G = ox.graph_from_xml("provi2.osm", simplify=True)

    # 🔑 convertir a no dirigido
    G = G.to_undirected()

    # 🔑 tomar componente conexa más grande
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    # 🔑 tomar pocos nodos CONECTADOS
    nodes = list(G.nodes)
    nodes = nodes[:6]   # pequeño a propósito
    subG = G.subgraph(nodes)

    print("Nodes:", len(subG.nodes))
    print("Edges:", len(subG.edges))

    # mapear nodos → letras
    node_map = {n: getLetra(i) for i, n in enumerate(subG.nodes)}

    # construir aristas BIDIRECCIONALES
    edges_list = []
    for u, v, data in subG.edges(data=True):
        w = data.get("length", 1)
        edges_list.append((node_map[u], node_map[v], w))
        edges_list.append((node_map[v], node_map[u], w))

    print("\nEdges list:")
    for e in edges_list:
        print(e)

    # posiciones
    node_positions = {
        node_map[n]: (G.nodes[n]['x'], G.nodes[n]['y'])
        for n in subG.nodes
    }

    n_vertex = len(node_positions)

    # 🔑 elegir inicio y fin que SÍ estén conectados
    start = 0
    end = n_vertex - 1

    print("\nStart:", getLetra(start))
    print("End:", getLetra(end))

    print("\nDijkstra clásico:")
    dj = Dijkstra(edges_list, n_vertex, node_positions)
    dj.dijkstra_PriorityQueue(start)
    dj.ReconstructPath(end)

    print("\nDijkstra + Euclidiana:")
    dj = Dijkstra(edges_list, n_vertex, node_positions)
    dj.dijkstra_Heuristic(start, end, distance_euclidean)
    dj.ReconstructPath(end)

main()
