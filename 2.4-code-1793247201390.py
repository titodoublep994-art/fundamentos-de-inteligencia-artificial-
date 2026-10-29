import heapq
import matplotlib.pyplot as plt

# Definición de la red de transporte (Grafo con costos de adyacencia)
red_transporte = {
    'A': {'B': 3, 'C': 5},
    'B': {'A': 3, 'C': 2, 'D': 4},
    'C': {'A': 5, 'B': 2, 'D': 1, 'E': 6},
    'D': {'B': 4, 'C': 1, 'E': 2, 'F': 3},
    'E': {'C': 6, 'D': 2, 'F': 1},
    'F': {'D': 3, 'E': 1}
}

# Coordenadas geográficas simuladas para el cálculo de la heurística
coordenadas = {
    'A': (0, 0),
    'B': (1, 1),
    'C': (2, 0),
    'D': (3, 1),
    'E': (4, 0),
    'F': (5, 1)
}

def heuristica(nodo, meta):
    # Distancia Euclidiana en línea recta
    x1, y1 = coordenadas[nodo]
    x2, y2 = coordenadas[meta]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# --- Implementación de Dijkstra ---
def dijkstra(grafo, inicio, meta):
    pq = [(0, inicio, [inicio])]
    visitados = set()
    nodos_expandidos = 0

    while pq:
        costo, actual, camino = heapq.heappop(pq)
        if actual in visitados:
            continue
        visitados.add(actual)
        nodos_expandidos += 1

        if actual == meta:
            return camino, costo, nodos_expandidos

        for vecino, peso in grafo[actual].items():
            if vecino not in visitados:
                heapq.heappush(pq, (costo + peso, vecino, camino + [vecino]))
    return [], float('inf'), nodos_expandidos

# --- Implementación de A* ---
def a_estrella(grafo, inicio, meta):
    h_ini = heuristica(inicio, meta)
    pq = [(h_ini, 0, inicio, [inicio])]
    visitados = set()
    nodos_expandidos = 0

    while pq:
        f, g, actual, camino = heapq.heappop(pq)
        if actual in visitados:
            continue
        visitados.add(actual)
        nodos_expandidos += 1

        if actual == meta:
            return camino, g, nodos_expandidos

        for vecino, peso in grafo[actual].items():
            if vecino not in visitados:
                g_vecino = g + peso
                h_vecino = heuristica(vecino, meta)
                f_vecino = g_vecino + h_vecino
                heapq.heappush(pq, (f_vecino, g_vecino, vecino, camino + [vecino]))
    return [], float('inf'), nodos_expandidos

# Ejecución de los algoritmos de 'A' a 'F'
inicio, meta = 'A', 'F'
camino_d, costo_d, nodos_d = dijkstra(red_transporte, inicio, meta)
camino_a, costo_a, nodos_a = a_estrella(red_transporte, inicio, meta)

print(f"Dijkstra -> Ruta: {camino_d} | Costo: {costo_d} | Nodos expandidos: {nodos_d}")
print(f"A*       -> Ruta: {camino_a} | Costo: {costo_a} | Nodos expandidos: {nodos_a}")

# --- Generación del Dashboard de Métricas ---
def generar_dashboard():
    algoritmos = ['Dijkstra', 'A*']
    nodos_exp = [nodos_d, nodos_a]
    costos = [costo_d, costo_a]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    axs[0].bar(algoritmos, nodos_exp, color=['#4C72B0', '#55A868'])
    axs[0].set_title('Eficiencia: Nodos Expandidos')
    axs[0].set_ylabel('Cantidad')

    axs[1].bar(algoritmos, costos, color=['#C44E52', '#8172B3'])
    axs[1].set_title('Costo Total de la Ruta')
    axs[1].set_ylabel('Costo / Distancia')

    plt.suptitle('Dashboard de Rendimiento: Red de Transporte GPS')
    plt.tight_layout()
    plt.savefig('dashboard_rendimiento.png')
    print("\n[INFO] Dashboard generado y guardado exitosamente como 'dashboard_rendimiento.png'.")

generar_dashboard()