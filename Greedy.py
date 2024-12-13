import tsplib95

def greedy_tsp_from_tsplib(file_path):
    """
    Solves the Traveling Salesman Problem using a greedy approach with data from a TSPLIB file.

    Args:
        file_path (str): Path to the TSPLIB file.

    Returns:
        tuple: A tuple (route, total_distance) where route is the order of cities visited,
               and total_distance is the total distance of the route.
    """
    # Load the TSPLIB problem
    problem = tsplib95.load(file_path)

    # Extract the list of nodes
    nodes = list(problem.get_nodes())
    n = len(nodes)
    best_route = []
    best_total_distance = float('inf')

    for start in range(n):
        visited = [False] * n
        route = [nodes[start]]
        visited[nodes[start] - 1] = True
        total_distance = 0
        current_city = nodes[start]

        for _ in range(n - 1):
            nearest_city, min_distance = min(
                ((next_city, problem.get_weight(current_city, next_city))
                 for next_city in nodes if not visited[next_city - 1]),
                key=lambda x: x[1]
            )
            visited[nearest_city - 1] = True
            route.append(nearest_city)
            total_distance += min_distance
            current_city = nearest_city

        total_distance += problem.get_weight(current_city, route[0])
        route.append(route[0])

        if total_distance < best_total_distance:
            best_route, best_total_distance = route, total_distance

    return problem, best_route, best_total_distance
"""
file_path = "xqf131.tsp"
problem, route, total_distance = greedy_tsp_from_tsplib(file_path)
print("Route:", route)
print("Total Distance:", total_distance)
"""


import matplotlib.pyplot as plt

def plot_path(path, node_coords, total_distance):
    """
    绘制路径图
    Args:
        path: 城市访问路径列表，例如 [0, 3, 2, 1, 4, 0]
        node_coords: 城市的坐标字典，例如 {0: (0, 0), 1: (1, 2), ...}
    """
    # 提取路径中每个城市的坐标
    x_coords = [node_coords[city][0] for city in path]
    y_coords = [node_coords[city][1] for city in path]

    # 绘制路径
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker="o", markersize=8, label="Path")  # 连线
    plt.scatter(x_coords, y_coords, c="red", label="Cities")  # 城市点

    # 设置图形标题和坐标轴
    plt.title(f"Greedy Traveling Salesman Path with total distance: {total_distance}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()

# plot_path(route, problem.node_coords, total_distance)