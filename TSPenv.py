import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

def get_Euclid_distance(problem, city1, city2):
    x1, y1 = problem.node_coords[city1 + 1]
    x2, y2 = problem.node_coords[city2 + 1]
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return round(distance)

def is_crossing(A, B, C, D):
    def cross_product(P, Q, R):
        return (Q[0] - P[0]) * (R[1] - P[1]) - (Q[1] - P[1]) * (R[0] - P[0])

    def is_between(P, Q, R):
        return min(P[0], R[0]) < Q[0] < max(P[0], R[0]) and \
               min(P[1], R[1]) < Q[1] < max(P[1], R[1])

    if cross_product(A, B, C) == 0 and is_between(A, C, B):
        return True
    if cross_product(A, B, D) == 0 and is_between(A, D, B):
        return True
    if cross_product(C, D, A) == 0 and is_between(C, A, D):
        return True
    if cross_product(C, D, B) == 0 and is_between(C, B, D):
        return True

    return (
        cross_product(A, B, C) * cross_product(A, B, D) < 0 and
        cross_product(C, D, A) * cross_product(C, D, B) < 0
    )

def calculate_new_path_crossings(new_segment, existing_path):
    A, B = new_segment
    crossings = 0
    for i in range(len(existing_path) - 1):
        C, D = existing_path[i], existing_path[i + 1]
        if is_crossing(A, B, C, D):
            crossings += 1
    return crossings

class TSPEnv(gym.Env):
    def __init__(self, tsp_problem):
        super(TSPEnv, self).__init__()

        self.start_city = 0
        self.problem = tsp_problem
        self.num_cities = self.problem.dimension
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.current_city = None
        self.visited_count = 1
        self.total_distance = 0
        self.action_space = spaces.Discrete(self.num_cities)
        self.observation_space = spaces.Discrete(self.num_cities)
        self.path = []

        self.reset()

    def reset(self, start=None, seed=None, options=None):
        self.visited = np.zeros(self.num_cities, dtype=bool)
        self.path = []
        self.current_city = random.randint(0, self.num_cities - 1) if start is None else start
        self.start_city = self.current_city
        self.visited[self.current_city] = True
        self.path.append(self.current_city)

        self.visited_count = 1
        self.total_distance = 0
        return self.current_city, {}

    def step(self, action):
        next_city = action
        distance = get_Euclid_distance(self.problem, self.current_city, next_city)
        reward = 0

        self.total_distance += distance
        self.visited[next_city] = True
        self.current_city = next_city
        self.visited_count += 1
        self.path.append(next_city)

        done = self.visited_count == self.num_cities
        if done and self.current_city != self.start_city:
            done = False
            self.visited_count -= 1
            self.visited[self.start_city] = False

        reward = -distance
        return self.current_city, [reward, distance] , done, False, {}

    def render(self, mode='human'):
        print(f"Current city: {self.current_city}, Total distance: {self.total_distance}")

    def get_total_distance(self):
        return self.total_distance


"""
A_coords = self.problem.node_coords[self.current_city + 1]
B_coords = self.problem.node_coords[next_city + 1]
existing_path_coords = [self.problem.node_coords[city + 1] for city in self.path]
crossings = calculate_new_path_crossings((A_coords, B_coords), existing_path_coords)
if crossings > 0:
    reward -= 100 * crossings
"""