#%%
import numpy as np

from Greedy import greedy_tsp_from_tsplib, plot_path
from QLearningAgent import QLearningAgent

#print(problem)
from TSPenv import TSPEnv
problem, start_path, total_distance = greedy_tsp_from_tsplib('dj38.tsp')
print(total_distance)
plot_path(start_path, problem.node_coords, total_distance)
#%%
train_env = TSPEnv(problem)
train_env.reset()

agent = QLearningAgent(train_env)

agent.train(episodes=10000)

#%%
test_env = TSPEnv(problem)
agent.env = test_env
best_path = []
best_reward = np.inf

for i in range(problem.dimension):
    state, info = test_env.reset(start = i)
    done = False
    total_reward = 0
    path = [state+1]

    while not done:
        action = agent.eval_choose_action(state)
        next_state, reward, terminated, truncated , _ = test_env.step(action)
        done = terminated or truncated
        state = next_state
        path.append(state+1)
        total_reward += problem.get_weight(path[-2], path[-1])
    if total_reward < best_reward:
        best_path = path
        best_reward = total_reward

print(f"Total distance: {best_reward}")
print(best_path)
test_env.close()
#%%
import matplotlib.pyplot as plt

def plot_path(path, node_coords):
    # 提取路径中每个城市的坐标
    x_coords = [node_coords[city][0] for city in path]
    y_coords = [node_coords[city][1] for city in path]

    # 绘制路径
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, marker="o", markersize=8, label="Path")  # 连线
    plt.scatter(x_coords, y_coords, c="red", label="Cities")  # 城市点

    # 设置图形标题和坐标轴
    plt.title(f"Traveling Salesman Path with total distance: {best_reward}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend()
    plt.show()

plot_path(best_path, problem.node_coords)