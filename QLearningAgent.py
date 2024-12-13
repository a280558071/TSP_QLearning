import random
from collections import defaultdict, deque

import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class QLearningAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 0.2,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.97,
        start_Q = None
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        if start_Q is not None:
            self.q_values = start_Q
        else:
            self.q_values = defaultdict( lambda: np.zeros(env.action_space.n))
        self.discount_factor = discount_factor
        self.lr = learning_rate

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = 32
        self.memory = ReplayBuffer()

        self.training_error = []

    def choose_action(self, obs: int):
        """Choose an action using an epsilon-greedy policy,
        but limit actions to unvisited points.

        Args:
            obs: The current state

        Returns:
            The action to take
        """
        # 获取未访问点的动作列表
        copyQ =np.copy(self.q_values[obs])
        copyQ[self.env.visited] = -np.inf

        # Epsilon-greedy 选择动作
        if np.random.rand() < self.epsilon:
            # 随机选择一个未访问的动作
            return np.random.choice([a for a in range(self.env.action_space.n) if not self.env.visited[a]])
        else:
            # 贪婪选择：从未访问点中选择 Q 值最大的动作
            best_action = np.argmax(copyQ)
            return best_action

    def eval_choose_action(self, obs: int):
        copyQ = np.copy(self.q_values[obs])
        copyQ[self.env.visited] = -np.inf

        # 贪婪选择：从未访问点中选择 Q 值最大的动作
        best_action = np.argmax(copyQ)
        return best_action

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update_Q_batch(self):
        """Sample from replay buffer and update Q-values in batch."""
        if len(self.memory) < self.batch_size:
            return  # 不足一个批次，不更新

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 将数据转为array便于批处理
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.bool_)

        # 根据 next_state 计算 max Q
        max_next_Q = np.array([np.max(self.q_values[s]) for s in next_states])
        targets = rewards + (1 - dones) * self.discount_factor * max_next_Q

        # 更新Q
        for i in range(self.batch_size):
            s = states[i]
            a = actions[i]
            td_error = targets[i] - self.q_values[s][a]
            self.q_values[s][a] += self.lr * td_error
            self.training_error.append(td_error)

    def update_Q_table(self, state: int,
                       action: int,
                       reward: float,
                       terminated: bool,
                       next_state: int) -> None:
        """Update the Q-value table using the Q-learning update rule.

        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            terminated: stop or not
        """
        future_Q = (not terminated) *  np.max(self.q_values[next_state])
        td_target = reward + self.discount_factor * future_Q - self.q_values[state][action]

        self.q_values[state][action] += self.lr * td_target
        self.training_error.append(td_target)

    def decay(self):
        """Decay the epsilon value."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes:int = 1000) -> None:
        """Train the agent for a number of episodes.

        Args:
            episodes: The number of episodes to train for
        """
        max_reward = np.inf
        rewards = []
        update_steps = 0
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0
            cnt = 0
            path = [state]

            while not done:
                action = self.choose_action(state)
                next_obs, reward_dist, terminated, truncated, info= self.env.step(action)
                #self.update_Q_table(state, action, float(reward_dist[0]) , terminated, next_obs)
                done = terminated or truncated

                self.store_experience(state, action, float(reward_dist[0]), next_obs, done)
                update_steps += 1
                if update_steps % 5 == 0:
                    self.update_Q_batch()

                state = next_obs
                path.append(state)
                total_reward += reward_dist[1]

            rewards.append(total_reward)
            max_reward = min(max_reward, total_reward)
            self.decay()
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes} - Avg Reward: {np.mean(rewards[-100:])}, Epsilon: {self.epsilon:.4f}")
        print(f"Max Reward: {max_reward}")


    def evaluate(self, episodes: int = 100) -> None:
        """Evaluate the agent over a number of episodes.

        Args:
            episodes: The number of episodes to evaluate over

        Returns:
            The average reward per episode
        """
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = self.eval_choose_action(state)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_obs
                total_reward += terminated

            rewards.append(total_reward)

        print(f"Success Rate: {sum(rewards) / episodes:.2f}")