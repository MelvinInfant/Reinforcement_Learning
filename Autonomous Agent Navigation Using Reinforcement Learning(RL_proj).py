import numpy as np
import pygame
import time

# Environment setup
class GridEnvironment:
    def __init__(self, grid_size=(5, 5), start=(0, 0), goal=(4, 4), obstacle_percentage=0.2):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.state = start
        self.obstacle_percentage = obstacle_percentage
        self.obstacles = self.generate_obstacles()  # Fixed random obstacles
        self.reset()

    def generate_obstacles(self):
        np.random.seed(42)  # Ensures consistent random obstacle placement
        num_obstacles = int(self.grid_size[0] * self.grid_size[1] * self.obstacle_percentage)
        obstacles = set()

        while len(obstacles) < num_obstacles:
            obstacle = (np.random.randint(self.grid_size[0]), np.random.randint(self.grid_size[1]))
            if obstacle != self.start and obstacle != self.goal:
                obstacles.add(obstacle)

        return obstacles

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        # Define action effects: up, down, left, right
        actions = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        dx, dy = actions[action]
        new_state = (x + dx, y + dy)

        # Check for boundaries and obstacles
        if (0 <= new_state[0] < self.grid_size[0] and
            0 <= new_state[1] < self.grid_size[1] and
            new_state not in self.obstacles):
            self.state = new_state

        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal

        return self.state, reward, done

    def render(self, screen, cell_size, current_reward):
        screen.fill((255, 255, 255))

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                if (i, j) in self.obstacles:
                    pygame.draw.rect(screen, (0, 0, 0), rect)  # Obstacles
                elif (i, j) == self.goal:
                    pygame.draw.rect(screen, (0, 255, 0), rect)  # Goal
                elif (i, j) == self.state:
                    pygame.draw.rect(screen, (0, 0, 255), rect)  # Agent
                else:
                    pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # Grid

        # Display the current reward for this step
        font = pygame.font.Font(None, 36)
        reward_text = font.render(f"Step Reward: {current_reward:.2f}", True, (0, 0, 0))
        screen.blit(reward_text, (10, 10))

        pygame.display.flip()

# Q-Learning implementation
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploitation
            if state in self.q_table:
                return np.argmax(self.q_table[state])
        # Exploration
        return np.random.choice([0, 1, 2, 3])

    def update_q_value(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]

        max_next_q = max(self.q_table.get(next_state, [0, 0, 0, 0]))
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.lr * (reward + self.gamma * max_next_q - current_q)

# Main training loop
def train_agent():
    pygame.init()

    grid_size = (5, 5)  # Smaller grid size
    cell_size = 100     # Larger cells for better visibility
    screen = pygame.display.set_mode((grid_size[1] * cell_size, grid_size[0] * cell_size))
    pygame.display.set_caption("Simulated Robot Navigation")

    env = GridEnvironment(grid_size=grid_size)
    agent = QLearningAgent(env)

    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Render the environment with the current step's reward
            env.render(screen, cell_size, total_reward)
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state

            total_reward += reward

            time.sleep(0.1)

        print(f"Episode {episode + 1}/{episodes} completed. Total Reward: {total_reward:.2f}")

    print("Training complete!")
    env.render(screen, cell_size, total_reward)
    time.sleep(3)
    pygame.quit()

if __name__ == "__main__":
    train_agent()