**AUTONOMOUS AGENT NAVIGATION USING REINFORCEMENT LEARNING**

**Problem Statement**
The goal of the project is to address the problem of robotic navigation in a grid-based environment. In other words, the task is to train an agent-a simulated robot-to find the shortest path from a starting point to a goal while avoiding obstacles. This mirrors real-world applications in which autonomous agents, such as robots, must navigate unknown environments efficiently to achieve a goal. The challenge lies in the design of an algorithm that can learn an optimal policy to reach the goal using minimal steps without causing a collision with obstacles.

**Implemented Idea**
The project applies the Q-Learning concept, a model-free reinforcement learning algorithm, to allow the agent to learn a policy for optimal navigation. Q-Learning allows the agent to learn the value of actions in a given state through interactions with the environment, thereby improving its decision-making over time. The environment can be represented as a grid with random obstacles, a starting point, and a goal. The agent uses a Q-table to store state-action values, which it updates according to rewards received upon taking certain actions in different states.

**Rules for the Reward System and Agent Movement**
1.	State and Actions:
o	States: The agent’s state is defined by its coordinates on the grid.
o	Actions: The agent can move in four possible directions: up, down, left, or right.
2.	Movement:
o	The agent changes its position from the current one to a new one according to the chosen action. If the new position is valid (within the grid boundaries and not an obstacle), the move is accepted.
o	If the move reached the goal state, it gets a reward of +1, which means the move was successful. If it hits an obstacle or goes out of bounds, then it gets a small negative reward of -0.1 to penalize the action.
3.	Q-Learning Updates:
o	Choosing Actions: The agent decides on its next action based on an exploration-exploitation strategy, balancing between exploring new actions and exploiting known best actions to maximize immediate rewards.
o	Q-value Updates: The Q-value for a state-action pair is updated using the formula below,
Q(state, action) = Q(state, action) + a × : (reward +y × max Q(next_state, a') –Q(state,action))a'
where a is the learning rate, y is the discount factor, and maxɑ' Q(next_state, a') is the maximum       Q-value for all possible next actions from the next state.


**Overall System and Components**
•	Grid Environment: Represents the simulated world in which the agent navigates. It has the capability to generate obstacles, to manage the state of the agent, and to determine outcomes resulting from the actions of the agent.
•	Q-Learning Agent: This module embodies the Q-Learning algorithm. It is responsible for keeping the Q-table, selecting the action from the current state, and updating the Q-values for every taken action.
•	Rendering: ‘Pygame’ library is used for visualizing the environment. The rendering updates the screen with the agent's position, obstacles, and current rewards to provide a visual feedback loop during training.
•	Training Loop: Repeatedly resets the environment, forces the agent to take actions, and updates the Q-values for a fixed number of episodes. Each episode represents a complete navigation task from start to goal, collecting rewards along the way to improve the agent's learning.
Conclusion
The project represents an example of real applicative Q-Learning in reinforcement learning to solve real-world problems of navigation, thereby contributing to both robotics and artificial intelligence fields.
