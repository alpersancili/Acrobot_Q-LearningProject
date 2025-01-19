# Acrobot_Q-LearningProject
This project implements a Q-learning agent to solve the Acrobot-v1 environment from the Gymnasium library. The Acrobot is a classic control problem involving a two-link pendulum system that must swing itself up to a target position. The solution is achieved using reinforcement learning with a discrete approximation of the continuous state space.

FEATURES

Q-Learning Algorithm: Implements an off-policy TD control algorithm to learn an optimal policy.
State Discretization: Converts the continuous state space into discrete bins to make the problem tractable for Q-learning.
Epsilon-Greedy Exploration: Balances exploration and exploitation with epsilon decay.
Training and Rendering: Provides options to train the agent or visualize its performance.
Model Persistence: Saves the Q-table (acrobot.pkl) whenever the agent achieves a new best reward.

HOW IT WORKS

The Acrobot's state space consists of six continuous variables:
Cosine and sine of the first link's angle
Cosine and sine of the second link's angle
Angular velocities of both links
The state space is discretized into bins for each variable to reduce complexity.
The Q-learning algorithm updates the Q-table based on rewards and the maximum expected future rewards.
The agent explores actions randomly at first, gradually shifting to exploiting the learned policy as training progresses.
The model is saved whenever the agent surpasses its previous best performance.

FILE DESCRIPTIONS

acrobot_agent.py: Main script containing the Q-learning implementation.

acrobot.pkl: Serialized Q-table saved during training.

acrobot.png: Visualization of mean rewards over episodes during training.

https://github.com/user-attachments/assets/2cfc141d-3a3a-4c9f-aefb-b2151619bd1e





