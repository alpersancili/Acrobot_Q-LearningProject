import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False, num_episodes=10000):
    env = gym.make('Acrobot-v1', render_mode='human' if render else None)

    # Hyperparameters
    learning_rate_a = 0.1        # Alpha (learning rate)
    discount_factor_g = 0.9      # Gamma (discount factor)
    epsilon = 1                  # Start epsilon at 1 (100% random actions)
    epsilon_decay_rate = 0.0005  # Epsilon decay rate
    epsilon_min = 0.05           # Minimum epsilon
    divisions = 15               # Used to discretize continuous state space

    # Discretization
    th1_cos  = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divisions)
    th1_sin  = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divisions)
    th2_cos  = np.linspace(env.observation_space.low[2], env.observation_space.high[2], divisions)
    th2_sin  = np.linspace(env.observation_space.low[3], env.observation_space.high[3], divisions)
    th1_w    = np.linspace(env.observation_space.low[4], env.observation_space.high[4], divisions)
    th2_w    = np.linspace(env.observation_space.low[5], env.observation_space.high[5], divisions)

    if is_training:
        q = np.zeros((len(th1_cos)+1, len(th1_sin)+1, len(th2_cos)+1, len(th2_sin)+1, len(th1_w)+1, len(th2_w)+1, env.action_space.n))
    else:
        with open('acrobot.pkl', 'rb') as f:
            q = pickle.load(f)

    best_reward = -np.inf
    rewards_per_episode = []

    for episode in range(num_episodes):
        state = env.reset()[0]
        s_i0 = np.digitize(state[0], th1_cos)
        s_i1 = np.digitize(state[1], th1_sin)
        s_i2 = np.digitize(state[2], th2_cos)
        s_i3 = np.digitize(state[3], th2_sin)
        s_i4 = np.digitize(state[4], th1_w)
        s_i5 = np.digitize(state[5], th2_w)

        terminated = False
        rewards = 0

        while not terminated:
            action = env.action_space.sample() if is_training and np.random.rand() < epsilon else np.argmax(q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, :])
            new_state, reward, terminated, _, _ = env.step(action)

            ns_i0 = np.digitize(new_state[0], th1_cos)
            ns_i1 = np.digitize(new_state[1], th1_sin)
            ns_i2 = np.digitize(new_state[2], th2_cos)
            ns_i3 = np.digitize(new_state[3], th2_sin)
            ns_i4 = np.digitize(new_state[4], th1_w)
            ns_i5 = np.digitize(new_state[5], th2_w)

            if is_training:
                q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, ns_i3, ns_i4, ns_i5, :]) -
                    q[s_i0, s_i1, s_i2, s_i3, s_i4, s_i5, action]
                )

            state = new_state
            s_i0, s_i1, s_i2, s_i3, s_i4, s_i5 = ns_i0, ns_i1, ns_i2, ns_i3, ns_i4, ns_i5
            rewards += reward

        rewards_per_episode.append(rewards)

        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                with open('acrobot.pkl', 'wb') as f:
                    pickle.dump(q, f)

        if is_training and episode % 100 == 0:
            mean_reward = np.mean(rewards_per_episode[-100:])
            print(f'Episode: {episode}, Epsilon: {epsilon:.2f}, Best Reward: {best_reward:.1f}, Mean Rewards: {mean_reward:.1f}')

            plt.plot([np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(episode)])
            plt.savefig('acrobot.png')

        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

    env.close()

if __name__ == '__main__':
    run(is_training=False, render=True, num_episodes=500)