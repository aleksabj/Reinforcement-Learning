import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# ---- Configuration ---- #
ENVIRONMENTS = ['Acrobot-v1', 'Pendulum-v1']
INPUT_DIMS = {
    'Acrobot-v1': 6,
    'Pendulum-v1': 3
}
OUTPUT_DIMS = {
    'Acrobot-v1': 3,      # Discrete action
    'Pendulum-v1': 1      # Continuous action
}

CONFIG = {
    'Acrobot-v1': {
        'hidden_size': 16,
        'pop_size': 80,
        'mutation_rate': 0.05,
        'mutation_strength': 0.10,
        'generations': 100
    },
    'Pendulum-v1': {
        'hidden_size': 32,
        'pop_size': 60,
        'mutation_rate': 0.10,
        'mutation_strength': 0.20,
        'generations': 150
    }
}

# ---- Neural Network Utilities ---- #
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def build_network(input_size, hidden_size, output_size, weights):
    i_h_size = input_size * hidden_size
    h_o_size = hidden_size * output_size
    bias_h_size = hidden_size
    bias_o_size = output_size

    i_h = weights[:i_h_size].reshape((input_size, hidden_size))
    h_o = weights[i_h_size:i_h_size+h_o_size].reshape((hidden_size, output_size))
    bias_h = weights[i_h_size+h_o_size:i_h_size+h_o_size+bias_h_size]
    bias_o = weights[-bias_o_size:]

    def net(x):
        h = relu(np.dot(x, i_h) + bias_h)
        out = np.dot(h, h_o) + bias_o
        return out
    return net

def weight_count(input_size, hidden_size, output_size):
    return input_size * hidden_size + hidden_size * output_size + hidden_size + output_size

# ---- Evaluation ---- #
def evaluate(env_name, weights, seed, hidden_size):
    env = gym.make(env_name)
    input_size = INPUT_DIMS[env_name]
    output_size = OUTPUT_DIMS[env_name]

    net = build_network(input_size, hidden_size, output_size, weights)

    total_reward = 0
    for _ in range(3):
        obs, _ = env.reset(seed=seed)
        ep_reward = 0
        for _ in range(500):
            action = net(obs)
            if env_name == 'Acrobot-v1':
                action = np.argmax(softmax(action))
            elif env_name == 'Pendulum-v1':
                action = np.clip(np.tanh(action) * 2, -2, 2)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        total_reward += ep_reward

    env.close()
    return total_reward / 3

# ---- Evolutionary Algorithm ---- #
def evolve(env_name, seed, log_file, config):
    np.random.seed(seed)

    input_size = INPUT_DIMS[env_name]
    output_size = OUTPUT_DIMS[env_name]
    hidden_size = config['hidden_size']
    pop_size = config['pop_size']
    mutation_rate = config['mutation_rate']
    mutation_strength = config['mutation_strength']
    generations = config['generations']
    elite_size = max(1, int(0.05 * pop_size))

    num_weights = weight_count(input_size, hidden_size, output_size)
    population = np.random.randn(pop_size, num_weights)
    best_rewards = []

    for gen in range(generations):
        rewards = np.array([evaluate(env_name, ind, seed + i, hidden_size) for i, ind in enumerate(population)])
        elite_indices = rewards.argsort()[-elite_size:]
        elites = population[elite_indices]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent = elites[np.random.randint(elite_size)]
            child = parent + mutation_strength * np.random.randn(num_weights) * (np.random.rand(num_weights) < mutation_rate)
            new_population = np.vstack([new_population, child])

        population = new_population[:pop_size]
        best = np.max(rewards)
        best_rewards.append(best)
        print(f"Gen {gen}: Best Reward = {best}")
        log_file.write(f"Gen {gen}: Best Reward = {best}\n")

    return best_rewards

# ---- Main and CLI ---- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True, choices=ENVIRONMENTS)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    config = CONFIG[args.env]

    all_runs = []
    seeds = [0, 1, 2, 3, 4]
    with open(f"results/log_{args.env}.txt", "w") as log_file:
        for seed in seeds:
            log_file.write(f"--- Seed {seed} ---\n")
            print(f"--- Seed {seed} ---")
            rewards = evolve(args.env, seed, log_file, config)
            all_runs.append(rewards)

    all_runs = np.array(all_runs)
    mean_rewards = all_runs.mean(axis=0)
    std_rewards = all_runs.std(axis=0)

    plt.figure()
    plt.title(f"Performance on {args.env}")
    plt.plot(mean_rewards, label='Mean Reward')
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
    plt.xlabel("Generation")
    plt.ylabel("Best Individual Reward")
    plt.legend()
    plt.savefig(f"results/{args.env}_performance.png")
    print(f"Results saved to results/{args.env}_performance.png")

if __name__ == '__main__':
    main()
