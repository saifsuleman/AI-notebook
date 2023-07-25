import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium
import time 

device = torch.device("cpu")

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return dict(
            states=np.array(self.states),
            actions=np.array(self.actions),
            probs=np.array(self.probs),
            vals=np.array(self.vals),
            rewards=np.array(self.rewards),
            dones=np.array(self.dones),
            batches=batches
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, checkpoint_dir="checkpoints"):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, "actor_ppo")

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims, device=device),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        ).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, checkpoint_dir="checkpoints"):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(checkpoint_dir, "critic_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        ).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        return self.critic(state)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class PPOAgent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("... saving models")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print("... loading models")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float32).to(device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            generated_batches = self.memory.generate_batches()

            states_arr = generated_batches["states"]
            actions_arr = generated_batches["actions"]
            probs_arr = generated_batches["probs"]
            vals_arr = generated_batches["vals"]
            rewards_arr = generated_batches["rewards"]
            dones_arr = generated_batches["dones"]
            batches = generated_batches["batches"]

            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0

                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.gamma * vals_arr[k+1] * (1 - int(dones_arr[k])) - vals_arr[k])
                    discount *= self.gamma * self.gae_lambda
                
                advantage[t] = a_t
            
            advantage = torch.tensor(advantage).to(device)

            values = torch.tensor(vals_arr, dtype=torch.float32).to(device)
            for batch in batches:
                states = torch.tensor(states_arr[batch], dtype=torch.float32).to(device)
                old_probs = torch.tensor(probs_arr[batch], dtype=torch.float32).to(device)
                actions = torch.tensor(actions_arr[batch], dtype=torch.float32).to(device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                ret = advantage[batch] + values[batch]
                critic_loss = ((ret - critic_value) ** 2).mean() # MSE

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

play = True
if __name__ == "__main__":
    render_mode = None
    if play:
        render_mode = "human"
    env = gymnasium.make("CartPole-v0", render_mode=render_mode)

    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003

    agent = PPOAgent(
        n_actions=env.action_space.n,
        batch_size=batch_size,
        alpha=alpha,
        n_epochs=n_epochs,
        input_dims=env.observation_space.shape,
    )

    if play:
        agent.load_models()

    n_games = 500

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            action, prob, val = agent.choose_action(observation)
            obs, reward, done, info, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = obs

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)

