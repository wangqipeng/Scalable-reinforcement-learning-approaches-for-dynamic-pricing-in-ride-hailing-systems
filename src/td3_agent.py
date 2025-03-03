import torch
import torch.optim as optim
from .models import Actor, Critic
from .replay_buffer import ReplayBuffer

class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device='cpu'):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.00005)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=0.0005)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=0.0005)
        
        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.device = device
        self.max_action = max_action
        self.tau = 0.01
        self.gamma = 0.99
        self.batch_size = 64

    def select_action(self, state, episode, max_episodes, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            noise = max(noise_scale * (1 - episode / max_episodes), 0.01)
            action = self.actor(state) + torch.randn(self.actor.net[-2].out_features).to(self.device) * noise
            action = torch.clamp(action, 0.33, self.max_action)
        return action.squeeze(0).cpu().numpy()

    def train(self, steps):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states = [x.to(self.device) for x in [states, actions, rewards, next_states]]

        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            noise = torch.randn_like(target_actions) * 0.2
            target_actions = torch.clamp(target_actions + noise, 0.33, self.max_action)
            q1_target = self.critic1_target(next_states, target_actions)
            q2_target = self.critic2_target(next_states, target_actions)
            q_target = rewards.unsqueeze(1) + self.gamma * torch.min(q1_target, q2_target)
        
        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = nn.MSELoss()(q1, q_target)
        critic2_loss = nn.MSELoss()(q2, q_target)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
        self.critic2_optimizer.step()
        
        policy_update_freq = min(steps / np.log(steps + 1), 10)
        if steps % int(policy_update_freq) == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # Soft update target networks
            for target, source in [(self.actor_target, self.actor), (self.critic1_target, self.critic1), (self.critic2_target, self.critic2)]:
                for target_param, param in zip(target.parameters(), source.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if steps % 500 == 0:
            self.critic1.net[-1].reset_parameters()
            self.critic2.net[-1].reset_parameters()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))