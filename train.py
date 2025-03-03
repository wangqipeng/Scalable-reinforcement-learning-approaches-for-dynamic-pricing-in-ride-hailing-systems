import torch
from src.environment import RideHailingEnv
from src.td3_agent import TD3Agent

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RideHailingEnv("data/dynamic_pricing.csv")
    agent = TD3Agent(env.state_dim, env.action_dim, env.max_action, device)
    
    episodes = 1000
    steps = 0
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        rides_served_total = 0
        
        for t in range(100):
            action = agent.select_action(state, episode, episodes)
            next_state, reward, rides_served = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state)
            
            episode_reward += reward
            rides_served_total += rides_served
            state = next_state
            steps += 1
            
            agent.train(steps)
        
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Rides Served: {rides_served_total:.2f}")
    
    agent.save("actor_model.pth")

if __name__ == "__main__":
    train()