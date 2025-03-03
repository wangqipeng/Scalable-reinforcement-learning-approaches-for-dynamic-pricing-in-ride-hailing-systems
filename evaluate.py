import torch
from src.environment import RideHailingEnv
from src.td3_agent import TD3Agent

def evaluate(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RideHailingEnv("data/dynamic_pricing.csv")
    agent = TD3Agent(env.state_dim, env.action_dim, env.max_action, device)
    agent.load(model_path)
    
    state = env.reset()
    total_reward = 0
    total_rides = 0
    
    for _ in range(100):
        action = agent.select_action(state, episode=999, max_episodes=1000, noise_scale=0)  # No noise
        next_state, reward, rides_served = env.step(action)
        total_reward += reward
        total_rides += rides_served
        state = next_state
    
    print(f"Evaluation - Total Reward: {total_reward:.2f}, Total Rides Served: {total_rides:.2f}")

if __name__ == "__main__":
    evaluate("actor_model.pth")