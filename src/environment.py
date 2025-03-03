# src/environment.py
import numpy as np
import pandas as pd

class RideHailingEnv:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.states = self.data.groupby(["Location_Category", "Time_of_Booking", "Vehicle_Type"])[["Number_of_Riders", "Number_of_Drivers"]].mean().reset_index()
        self.state_dim = 2 * len(self.states)  # 48 (2 metrics x 3 zones x 4 times x 2 vehicles)
        self.action_dim = len(self.states)     # 24
        self.min_action, self.max_action = 0.33, 3.0
        self.state = None
        self.reset()

    def reset(self):
        self.state = self.states[["Number_of_Riders", "Number_of_Drivers"]].values.flatten()
        return self.state

    def step(self, action):
        state_df = pd.DataFrame(self.state.reshape(-1, 2), columns=["Number_of_Riders", "Number_of_Drivers"])
        price_multipliers = np.clip(action, self.min_action, self.max_action)
        
        # Demand and supply dynamics
        demand_factor = np.exp(-0.2 * (price_multipliers - 1))
        new_riders = np.maximum(state_df["Number_of_Riders"] * demand_factor, state_df["Number_of_Riders"] * 0.5)
        supply_factor = 1 + 0.5 * (price_multipliers - 1)
        new_drivers = np.maximum(state_df["Number_of_Drivers"] * supply_factor, state_df["Number_of_Drivers"] * 0.5)
        
        # Reward calculation
        rides_served = np.minimum(new_riders, new_drivers)
        revenue = (price_multipliers * self.data["Historical_Cost_of_Ride"].mean() * rides_served).sum()
        cost = (0.75 * revenue + self.data["Expected_Ride_Duration"].mean() * 0.01 * rides_served).sum()
        reward = (revenue - cost) / 1000 + rides_served.sum() * 0.1
        
        # Next state
        next_state = np.vstack([new_riders, new_drivers]).T.flatten()
        self.state = next_state
        return next_state, reward, rides_served.sum()