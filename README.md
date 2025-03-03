# Scalable-reinforcement-learning-approaches-for-dynamic-pricing-in-ride-hailing-systems
Dynamic pricing is a cornerstone strategy for ride-hailing platforms like Uber and Lyft, enabling them to balance the supply of drivers with the demand from riders in real time. The paper "Scalable reinforcement learning approaches for dynamic pricing in ride-hailing systems" by Zengxiang Lei and Satish V. Ukkusuri, published as a preprint in 2023, tackles this challenge by proposing innovative reinforcement learning (RL) solutions. The authors formulate dynamic pricing as a Markov Decision Process (MDP) with continuous state and action spaces and prove the existence of an optimal deterministic stationary policy. They enhance the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm—a state-of-the-art RL method for continuous control—by introducing three novel mechanisms: incremental delayed policy updates, "forgetting" in the critic network, and permutation of origin-destination (OD) features. These enhancements improve stability and scalability, making the approach suitable for large-scale ride-hailing systems. The paper demonstrates its effectiveness through experiments on a 4x4 grid network and a realistic New York City network with 242 zones, using 2013 NYC taxi data. This project implements the enhanced TD3 algorithm with the dynamic_pricing.csv dataset, adapting the paper’s methodology to optimize price multipliers and maximize profit in a simulated ride-hailing environment.

## Setup

1.Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dynamic_pricing_td3.git
   cd Scalable-reinforcement-learning-approaches-for-dynamic-pricing-in-ride-hailing-systems

2 Install dependencies
pip install -r requirements.txt

3. Place dynamic_pricing.csv in the data/ directory.

## Usage
Train the model:
python train.py

## valuate the trained model:
python evaluate.py

