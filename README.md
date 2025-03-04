# Scalable-reinforcement-learning-approaches-for-dynamic-pricing-in-ride-hailing-systems
This project implements an enhanced version of the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm to optimize dynamic pricing in a ride-hailing system, inspired by the research in "Scalable reinforcement learning approaches for dynamic pricing in ride-hailing systems." Using the dynamic_pricing.csv dataset, the implementation simulates a ride-hailing environment where price multipliers are adjusted across different zones, times, and vehicle types to maximize platform profit. The code incorporates the paper’s novel mechanisms—incremental delayed policy updates and "forgetting" in the critic network—while adapting to the dataset’s constraints, such as the absence of origin-destination (OD) pairs. Structured for modularity and scalability, this repository organizes the environment, models, and TD3 agent into separate modules, making it easy to extend or deploy on GitHub for collaboration and further development.

The paper "Scalable reinforcement learning approaches for dynamic pricing in ride-hailing systems" by Zengxiang Lei and Satish V. Ukkusuri (2023 preprint) proposes advanced reinforcement learning (RL) solutions for dynamic pricing in ride-hailing platforms. It models the problem as a Markov Decision Process (MDP) with continuous state and action spaces, proving the existence of an optimal deterministic stationary policy. The authors enhance the TD3 algorithm with three innovations: incremental delayed policy updates, "forgetting" in the critic network, and permutation of OD features, improving stability and scalability. Tested on a 4x4 grid network and a 242-zone New York City network using 2013 taxi data, the approach demonstrates significant profit improvements over baselines, offering valuable insights for real-world ride-hailing operations.


## Setup
1. Clone the repository:
git clone https://github.com/wangqipeng/Scalable-reinforcement-learning-approaches-for-dynamic-pricing-in-ride-hailing-systems.git

    cd Scalable-reinforcement-learning-approaches-for-dynamic-pricing-in-ride-hailing-systems

3. Install dependencies
pip install -r requirements.txt

4. Place dynamic_pricing.csv in the data/ directory.

## Usage
Train the model:
python train.py

## Evaluate the trained model:
python evaluate.py

