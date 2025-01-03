![Self-Balancing Cart Example](cart.png)

This repository contains implementations of three reinforcement learning algorithms for balancing a robot in the PyBullet simulation environment. The available algorithms are:
- Q-Learning (`main_q.py`)
- Deep Q-Learning (DQN)** (`main_dqn.py`)
- Deep Deterministic Policy Gradient (DDPG) (`main_ddpg.py`)

### Prerequisites

Before running any of the algorithms, ensure that the following are installed on your system:
- Python
- TensorFlow (for `main_dqn.py` and `main_ddpg.py`)
- PyBullet
- NumPy
- Matplotlib

You can install the required Python packages using:
```bash
pip install tensorflow pybullet numpy matplotlib
```

### Simulation Files
- Robot URDF File: `robot4.urdf` 
- meshes: robot physics file
- Algorithm Scripts
  - `main_q.py`: Q-Learning algorithm
  - `main_dqn.py`: Deep Q-Learning algorithm
  - `main_ddpg.py`: Deep Deterministic Policy Gradient algorithm

---

### How to Run Each Algorithm

#### 1. Q-Learning (`main_q.py`)

This script implements a tabular Q-Learning approach to train the robot to balance. No neural networks are used.

Run the script: python main_q.py



Key Outputs:
- Rewards per episode are printed to the console.
- A plot of episode rewards is displayed at the end.

---

#### 2. Deep Q-Learning (`main_dqn.py`)

This script uses a neural network to approximate Q-values and trains the robot to balance using the DQN algorithm.

Run the script: python main_dqn.py


Key Features:
- Calculates the **stability score**: Fraction of steps within -50+ pitch angle.
- Saves the trained model as `dqn_self_balancing_robot.h5`.

- Key Outputs:
- Rewards, average loss, and stability score per episode are printed to the console.
- Plots:
  - Total rewards per episode
  - Average loss per episode
  - Stability score per episode

---

#### 3. Deep Deterministic Policy Gradient (DDPG) (`main_ddpg.py`)

This script trains the robot using the DDPG algorithm, which is designed for continuous action spaces. It trains an actor-critic model.

Run the script:

python main_ddpg.py


Key Features:
- Implements actor and critic networks for continuous control.
- Uses an Ornstein-Uhlenbeck process for exploration.
- Saves the trained actor model as `ddpg_actor_self_balancing_robot.h5`.

Key Outputs:
- Rewards per episode are printed to the console.
- Plots:
  - Total rewards per episode
  - Actor and critic losses over training steps
  - Stability score

---

### Robot File

The `robot4.urdf` file defines the robot's structure and physical properties.

Ensure this file is in the same directory as the Python scripts.

---

### Tips for Running the Scripts

1. Simulation Setup:
   - When running the scripts, a PyBullet GUI window will open. This allows you to observe the robot's behavior during training.
   - Press ESC to close the simulation window after the script finishes.

2. Training Time:
   - Training time depends on the algorithm:
     - Q-Learning: Faster, as it uses a tabular approach.
     - DQN and DDPG: Slower due to neural network training.

3. Model Reuse:
   - For `main_dqn.py` and `main_ddpg.py`, the trained models are saved. You can reuse them to skip training in future runs by loading the saved models.

---

### Directory Structure

/self_balancing_robot/
    ├── meshes/
    |── urdf/
        ├── robot4.urdf
    ├── main_q.py
    ├── main_dqn.py
    ├── main_ddpg.py
    ├── dqn_self_balancing_robot.h5   (Generated by main_dqn.py)
    ├── actor_model_DDPG.h5 (Generated by main_ddpg.py)
    ├── critic_model.h5 (Generated by main_ddpg.py)
    ├── q_table (Generated by main_q.py)
    ├── test_q.py (test q_table)
    ├── test_dqn (test dqn model)
    ├── test_ddpg (test actor_model_DDPG model)

### Test the trained models
1. Q-Learning:
Run script: python test_q.py

2. Deep Q-Learning:
Run script: python test_dqn.py

3. DDPG:
Run script: python test_ddpg.py
