# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

q_table = np.load("q_table.npy")
print("Q-table loaded successfully.")

urdf_path = r"C:\Users\lilil\OneDrive\Documents\school\MSML642 Robotics\self_balancing_robot\urdf\robot4.urdf"


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)
robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])

# Disable velocity control for the wheel joints
right_wheel_joint = 1
left_wheel_joint = 2
p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)

# Define test parameters
n_states = 20  # Discretized states
n_actions = 3  # [-1, 0, +1]
test_episodes = 10
test_steps = 1000
rewards = []
stability_scores = []

def discretize(value, bins):
    """Convert a continuous value into a discrete state."""
    return int(np.digitize(value, bins) - 1)


for episode in range(test_episodes):
    
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])
    
    total_reward = 0
    stable_steps = 0
    total_steps = 0
    
    # Get initial state
    orientation = p.getBasePositionAndOrientation(robot_id)[1]
    euler = p.getEulerFromQuaternion(orientation)
    pitch = euler[1]
    angular_velocity = p.getBaseVelocity(robot_id)[1][1]

    state_angle = discretize(pitch, np.linspace(-np.pi / 4, np.pi / 4, n_states))
    state_velocity = discretize(angular_velocity, np.linspace(-5, 5, n_states))
    
    for step in range(test_steps):
        total_steps += 1

        # Check stability 
        if abs(pitch) < 0.05:
            stable_steps += 1

        
        action = np.argmax(q_table[state_angle, state_velocity])  # Best action based on Q-table
        force = (action - 1) * 1  # Map action [0, 1, 2] to [-1, 0, +1]
        p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)
        p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
        orientation = p.getBasePositionAndOrientation(robot_id)[1]
        euler = p.getEulerFromQuaternion(orientation)
        pitch = euler[1]
        angular_velocity = p.getBaseVelocity(robot_id)[1][1]

      
        new_state_angle = discretize(pitch, np.linspace(-np.pi / 4, np.pi / 4, n_states))
        new_state_velocity = discretize(angular_velocity, np.linspace(-5, 5, n_states))

        reward = 5 if abs(pitch) < 0.05 else -abs(pitch)
        total_reward += reward

    
        state_angle = new_state_angle
        state_velocity = new_state_velocity
        if abs(pitch) > 0.5:
            print(f"Episode {episode + 1}: Robot fell after {step + 1} steps.")
            break

   
    rewards.append(total_reward)
    stability_scores.append(stable_steps / total_steps if total_steps > 0 else 0)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}, Stability Score = {stability_scores[-1]:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(rewards, label="Total Reward", color="blue")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Rewards During Testing")
plt.legend()
plt.grid()


plt.subplot(1, 2, 2)
plt.plot(stability_scores, label="Stability Score", color="green")
plt.xlabel("Episode")
plt.ylabel("Stability Score")
plt.title("Stability Scores During Testing")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

p.disconnect()
