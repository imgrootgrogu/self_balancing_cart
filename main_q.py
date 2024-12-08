import pybullet as p
import pybullet_data
import time  
import numpy as np
import matplotlib.pyplot as plt

urdf_path = r"C:\Users\lilil\OneDrive\Documents\school\MSML642 Robotics\self_balancing_robot\urdf\robot4.urdf"

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)  
robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])

n_states = 20  # Discretized states
n_actions = 3  # [-1, 0, +1]
q_table = np.zeros((n_states, n_states, n_actions))
alpha = 0.1 
gamma = 0.9 
epsilon = 1.0  
epsilon_decay = 0.995
p.setTimeStep(1.0 / 240.0)  

# Disable velocity control for the wheel joints
right_wheel_joint = 1
left_wheel_joint = 2
p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)

def discretize(value, bins):
    """Convert a continuous value into a discrete state."""
    return int(np.digitize(value, bins) - 1)


def calculate_reward(angle, angular_velocity):
    """Reward function for balancing."""
    if abs(angle) < 0.05:  # Very small tilt angle
        return 5
    elif abs(angle) < 0.1:  # Small tilt angle
        return 1
    elif abs(angle) > 0.5:  # Large tilt angle
        return -10
    else:
        return -abs(angle)  # Penalize based on tilt

try:
    
    episode_rewards = []
    episode_losses = []  
    stability_scores = []  

    for episode in range(500):
      
        total_reward = 0
        total_loss = 0
        stable_steps = 0
        total_steps = 0
        orientation = p.getBasePositionAndOrientation(robot_id)[1]
        euler = p.getEulerFromQuaternion(orientation)
        pitch = euler[1]
        angular_velocity = p.getBaseVelocity(robot_id)[1][1]

        state_angle = discretize(pitch, np.linspace(-np.pi / 4, np.pi / 4, n_states))
        state_velocity = discretize(angular_velocity, np.linspace(-5, 5, n_states))

        for step in range(1000):  
            total_steps += 1

            # Check stability
            if abs(pitch) < 0.05:
                stable_steps += 1

            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(q_table[state_angle, state_velocity])

           
            force = (action - 1) * 1  # Reduced force for finer control
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
            reward = calculate_reward(pitch, angular_velocity)
            target = reward + gamma * np.max(q_table[new_state_angle, new_state_velocity])
            loss = abs(target - q_table[state_angle, state_velocity, action])
            q_table[state_angle, state_velocity, action] += alpha * (target - q_table[state_angle, state_velocity, action])
            total_loss += loss

            total_reward += reward
            state_angle = new_state_angle
            state_velocity = new_state_velocity

        
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / total_steps)  
        stability_scores.append(stable_steps / total_steps)  # Fraction of stable steps

        # Decay epsilon
        epsilon *= epsilon_decay
        print(f"Episode {episode + 1} completed. Epsilon: {epsilon:.4f}. Total Reward: {total_reward:.2f}, Avg Loss: {total_loss / total_steps:.4f}, Stability Score: {stability_scores[-1]:.4f}")


    # np.save("q_table.npy", q_table)
    # print("Q-table saved to q_table.npy")

 
    plt.figure(figsize=(18, 6))
    plt.subplot(3, 1, 1)
    plt.plot(episode_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(episode_losses, label="Average Loss", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Average Loss per Episode")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(stability_scores, label="Stability Score", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Stability Score")
    plt.title("Stability Score per Episode")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

finally:
    p.disconnect()
    print("Simulation closed cleanly.")
