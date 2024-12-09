import pybullet as p
import pybullet_data
import time
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import os

root_path = os.path.dirname(__file__)
urdf_path = os.path.join(root_path, "urdf", "robot4.urdf")


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)
robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])

state_size = 2  
action_size = 3  
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
memory = deque(maxlen=2000)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')
    return model


policy_net = build_model()
target_net = build_model()
target_net.set_weights(policy_net.get_weights())  

right_wheel_joint = 1
left_wheel_joint = 2
p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)


def calculate_reward(angle, angular_velocity):
    if abs(angle) < 0.05:
        return 5
    elif abs(angle) < 0.1:
        return 1
    elif abs(angle) > 0.5:
        return -10
    else:
        return -abs(angle)
losses = []
def replay():
    """Train the policy network using experience replay."""
    if len(memory) < batch_size:
        return 0  # Return 0 loss if there's not enough data

    minibatch = random.sample(memory, batch_size)
    states, targets = [], []

    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(target_net.predict(next_state)[0])
        target_f = policy_net.predict(state)
        target_f[0][action] = target
        states.append(state[0])
        targets.append(target_f[0])

  
    history = policy_net.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
    return history.history['loss'][0]  


def reset_robot():
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


try:
    episode_rewards = []
    episode_losses = []
    stability_scores = []
    episode_lengths = []
    for episode in range(1000):
        reset_robot()
        total_reward = 0
        total_loss = 0
        loss_count = 0
        stable_steps = 0
        total_steps = 0
        done = False

        orientation = p.getBasePositionAndOrientation(robot_id)[1]
        euler = p.getEulerFromQuaternion(orientation)
        pitch = euler[1]
        angular_velocity = p.getBaseVelocity(robot_id)[1][1]
        state = np.reshape([pitch, angular_velocity], [1, state_size])

        for step in range(1000):
            total_steps += 1

            # Check stability
            if abs(pitch) <= 0.05:  
                stable_steps += 1

            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(action_size)
            else:
                q_values = policy_net.predict(state)
                action = np.argmax(q_values[0])

          
            force = (action - 1) * 1
            p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)
            p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

       
            orientation = p.getBasePositionAndOrientation(robot_id)[1]
            euler = p.getEulerFromQuaternion(orientation)
            pitch = euler[1]
            angular_velocity = p.getBaseVelocity(robot_id)[1][1]
            next_state = np.reshape([pitch, angular_velocity], [1, state_size])
            reward = calculate_reward(pitch, angular_velocity)
            total_reward += reward

            # Check if episode is done
            done = abs(pitch) > 0.5

            # Store experience in replay memory
            memory.append((state, action, reward, next_state, done))

           
            state = next_state

            if done:
                print(f"Episode {episode + 1} ended early due to large tilt.")
                break

    
        loss = replay()
        total_loss += loss
        loss_count += 1

        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        episode_losses.append(avg_loss)

        stability_score = stable_steps / total_steps if total_steps > 0 else 0
        stability_scores.append(stability_score)

        if episode % 10 == 0:
            target_net.set_weights(policy_net.get_weights())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_lengths.append(total_steps)
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Avg Loss = {avg_loss:.4f}, Stability Score = {stability_score:.4f}, Epsilon = {epsilon:.4f}")

#    policy_net.save("dqn_self_balancing_robot.h5")
#    print("Model saved to dqn_self_balancing_robot.h5")


    plt.figure(figsize=(12, 10))
    plt.subplot(4, 1, 1)
    plt.plot(episode_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards per Episode")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(episode_losses, label="Average Loss", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Average Loss per Episode")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(stability_scores, label="Stability Score", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Stability Score")
    plt.title("Stability Score per Episode")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(episode_lengths, label="Episode Length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Steps)")
    plt.title("Episode Length per Episode")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

finally:
    p.disconnect()
