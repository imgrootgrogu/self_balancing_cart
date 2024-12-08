import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt
import time


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

def reset_robot():
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


def calculate_reward(angle, angular_velocity):

    if abs(angle) < 0.05:  # Near upright
        return 5  # High reward for good balance
    elif abs(angle) < 0.1:  # Slight tilt
        return 1  # Small reward for moderate balance
    elif abs(angle) > 0.5:  # Large tilt
        return -10  # Heavy penalty for falling over
    else:
        return -abs(angle)  # Penalize proportionally to the tilt


# Define Actor Network
def build_actor(state_size, action_size, action_limit):
    inputs = tf.keras.Input(shape=(state_size,))
    x = tf.keras.layers.Dense(400, activation='relu')(inputs)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_size, activation='tanh')(x)
    outputs = tf.keras.layers.Lambda(lambda x: x * action_limit)(outputs)  
    model = tf.keras.Model(inputs, outputs)
    return model

# Define Critic Network
def build_critic(state_size, action_size):
    state_input = tf.keras.Input(shape=(state_size,))
    action_input = tf.keras.Input(shape=(action_size,))
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(400, activation='relu')(concat)
    x = tf.keras.layers.Dense(300, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model


state_size = 2 
action_size = 1  
action_limit = 2.0  
gamma = 0.99 
tau = 0.005 
actor_lr = 0.001
critic_lr = 0.002
batch_size = 64
memory_size = 20000

actor = build_actor(state_size, action_size, action_limit)
target_actor = build_actor(state_size, action_size, action_limit)
target_actor.set_weights(actor.get_weights())

critic = build_critic(state_size, action_size)
target_critic = build_critic(state_size, action_size)
target_critic.set_weights(critic.get_weights())


actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)


replay_buffer = deque(maxlen=memory_size)


def soft_update(target_model, source_model, tau):
    """Perform soft update of target network parameters."""
    target_weights = target_model.get_weights()
    source_weights = source_model.get_weights()
    updated_weights = [tau * sw + (1 - tau) * tw for tw, sw in zip(target_weights, source_weights)]
    target_model.set_weights(updated_weights)

class OUActionNoise:
    """Ornstein-Uhlenbeck Noise for exploration."""
    def __init__(self, mean, std_dev, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_dev
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x
        return x


noise = OUActionNoise(mean=np.zeros(action_size), std_dev=0.2 * np.ones(action_size))


def store_experience(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))
critic_losses = []
actor_losses = []

def train():
    if len(replay_buffer) < batch_size:
        return

    # Sample from replay buffer
    minibatch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    
    states = np.array(states).reshape(-1, state_size)
    actions = np.array(actions).reshape(-1, action_size)
    rewards = np.array(rewards).reshape(-1, 1)
    next_states = np.array(next_states).reshape(-1, state_size)
    dones = np.array(dones).reshape(-1, 1)

   
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    # Train critic
    next_actions = target_actor(next_states)
    target_q_values = target_critic([next_states, next_actions])
    targets = rewards + gamma * target_q_values * (1 - dones)
    with tf.GradientTape() as tape:
        q_values = critic([states, actions])
        critic_loss = tf.reduce_mean(tf.square(targets - q_values))
    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))
    critic_losses.append(critic_loss.numpy())

    # Train actor
    with tf.GradientTape() as tape:
        actions_pred = actor(states)
        actor_loss = -tf.reduce_mean(critic([states, actions_pred]))
    actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
    actor_losses.append(actor_loss.numpy())
    # Update target networks
    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)



episode_lengths = []
stability_scores = []

try:
    episode_rewards = []
    for episode in range(500):
        reset_robot()
        noise.reset()
        total_reward = 0
        steps = 0
        pitch_deviation = []

     
        state = np.reshape([p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot_id)[1])[1],
                            p.getBaseVelocity(robot_id)[1][1]], [1, state_size])

        for step in range(1000):
            # Select action with noise for exploration
            action = actor.predict(state)[0] + noise()
            action = np.clip(action, -action_limit, action_limit)

            # Apply action
            p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=action[0])
            p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=action[0])
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            # Get next state and reward
            next_state = np.reshape([p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot_id)[1])[1],
                                     p.getBaseVelocity(robot_id)[1][1]], [1, state_size])
            reward = calculate_reward(next_state[0][0], next_state[0][1])
            total_reward += reward
            pitch_deviation.append(abs(next_state[0][0]))  # Track pitch deviation
            steps += 1

            # Store experience and train
            done = abs(next_state[0][0]) > 0.5
            store_experience(state, action, reward, next_state, done)
            train()

            state = next_state
            if done:
                break

    
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        stability_scores.append(np.mean(pitch_deviation))  # Mean absolute pitch deviation

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, "
              f"Steps = {steps}, Stability Score = {stability_scores[-1]:.4f}")

    actor.save("actor_model_DDPG.h5")
    critic.save("critic_model.h5")


    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards, label="Episode Rewards", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Cumulative Rewards Over Episodes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths, label="Episode Length", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length Over Episodes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(stability_scores, label="Stability Score", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Mean Pitch Deviation")
    plt.title("Stability Score Over Episodes")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

finally:
    p.disconnect()
