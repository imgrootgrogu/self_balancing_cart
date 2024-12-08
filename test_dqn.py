# -*- coding: utf-8 -*-
import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.losses import MeanSquaredError
import os

root_path = os.path.dirname(__file__)
urdf_path = os.path.join(root_path, "urdf", "robot4.urdf")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])


right_wheel_joint = 1
left_wheel_joint = 2
p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)



model_path = "dqn_self_balancing_robot.h5" 
policy_net = tf.keras.models.load_model(model_path, custom_objects={"mse": MeanSquaredError()})
print(f"Loaded model from {model_path}")



state_size = 2  # [pitch, angular_velocity]
action_size = 3  # [-1, 0, +1]
total_episodes = 10  
max_steps = 1000  


def reset_robot():
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


try:
    for episode in range(total_episodes):
        reset_robot()
        total_reward = 0
        stable_steps = 0
        total_steps = 0

        
        orientation = p.getBasePositionAndOrientation(robot_id)[1]
        euler = p.getEulerFromQuaternion(orientation)
        pitch = euler[1]
        angular_velocity = p.getBaseVelocity(robot_id)[1][1]
        state = np.reshape([pitch, angular_velocity], [1, state_size])

        for step in range(max_steps):
            total_steps += 1

            
            q_values = policy_net.predict(state)
            action = np.argmax(q_values[0])  # Choose the action with the highest Q-value

            # Apply the action
            force = (action - 1) * 1  # Convert action index to torque
            p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)
            p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=force)

            # Step the simulation
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            # Get next state
            orientation = p.getBasePositionAndOrientation(robot_id)[1]
            euler = p.getEulerFromQuaternion(orientation)
            pitch = euler[1]
            angular_velocity = p.getBaseVelocity(robot_id)[1][1]
            next_state = np.reshape([pitch, angular_velocity], [1, state_size])

            # Calculate reward and track stability
            reward = 5 if abs(pitch) <= 0.8727 else -10  
            total_reward += reward
            if abs(pitch) <= 0.05:
                stable_steps += 1

           
            state = next_state

       
        stability_score = stable_steps / total_steps if total_steps > 0 else 0
        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Stability Score = {stability_score:.2f}")

finally:
    p.disconnect()
