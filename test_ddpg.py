import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
import time
import os

root_path = os.path.dirname(__file__)
urdf_path = os.path.join(root_path, "urdf", "robot4.urdf")

actor = tf.keras.models.load_model("actor_model_DDPG.h5")
print("Actor model loaded successfully!")

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)
robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])


right_wheel_joint = 1
left_wheel_joint = 2
p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.VELOCITY_CONTROL, force=0)


def reset_robot():
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(robot_id, [0, 0, 0], [0, 0, 0])


def calculate_reward(angle, angular_velocity):

    if abs(angle) < 0.05: 
        return 5 
    elif abs(angle) < 0.1:  
        return 1  
    elif abs(angle) > 0.5:  
        return -10  
    else:
        return -abs(angle) 

def run_robot():
    reset_robot()
    total_reward = 0
    steps = 0

    state_size = 2  
    action_limit = 2.0

    
    state = np.reshape([p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot_id)[1])[1],
                        p.getBaseVelocity(robot_id)[1][1]], [1, state_size])

    for step in range(1000): 
       
        action = actor.predict(state)[0]
        action = np.clip(action, -action_limit, action_limit)

       
        p.setJointMotorControl2(robot_id, right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=action[0])
        p.setJointMotorControl2(robot_id, left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=action[0])
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

        next_state = np.reshape([p.getEulerFromQuaternion(p.getBasePositionAndOrientation(robot_id)[1])[1],
                                 p.getBaseVelocity(robot_id)[1][1]], [1, state_size])

        reward = calculate_reward(next_state[0][0], next_state[0][1])
        total_reward += reward

        state = next_state
        steps += 1

        if abs(state[0][0]) > 0.5:
            print("Robot fell over!")
            break

    print(f"Run completed. Total reward: {total_reward}, Steps: {steps}")

run_robot()
