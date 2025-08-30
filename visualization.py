import pybullet as p
import pybullet_data
import numpy as np
import time
data1 = np.load("results/result50000.npy")
data = []
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")  # optionally, set the path to the pybullet data
robot = p.loadURDF("humanoid_torso.urdf", [0,0,1.3])
joint_name = []
joint_id = {}
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    name = info[1].decode("utf-8")
    jointType = info[2]
    if (jointType == p.JOINT_REVOLUTE):
        joint_id[name] = i
        joint_name.append(name)
for j in range(len(data1)):
    tmp = {}
    for i in range(len(joint_name)):
        tmp[joint_name[i]] = data1[j][i]
    data.append(tmp)
p.setGravity(0,0,-9.8)
p.setTimeStep(1./240.)
p.setRealTimeSimulation(0)
step=0
stage_num = 0
steps = 0
print(data)
for stage in range(3):
    frame_num = 0
    while frame_num<len(data):
        if step>=150:
            for j in joint_name:
                p.setJointMotorControl2(robot,joint_id[j],p.POSITION_CONTROL, data[frame_num][j], force = 50)

            frame_num+=1
            step = 0
        step+=1
        time.sleep(0.05)
        steps+=1
        p.stepSimulation()
    print(steps)