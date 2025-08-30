import pybullet as p
import pybullet_data
import time
import numpy as np
import xml.etree.ElementTree as ET
import random
import copy
import matplotlib.pyplot as plt

num_of_simul = 15
num_of_stage = 3
prob_mut = 0.3
gen_num = 1000
front_angle = np.load("man_walking_front_angle.npy")
front_angle_name = ["right_hip_x", "left_hip_x", "right_shoulder1", "left_shoulder1"]
side_angle = np.load("man_walking_side_angle.npy")
side_angle_name = ["right_hip_y","left_hip_y","right_knee","left_knee","right_shoulder2","left_shoulder2","right_elbow","left_elbow"]
gene_len = len(front_angle)
head_offset_local = [0.0, 0.0, 0.19]

class PyBulletHumanoid:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")  # optionally, set the path to the pybullet data
        self.load_robot()
        self.get_joints()
        self.get_range()
        self.max_gen = []
        self.fixed_constraints = {}

    def load_robot(self):
        self.urdf_id = "humanoid_torso.urdf"
        self.robot = []
        for i in range(num_of_simul):
            self.robot.append(p.loadURDF(self.urdf_id, [0,2*i,1.3], p.getQuaternionFromEuler([0,0,0]), useFixedBase=False))

    def make_initial_gene(self):
        self.init_gene = []
        for i in range(gene_len):
            angle = {}
            for j in range(len(front_angle_name)):
                angle[front_angle_name[j]] = front_angle[i][j]
            for j in range(len(side_angle_name)):
                angle[side_angle_name[j]] = side_angle[i][j]
            for j in self.joint_names:
                if j not in front_angle_name and j not in side_angle_name:
                    angle[j] = 0.
            self.init_gene.append(angle)

    def get_joints(self):
        self.joint_names = []
        self.joint_id = {}
        for i in range(p.getNumJoints(self.robot[0])):
            info = p.getJointInfo(self.robot[0], i)
            name = info[1].decode("utf-8")
            jointType = info[2]
            if (jointType == p.JOINT_REVOLUTE):
                self.joint_names.append(name)
                self.joint_id[name] = i

    def get_range(self):
        tree = ET.parse(self.urdf_id)
        root = tree.getroot()
        self.joint_ranges = {}

        for joint in root.findall('.//joint'):
            name = joint.get('name')
            if name not in self.joint_names:
                continue
            limit = joint.find('limit')
            if name and limit is not None:
                lower = limit.get('lower')
                upper = limit.get('upper')
                if lower is not None and upper is not None:
                    self.joint_ranges[name] = (float(lower), float(upper))
    def make_gene_set(self):
        self.make_initial_gene()
        self.gene_set = []
        for i in range(num_of_simul):
            copied = copy.deepcopy(self.init_gene)
            self.gene_set.append(copied)

    def make_ran_gene(self):
        ran_gene = {}
        for i in self.joint_names:
            ran_gene[i] = random.uniform(self.joint_ranges[i][0], self.joint_ranges[i][1])
        return ran_gene


    def compute_fitness(self):
        self.fitness_set = []
        for i in self.robot:
            position, orientation = p.getBasePositionAndOrientation(i)
            fitness = position[0]
            self.fitness_set.append(fitness)
        self.max_gen.append(max(self.fitness_set))
        a = list(range(num_of_simul))
        paired = list(zip(a,self.fitness_set))
        sorted_pairs = sorted(paired, key = lambda paired: paired[1])
        sorted_elements = [a for a, k in sorted_pairs]
        print(self.fitness_set)
        print(sorted_elements)
        self.dad = self.gene_set[sorted_elements[-1]]
        self.mom = self.gene_set[sorted_elements[-2]]



    def mix(self):
        mixed = []
        b = False
        for j in range(gene_len):
            tmp = {}
            new = self.make_ran_gene()
            for i in self.joint_names:
                a = random.random()
                if 0 <= a < 0.5 - (prob_mut / 2):
                    tmp[i] = self.dad[j][i]
                elif 0.5 - (prob_mut / 2) <= a < 1 - prob_mut:
                    tmp[i] = self.mom[j][i]
                else:
                    tmp[i] = new[i]
                    b = True
            mixed.append(tmp)
        return mixed


    def make_new_gene_set(self):
        self.gene_set=[copy.deepcopy(self.dad), copy.deepcopy(self.mom)]
        for i in range(2,num_of_simul):
            new_gene = self.mix()
            self.gene_set.append(new_gene)

    def fix_robot(self, index):
        base_pos, _ = p.getBasePositionAndOrientation(self.robot[index])
        if index in self.fixed_constraints:
            return
        p.resetBaseVelocity(
            self.robot[index],
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        for i in range(3):
            p.stepSimulation()

        cid = p.createConstraint(
            parentBodyUniqueId=self.robot[index],
            parentLinkIndex=-1,  # 베이스 링크
            childBodyUniqueId=-1,  # 월드
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=base_pos
        )
        self.fixed_constraints[index] = cid

    def unfix_robot(self, index):
        if index not in self.fixed_constraints:
            return  # 이미 해제된 상태

        cid = self.fixed_constraints[index]
        p.removeConstraint(cid)
        del self.fixed_constraints[index]


    def unfix_all(self):
        """
        모든 로봇에 걸려 있는 고정 제약을 해제.
        - 세대가 바뀔 때나 시뮬레이션 시작 직전에 호출.
        """
        for idx in list(self.fixed_constraints.keys()):
            p.removeConstraint(self.fixed_constraints[idx])
        self.fixed_constraints.clear()


    def get_head(self):
        self.head = []
        for i in range(num_of_simul):
            base_pos, base_orn = p.getBasePositionAndOrientation(self.robot[i])
            head_offset_world = p.rotateVector(base_orn, head_offset_local)
            head_pos = [
                base_pos[0] + head_offset_world[0],
                base_pos[1] + head_offset_world[1],
                base_pos[2] + head_offset_world[2]
            ]
            self.head.append(head_pos)
    def reset_pos(self):
        for i in range(num_of_simul):
            p.resetBasePositionAndOrientation(
                self.robot[i],
                [0, 2 * i, 1.3],
                p.getQuaternionFromEuler([0, 0, 0])
            )
            p.resetBaseVelocity(self.robot[i],
                                linearVelocity=[0, 0, 0],
                                angularVelocity=[0, 0, 0])

            for j_name in self.joint_names:
                joint_index = self.joint_id[j_name]
                p.resetJointState(self.robot[i],
                                  joint_index,
                                  targetValue=0.0,
                                  targetVelocity=0.0)

        for i in range(num_of_simul):
            for j_name in self.joint_names:
                p.setJointMotorControl2(
                    bodyIndex=self.robot[i],
                    jointIndex=self.joint_id[j_name],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=500  # 필요에 따라 torque 조정
                )
        for _ in range(20):
            p.stepSimulation()
        time.sleep(0.1)
    def simulate(self):
        self.unfix_all()
        p.setGravity(0,0,-9.8)
        p.setTimeStep(1./240.)
        p.setRealTimeSimulation(0)
        stop_simul = set([])
        self.reset_pos()
        delay_step=50
        step=0
        stage_num = 0
        steps = 0
        for stage in range(num_of_stage):
            frame_num = 0
            while frame_num<gene_len:
                self.get_head()
                for i in range(num_of_simul):
                    if self.head[i][2] <= 1.0:
                        self.fix_robot(i)
                        stop_simul.add(i)
                        continue
                if step>=delay_step:

                    for i in range(num_of_simul):
                        if i in stop_simul:
                            continue
                        for j in self.joint_names:
                            p.setJointMotorControl2(self.robot[i],self.joint_id[j],p.POSITION_CONTROL, self.gene_set[i][frame_num][j], force = 50)

                    frame_num+=1
                    step = 0
                step+=1

                steps+=1
                if len(self.fixed_constraints) >= num_of_simul:
                    break
                p.stepSimulation()
            print(steps)

    def make_graph(self):
        plt.plot(self.max_gen)
        plt.show()

    def genetic_simulate(self):
        self.make_gene_set()
        for i in range(gen_num):
            self.simulate()
            self.compute_fitness()
            self.make_new_gene_set()
            print(self.max_gen[-1],self.dad[0],self.gene_set[0][0])
        l = []
        for i in range(gene_len):
            l.append(list(self.gene_set[0][i].values()))
        l = np.array(l)
        np.save("results/result50000.npy",l)
        self.make_graph()


if __name__ == "__main__":
    humanoid1 = PyBulletHumanoid()
    humanoid1.genetic_simulate()

  