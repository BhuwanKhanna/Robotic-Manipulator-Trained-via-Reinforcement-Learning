import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math

class RoboticArmEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The agent controls a 7-DOF Kuka arm to locate a block, 'grasp' it 
    using a simulated vacuum gripper, and lift it to a target height.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(RoboticArmEnv, self).__init__()
        self.render_mode = render_mode
        self.max_steps = 200
        self.current_step = 0
        
        # Action space: dx, dy, dz (end effector movement), vacuum_command
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), 
                                       high=np.array([1, 1, 1, 1]), 
                                       dtype=np.float32)
        
        # Observation: EE(x,y,z), Obj(x,y,z), GripperAttached(1/0)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)

        self.physicsClient = None
        self.arm_id = None
        self.block_id = None
        self.target_z = 0.5
        self.ee_index = 6 # End effector link index for Kuka
        self.constraint_id = None

        if self.render_mode == 'human':
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        p.resetSimulation(physicsClientId=self.physicsClient)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClient)
        
        # Load plane and table
        p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)
        table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65], physicsClientId=self.physicsClient)
        
        # Load robotic arm (Kuka iiwa)
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.physicsClient)
        
        # Generate random object position
        obj_x = self.np_random.uniform(0.4, 0.6)
        obj_y = self.np_random.uniform(-0.2, 0.2)
        self.block_id = p.loadURDF("cube_small.urdf", basePosition=[obj_x, obj_y, 0.05], physicsClientId=self.physicsClient)
        
        self.constraint_id = None
        
        # Let physics settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physicsClient)
            
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # Current EE position
        ee_state = p.getLinkState(self.arm_id, self.ee_index, physicsClientId=self.physicsClient)
        ee_pos = list(ee_state[0])
        
        # Compute new target EE position based on action
        dx, dy, dz, vacuum_cmd = action
        target_pos = [
            ee_pos[0] + dx * 0.05,
            ee_pos[1] + dy * 0.05,
            ee_pos[2] + dz * 0.05
        ]
        
        # Clip target_z to avoid smashing into table
        target_pos[2] = max(0.01, target_pos[2])
        
        # IK to get joint angles
        joint_poses = p.calculateInverseKinematics(self.arm_id, self.ee_index, target_pos, physicsClientId=self.physicsClient)
        
        # Control joints
        for i in range(len(joint_poses)):
            p.setJointMotorControl2(bodyIndex=self.arm_id, 
                                    jointIndex=i, 
                                    controlMode=p.POSITION_CONTROL, 
                                    targetPosition=joint_poses[i],
                                    force=500,
                                    physicsClientId=self.physicsClient)
            
        # Simulate vacuum gripper
        obj_pos, _ = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.physicsClient)
        dist = np.linalg.norm(np.array(target_pos) - np.array(obj_pos))
        
        if vacuum_cmd > 0 and dist < 0.1 and self.constraint_id is None:
            # Create constraint (attach)
            self.constraint_id = p.createConstraint(self.arm_id, self.ee_index, self.block_id, -1, 
                                                    p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                                    physicsClientId=self.physicsClient)
        elif vacuum_cmd <= 0 and self.constraint_id is not None:
            # Remove constraint (detach)
            p.removeConstraint(self.constraint_id, physicsClientId=self.physicsClient)
            self.constraint_id = None

        # Step simulation
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.physicsClient)
            
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward_and_done(obs)
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        ee_state = p.getLinkState(self.arm_id, self.ee_index, physicsClientId=self.physicsClient)
        ee_pos = ee_state[0]
        obj_pos, _ = p.getBasePositionAndOrientation(self.block_id, physicsClientId=self.physicsClient)
        attached = 1.0 if self.constraint_id is not None else 0.0
        
        return np.array([
            ee_pos[0], ee_pos[1], ee_pos[2],
            obj_pos[0], obj_pos[1], obj_pos[2],
            attached
        ], dtype=np.float32)

    def _compute_reward_and_done(self, obs):
        ee_pos = obs[0:3]
        obj_pos = obs[3:6]
        attached = obs[6]
        
        dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
        reward = -dist_to_obj * 0.1  # penalty for being far from object
        
        if attached == 1.0:
            reward += 1.0 # Bonus for picking it up
            # Check lifting
            if obj_pos[2] > self.target_z:
                reward += 10.0 # Big bonus for reaching target height
                return reward, True, False
                
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return reward, terminated, truncated

    def render(self):
        pass

    def close(self):
        p.disconnect(physicsClientId=self.physicsClient)
