import math
import numpy as np

class StewartPlatform():
    def __init__(self, client):
        self.sim_ = client.require('sim')
        self.simIK_ = client.require('simIK')

        self.motors_ = [self.sim_.getObject('./motor' + str(i)) for i in range(1, 7)]
        self.tips_ = [self.sim_.getObject('./downArm' + str(i) + 'Tip') for i in range(1, 6)]
        self.targets_ = [self.sim_.getObject('./downArm' + str(i) + 'Target') for i in range(1, 6)]
        self.base_ = self.sim_.getObject('./stewartPlatform')
        self.tip_ = self.sim_.getObject('./tip')
        self.target_ = self.sim_.getObject('./target')

        self.initTipMatrix_ = self.sim_.getObjectPose(self.tip_, self.base_)
        print(self.initTipMatrix_)
        self.z0_ = self.initTipMatrix_[2]

        self.ikEnv_ = self.simIK_.createEnvironment()
        self.ikGroup_ = self.simIK_.createGroup(self.ikEnv_)
        self.simToIkMapping = []
        for tip_obj, target_obj in zip(self.tips_, self.targets_):
            self.simToIkMapping.append(self.simIK_.addElementFromScene(self.ikEnv_, self.ikGroup_, self.base_, tip_obj, target_obj, self.simIK_.constraint_position))

        self.tipTask_ = self.simIK_.addElementFromScene(self.ikEnv_, self.ikGroup_, self.base_, self.tip_, self.target_, self.simIK_.constraint_pose)

    def set_fk(self):
        print(self.tipTask_)
        self.simIK_.setElementFlags(self.ikEnv_, self.ikGroup_, self.tipTask_, 0)
        for motor in self.motors_:
            self.simIK_.setJointMode(self.ikEnv_, simToIkMapping[motor], self.simIK_.jointmode_passive)

    def set_ik(self):
        self.simIK_.setElementFlags(self.ikEnv_, self.ikGroup_, self.tipTask_, 1)
        for motor in self.motors_:
            self.simIK_.setJointMode(self.ikEnv_, simToIkMapping[motor], self.simIK_.jointmode_ik)

    def pose_to_matrix(self, pose_list):
        position = np.array(pose_list[:3])
        quaternion = np.array(pose_list[3:])
        qw, qx, qy, qz = quaternion

        # Construct rotation matrix from quaternion
        rotation_matrix = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

        # Combine rotation matrix and position into a transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = position

        return transformation_matrix

    def matrix_to_pose(self, matrix):
        # Extract rotation matrix and position from the transformation matrix
        rotation_matrix = matrix[:3, :3]
        position = matrix[:3, 3]

        # Extract quaternion from rotation matrix
        sum_of_squares = (1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2])
        if sum_of_squares >= 0:
            qw = np.sqrt(sum_of_squares) / 2
        else:
            print("Warning: Negative sum of squares encountered.")
            qw = 1
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

        return [position[0], position[1], position[2], qx, qy, qz, qw]
        
    def example_ik(self, t):
       
        # Parameters for the function z(t)
        z0 = self.z0_
        b = 3.09
        n = 3
        tau = 1
        phi = 0

        # Compute z(t)
        z_t = z0-b * (np.cos(np.pi * (t / tau)- phi)**(2*n))
        zg = z_t
        o_r = b
        n_r = 0.2
        z_t = (((z_t-(z0-b))*n_r)/o_r)-0.1
        

        T_matrix = self.pose_to_matrix(self.initTipMatrix_)
        new_pose_matrix = T_matrix
        new_pose_matrix[2][3] = new_pose_matrix[2][3]+z_t  # Update z position according to z(t)
        new_pose = self.matrix_to_pose(new_pose_matrix)
        self.sim_.setObjectPose(self.target_, new_pose, self.base_)
        self.simIK_.handleGroup(self.ikEnv_, self.ikGroup_, {"syncWorlds": True})
