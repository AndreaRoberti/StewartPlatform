#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from StewartPlatform import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

class StewartPlatformEKF():

    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.sphere_matrix_ = 0

        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

        self.rk = ExtendedKalmanFilter(dim_x=1, dim_z=3)
        self.tempo_ = []
        self.xs_ = []  # Tutti i valori di x
        self.pos_ = []  # Posizioni reali z

        client = RemoteAPIClient()
        self.stewart_platform = StewartPlatform(client)
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()

        # Definisco i parametri costanti della classe
        self.z0_ = self.stewart_platform.z0_
        self.b_ = 1
        self.tau_= 1
        self.n_ = 3
      

    def pose_callback(self, msg):
        self.sphere_pose_ = msg
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x, self.sphere_pose_.pose.orientation.y, self.sphere_pose_.pose.orientation.z, self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix([self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)

    def get_transform(self, source_frame, target_frame):
        try:
            (trans, rot) = self.tf_listener_.lookupTransform(target_frame, source_frame, rospy.Time(0))
            rot_matrix = tf.transformations.quaternion_matrix([rot[0], rot[1], rot[2], rot[3]])
            trasl_matrix = tf.transformations.translation_matrix([trans[0], trans[1], trans[2]])
            T_matrix = np.dot(trasl_matrix, rot_matrix)
            return T_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print('No transform available %s - %s', source_frame, target_frame)

    def Hx(self, m):
        T_sphere_to_camera = self.get_transform('camera_link', 'red_sphere')
        ybar = T_sphere_to_camera[:3, 3]
        alpha = m[0]
        b = self.b_
        z0 = self.z0_
        n = self.n_
        eig = np.array([1, 1, 1])
        T_camera_to_world = self.get_transform('world', 'camera_link')
        rW = T_camera_to_world[:3,3] #translation vector
        Rrw = T_camera_to_world[:3,:3] #rotation matrix
        #rW = np.array([1, 1, 1])
        #Rrw = np.eye(3)
        p1 = Rrw.dot(eig * (z0 - b * (np.cos(alpha)**(2 * n))) + ybar - rW)
        return p1

    def add_measurement_noise(self, measurement, R):
        noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        return measurement + noise

    def Jacobian(self, m):
        alpha = m[0]
        b = self.b_
        n = self.n_
        eig = np.array([1, 1, 1])
        T_camera_to_world = self.get_transform('world', 'camera_link')
        #Rrw = np.eye(3)
        rW = T_camera_to_world[:3,3] #translation vector
        Rrw = T_camera_to_world[:3,:3] #rotation matrix
        H = np.zeros((3, 1))
        H[:, 0] = Rrw.dot(eig) * (-2 * n * b * np.sin(alpha) * (np.cos(alpha)**(2 * n - 1)))
        return H

    def update(self):
        t = rospy.get_time() - self.start_t_
        self.stewart_platform.respiration_ik(t)
        self.sim.step()

        alpha = np.pi * t / self.tau_
        self.rk.x = np.array([alpha]).T
        #self.rk.F = np.eye(1)
        self.rk.F = np.array(1+self.tau_*t/alpha)
        phi_alpha = 0.1
        self.rk.Q = np.array([[phi_alpha * t]])
        #self.rk.R = np.eye(3) * (np.random.normal(0, 0.001, 1)**2)
        self.rk.R = np.eye(3) * 1**2

        for i in range(250):
            T_sphere_to_camera = self.get_transform('camera_link', 'red_sphere')
            if isinstance(T_sphere_to_camera, np.ndarray):
                z = T_sphere_to_camera[:3, 3]
                self.rk.predict()
                #self.rk.update(z, self.Jacobian, self.Hx, R=self.rk.R)
                self.rk.update(z, self.Jacobian, self.Hx)

        #zz = self.get_transform('camera_link', 'red_sphere')
        zz = self.get_transform('red_sphere', 'world')
        if isinstance(zz, np.ndarray):
            zz = zz[:3, 3]
            self.pos_.append(zz)
            self.xs_.append(self.rk.x)

        if t > 20.00:
            self.xs_ = asarray(self.xs_)
            self.pos_ = asarray(self.pos_)
            print(self.xs_)
            print("***************")
            print(self.pos_)
            z2= self.z0_ - self.b_* ((np.cos(self.xs_[:, 0]))**(2 * self.n_))
            plt.plot(range(len(z2)), z2, label='EKF', color='b', marker='o')
            pos = self.pos_[:, 2]
            plt.plot(range(len(z2)), pos, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.legend()
            plt.show()
            rospy.signal_shutdown('Plot completato')

    def stopSimulation(self):
        self.sim.stopSimulation()

def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')

    stewart_platform_ekf = StewartPlatformEKF()
    
    rate = 100 # Hz
    ros_rate = rospy.Rate(rate)

    while not rospy.is_shutdown():
        stewart_platform_ekf.update()
        ros_rate.sleep()

    stewart_platform_ekf.stopSimulation()

if __name__ == '__main__':
    main()
