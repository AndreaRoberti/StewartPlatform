#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
import math
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
import matplotlib.pyplot as plt
from StewartPlatform import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient




class StewartPlatformEKF():
    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        self.centroid_topic_name_ = rospy.get_param('~centroid_topic_name','/output/centroid_pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.centroid_sub_ = rospy.Subscriber(self.centroid_topic_name_, PoseStamped, self.centroid_callback)
        self.sphere_matrix_ = 0

        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

        self.rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)
        self.tempo_ = []
        self.xs_ = []  # Tutti i valori di x (stati calcolati)
        self.pos_ = []  # Posizioni reali z 
        self.pose_points_ = []
        self.centroid_pose_= PoseStamped()

        client = RemoteAPIClient()
        self.stewart_platform = StewartPlatform(client)
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()


        # Ingressi 
        self.A_ = 500 #ampiezza
        self.phi_ = 0 #fase
      
    def pose_callback(self, msg):
        self.sphere_pose_ = msg
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x, self.sphere_pose_.pose.orientation.y, self.sphere_pose_.pose.orientation.z, self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix([self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)
    
    def centroid_callback(self, msg):
        self.centroid_pose_= msg


    def plotData(self):
        self.xs_ = asarray(self.xs_)
        self.pos_ = asarray(self.pos_)
        plt.plot(range(len(self.xs_)), self.xs_, label='EKF', color='b', marker='o')
        plt.plot(range(len(self.xs_)), self.pos_, label='Real position', color='r', marker='x')
        plt.xlabel('Numero misurazioni nel tempo')
        plt.ylabel('Posizione')
        plt.legend()
        plt.show()


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
        """function which takes as input the state variable (self.x) along
        with the optional arguments in hx_args, and returns the measurement
        that would correspond to that state. """

        return m[0] #l'unica misurazione che abbiamo è sulla posizione che è il primo elemento del vettore dellle variabili di stato
    
    def Jacobian(self, m):
        """function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H."""
        H = np.array([[1, 0, 0]])
        return H
    
    def update(self):
        t = rospy.get_time() - self.start_t_
        self.stewart_platform.respiration_ik(t)
        self.sim.step()

        omega = self.stewart_platform.alpha_*t+ self.stewart_platform.beta_
        #state vector
        x1 =  self.A_ * np.sin(omega*t+self.phi_)
        x2 = self.A_ *omega* np.cos(omega*t+self.phi_)
        x3 = omega
        #x3 = self.omega_
        x = np.array([x1,x2,x3])

        #derivatives
        omegadot = self.stewart_platform.alpha_
        omegadotdot = 0

        self.rk.x = x #state vector
        #self.rk.F = np.array([[np.cos(self.omega_*t),np.sin(self.omega_*t)/self.omega_,0],[-self.omega_*np.sin(self.omega_*t), np.cos(self.omega_*t),0],[0,0,1]])#state transition matrix
        self.rk.F = np.array([[0,1,0],[-(t*omegadot+x3)**2, (t*omegadotdot+omegadot)/(t*omegadot+x3), -x1*(2*x3+2*t*omegadot)-(x2*(t*omegadotdot+omegadot)/((t*omegadot+omega)**2))],[0,0,0]])
        sigma1 = 1
        sigma2 = 1
        sigma3 = 1
        self.rk.Q = np.array([[sigma1*(t**2), 0,0],[0,sigma2,0],[0,0,sigma3]]) #process noise covariance matrix -> errore sul modello
        noise_variance =0.001
        noise= np.random.normal(loc=0, scale=np.sqrt(noise_variance))
        self.rk.R = np.array([[noise_variance]])#measurement noise covariance matrix ->errore sulla misurazione

        for i in range(10):

            z = self.centroid_pose_.pose.position.z + noise           
            self.rk.predict()
            self.rk.update(z, self.Jacobian, self.Hx)

        self.pos_.append(self.centroid_pose_.pose.position.z)
        #print(self.rk.x[0])
        self.xs_.append(self.rk.x[0])

    def stopSimulation(self):
        self.sim.stopSimulation()




def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')

    stewart_platform_ekf = StewartPlatformEKF()
    
    # rate = 150# Hz
    # ros_rate = rospy.Rate(rate)

    while (t := stewart_platform_ekf.sim.getSimulationTime()) < 10:
        stewart_platform_ekf.update()
    

    # while not rospy.is_shutdown():
    #     stewart_platform_ekf.update()
    #     ros_rate.sleep()

    
    stewart_platform_ekf.stopSimulation()
    stewart_platform_ekf.plotData()
    
    # stewart_platform_ekf.plotData()  # CREA LA FUNZIONE E LA CHIAMI QUA

if __name__ == '__main__':
    main()
