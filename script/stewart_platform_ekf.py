#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import array, eye, asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from filterpy.common import Saver
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from StewartPlatform import *


class StewartPlatformEKF():
    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.sphere_matrix_ = 0


        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

    def pose_callback(self,msg):
        self.sphere_pose_ = msg
        #print(self.sphere_pose_)
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x,self.sphere_pose_.pose.orientation.y,self.sphere_pose_.pose.orientation.z,self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix( [self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)
        #print(self.sphere_matrix_)

    def get_transform(self, source_frame, target_frame):
        try:
            (trans,rot) = self.tf_listener_.lookupTransform(target_frame, # 'world'
                                       source_frame,  # 'red_sphere'
                                       rospy.Time(0)) #get the tf at first available time
            # print(trans)
            # print(rot)
            rot_matrix = tf.transformations.quaternion_matrix([rot[0],rot[1],rot[2],rot[3]])
            trasl_matrix = tf.transformations.translation_matrix( [trans[0], trans[1], trans[2]])
            T_matrix = np.dot(trasl_matrix, rot_matrix)
            #print(T_matrix)
            #print('-------')
            return T_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print('No transform available')
        
    def Hx(self, x):
        """ takes a state variable and returns the measurement
        that would correspond to that state.
        ybar è la posizione che ricavo dalla simulazione/ mean position of the feature in 3d space
        """
        T_sphere_to_camera = self.get_transform('camera_link','red_sphere')
        ybar = T_sphere_to_camera[:3,3]
        alpha = x[0]
        tau= x[1]
        b = x[2]
        z0 = x[3]
        eig = np.array([1,1,1])
        n=3
        T_camera_to_world = self.get_transform('world', 'camera_link') #trasformation matrix camera world 
        rW = T_camera_to_world[:3,3] #translation vector
        Rrw = T_camera_to_world[:3,:3] #rotation matrixù
        #print(Rrw.dot(eig))
        p1 = Rrw.dot(eig)*(z0-b*(np.cos(alpha)**(2*n))) + ybar -rW
        #print(np.shape(p1))
        return p1

    def Jacobian(self,m):
            """ compute Jacobian"""
            alpha = m[0]
            tau= m[1]
            b = m[2]
            z0 = m[3]
            eig =[1,1,1]
            n = 3
            T_camera_to_world = self.get_transform('world', 'camera_link') #trasformation matrix camera world 
            rW = T_camera_to_world[:3,3] #translation vector
            Rrw = T_camera_to_world[:3,:3] #rotation matrix

            
            dalpha = Rrw.dot(eig) * (n * b * np.sin(alpha) * (np.cos(alpha))**(n - 1))
            dtau = [0,0,0]
            db =-Rrw.dot(eig)*(np.cos(alpha)**n)
            dz0 = Rrw.dot(eig)
            
            H = np.array([dalpha,dtau,db,dz0])
            #print(H.T)
            return H.T 
            #return (np.array([dalpha,dtau,db,dz0]))   
    
    def update(self):
        #T_sphere_to_world = self.get_transform('world','red_sphere')
        #self.Hx(x,n)
        rk = ExtendedKalmanFilter(dim_x=4, dim_z=3)
        t= rospy.get_time() - self.start_t_ #time
        #print(t)
        z0 = 0.95 #exale position
        b = 3.09 #amplitude 
        tau = 32.38 #frequency
        n = 3
        alpha = np.pi*t/tau 

        #parametri vari
        rk.x = [alpha, tau, b, z0]
        rk.F = np.eye(4) + np.array(([[0, 1, 0,0],[0, 0, 0, 0],[0, 0, 0, 0],[0,0,0,0]]))*t
        #print(rk.F)
        phi_tau = phi_b = phi_z0 = 1
        rk.Q = ([[phi_tau  * (t**3) / 3, phi_tau * (t**2) / 2, 0, 0],
                        [phi_tau * (t**2) / 2, phi_tau * t, 0, 0],
                        [0, 0, phi_b * t, 0],
                        [0, 0, 0, phi_z0 * t]])
        rk.R = np.eye(3)*(np.random.normal(0,0.01,1)**2)#rumore bianco gaussiano (media,ampiezza,numero_elementi), in alternativa si può generale con filterpy
        

        rs = [] #misurazioni simulate z
        xs = [] #tutti i valori di x
        ps = [] #tutti i valori di P
        pos = [] #posizioni reali z

        s = Saver(rk)
        for i in range(20): #????
            #print(i)
            T_sphere_to_camera = self.get_transform('camera_link','red_sphere')
            if isinstance(T_sphere_to_camera, np.ndarray):
               # print(T_sphere_to_camera)
                z = T_sphere_to_camera[:3,3] #tutto il vettore di traslazione
                pos.append(z)
                #print('--------------------------------')
                #print(np.shape(np.subtract(z,self.Hx(rk.x))))
               
                #print('********************************')
                #print(self.Jacobian(rk.x))
                #print('-----------------------------')
                #print(np.shape(z))
                rk.update(z, self.Jacobian, self.Hx,R=rk.R)
                #print('update concluso')
                #print(rk.P)
                #rk.predict_update(z,self.Jacobian,self.Hx)
                ps.append(rk.P)
                rk.predict()
                xs.append(rk.x)
                rs.append(z)
                s.save()

        # vedi test mahalanobis, serve?
        # self.sphere_matrix_
        s.to_array()
        xs = asarray(xs)
        ps = asarray(ps)
        rs = asarray(rs)


#------------------------------------------------------------------

def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')

    stewart_platform_ekf = StewartPlatformEKF()
    
    rate = 100 # Hz
    ros_rate = rospy.Rate(rate)

    while not rospy.is_shutdown():
        stewart_platform_ekf.update()
        ros_rate.sleep()

if __name__ == '__main__':
    main()