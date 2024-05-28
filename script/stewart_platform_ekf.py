#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import array, eye, asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from filterpy.common import Saver
import matplotlib.pyplot as plt


class StewartPlatformEKF():
    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.sphere_matrix_ = 0


        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()

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

    def update(self):
        T_sphere_to_world = self.get_transform('world','red_sphere')
                
        def Hx(x,n):
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
            eig = [1,1,1]
            T_camera_to_world = self.get_transform('world', 'camera_link') #trasformation matrix camera world 
            rW = T_camera_to_world[:3,3] #translation vector
            Rrw = T_camera_to_world[:2,:2] #rotation matrix
            p1 = Rrw.dot(eig)*(z0-b*(np.cos(alpha)**(2*n))) + ybar - rW #Rrw è la predicted camera rotation e rW è la predicted camera position, sono da considerare nulli???? In quale sr lavoro
            return p1[3] #solo la terza componente?

        def Jacobian(m):
            """ compute Jacobian"""
            alpha = m[0]
            tau= m[1]
            b = m[2]
            z0 = m[3]
            eig =[1,1,1]
            n = 3
            T_camera_to_world = self.get_transform('world', 'camera_link') #trasformation matrix camera world 
            rW = T_camera_to_world[:2,3] #translation vector
            Rrw = T_camera_to_world[:2,:2] #rotation matrix

        
            dalpha = Rrw.dot(eig) * (n * b * np.sin(alpha) * (np.cos(alpha))**(n - 1))
            dtau = [0,0,0]
            db =-Rrw.dot(eig)*(np.cos(alpha)**n)
            dz0 = Rrw.dot(eig)
            
            return np.transpose(np.array([dalpha,dtau,db,dz0]))

        rk = ExtendedKalmanFilter(dim_x=4, dim_z=1)
        t= rospy.get_time()#time
        z0 = 0.95 #exale position
        b = 3.09 #amplitude 
        tau = 32.38 #frequency
        n = 3
        alpha = np.pi*t/tau 

        #parametri vari
        rk.x = [alpha, tau, b, z0]
        
        rk.F = np.eye(4) + np.array(([[0, 1, 0,0],[0, 0, 0, 0],[0, 0, 0, 0],[0,0,0,0]]))*t
        Φ_tau = Φ_b = Φ_z0 = 1
        rk.Q = ([[Φ_tau  * (t**3) / 3, Φ_tau * (t**2) / 2, 0, 0],
                        [Φ_tau * (t**2) / 2, Φ_tau * t, 0, 0],
                        [0, 0, Φ_b * t, 0],
                        [0, 0, 0, Φ_z0 * t]])
        rk.R = np.diag(
            (np.random.normal(0,0.01,1))**2)#rumore bianco gaussiano (media,ampiezza,numero_elementi), in alternativa si può generale con filterpy
        rk.P = np.eye(4)

        rs = [] #misurazioni simulate z
        xs = [] #tutti i valori di x
        ps = [] #tutti i valori di P
        pos = [] #posizioni reali z

        s = Saver(rk)
        for i in range(200): #????
            T_sphere_to_camera = self.get_transform('camera_link','red_sphere')
            print(T_sphere_to_camera)
            z = T_sphere_to_camera[2,3] #solo spostamento lungo la z
            pos.append(z)

            rk.update(z, Jacobian, Hx)
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