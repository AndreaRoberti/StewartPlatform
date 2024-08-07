#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray, Pose
import matplotlib.pyplot as plt
from StewartPlatform import *
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class StewartPlatformEKF():
    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        self.array_topic_name_ = rospy.get_param('~array_topic_name','/output/pose_array')
        self.centroid_topic_name_ = rospy.get_param('~centroid_topic_name','/output/centroid_pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.pose_array_sub_ = rospy.Subscriber(self.array_topic_name_, PoseArray, self.pose_array_callback)
        self.centroid_sub_ = rospy.Subscriber(self.centroid_topic_name_, PoseStamped, self.centroid_callback)
        self.sphere_matrix_ = 0

        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

        self.rk = ExtendedKalmanFilter(dim_x=6, dim_z=3)
        self.tempo_ = []
        self.xsx_ = []  # Tutti i valori di x (stati calcolati)
        self.xsy_ = []  # Tutti i valori di y (stati calcolati)
        self.xsz_ = []  # Tutti i valori di z (stati calcolati)
        self.posx_ = []  # Posizioni reali x 
        self.posy_ = []  # Posizioni reali y
        self.posz_ = []  # Posizioni reali z
        self.pose_array_ = PoseArray()
        self.pose_points_ =[]
        self.sphere_pose_ = PoseStamped()
        self.pose_points_ = []
        self.centroid_pose_= PoseStamped()

        client = RemoteAPIClient()
        self.stewart_platform = StewartPlatform(client)
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()

        # Ingressi 
        self.Ax_ = 1 #ampiezza
        self.omegax_ = 1 #frequenza
        self.phix_ = 0 #fase
        self.Ay_ = 1 #ampiezza
        self.omegay_ = 1 #frequenza
        self.phiy_ = 0 #fase
        self.Az_ = 1 #ampiezza
        self.omegaz_ = 1 #frequenza
        self.phiz_ = 0 #fase
      
    def pose_array_callback(self,msg):
        self.pose_array_ = msg
        self.pose_points_ = np.zeros((len(self.pose_array_.poses), 1))
        for i in range(len(self.pose_array_.poses)):
            self.pose_points_[i] = np.array([self.pose_array_.poses[i].position.z])

        #self.pose_points_ = np.array([self.pose_array_.pose.position.x,self.pose_array_.pose.position.y,self.pose_array_.pose.position.z])

        
    def pose_callback(self, msg):
        self.sphere_pose_ = msg
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x, self.sphere_pose_.pose.orientation.y, self.sphere_pose_.pose.orientation.z, self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix([self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)

    def centroid_callback(self, msg):
        self.centroid_pose_= msg

    def plotData(self):
        self.xsx_ = asarray(self.xsx_)
        self.xsy_ = asarray(self.xsy_)
        self.xsz_ = asarray(self.xsz_)
        self.posx_ = asarray(self.posx_)
        self.posy_ = asarray(self.posy_)
        self.posz_ = asarray(self.posz_)

        # Primo grafico per x
        plt.figure()
        plt.plot(range(len(self.xsz_)), self.xsx_, label='EKF x', color='b', marker='o')
        plt.plot(range(len(self.xsz_)), self.posx_, label='Real position', color='r', marker='x')
        plt.xlabel('Numero misurazioni nel tempo')
        plt.ylabel('Posizione')
        plt.title('Confronto EKF vs Posizione Reale (x)')
        plt.legend()

        # Secondo grafico per y
        plt.figure()
        plt.plot(range(len(self.xsz_)), self.xsy_, label='EKF y', color='b', marker='o')
        plt.plot(range(len(self.xsz_)), self.posy_, label='Real position', color='r', marker='x')
        plt.xlabel('Numero misurazioni nel tempo')
        plt.ylabel('Posizione')
        plt.title('Confronto EKF vs Posizione Reale (y)')
        plt.legend()

        # Terzo grafico per z
        plt.figure()
        plt.plot(range(len(self.xsz_)), self.xsz_, label='EKF z', color='b', marker='o')
        plt.plot(range(len(self.xsz_)), self.posz_, label='Real position', color='r', marker='x')
        plt.xlabel('Numero misurazioni nel tempo')
        plt.ylabel('Posizione')
        plt.title('Confronto EKF vs Posizione Reale (z)')
        plt.legend()

        # Mostra tutti i grafici
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
        mes = np.array([m[0],m[2],m[4]])
        return mes
    
    def Jacobian(self, m):
        """function which computes the Jacobian of the H matrix (measurement
           function). Takes state variable (self.x) as input, along with the
           optional arguments in args, and returns H."""
        H = np.array([[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]])
        return H
    
    def update(self):
        t = rospy.get_time() - self.start_t_
        self.stewart_platform.example_ik(t)
        self.sim.step()

        #state vector
        x1 =  self.Ax_ * np.sin(self.omegax_*t+self.phix_)
        x2 = self.Ax_ *self.omegax_* np.cos(self.omegax_*t+self.phix_)
        x3 =  self.Ay_ * np.sin(self.omegay_*t+self.phiy_)
        x4 = self.Ay_ *self.omegay_* np.cos(self.omegay_*t+self.phiy_)
        x5 =  self.Az_ * np.sin(self.omegaz_*t+self.phiz_)
        x6 = self.Az_ *self.omegaz_* np.cos(self.omegaz_*t+self.phiz_)
        
        x = np.array([x1,x2,x3,x4,x5,x6])

        self.rk.x = x #state vector
        self.rk.F = np.array([[np.cos(self.omegax_ * t), np.sin(self.omegax_ * t) / self.omegax_, 0, 0, 0, 0],
                              [-self.omegax_ * np.sin(self.omegax_ * t), np.cos(self.omegax_ * t), 0, 0, 0, 0],
                              [0, 0, np.cos(self.omegay_ * t), np.sin(self.omegay_ * t) / self.omegay_, 0, 0],
                              [0, 0, -self.omegay_ * np.sin(self.omegay_ * t), np.cos(self.omegay_ * t), 0, 0],
                              [0, 0, 0, 0, np.cos(self.omegaz_ * t), np.sin(self.omegaz_ * t) / self.omegaz_],
                              [0, 0, 0, 0, -self.omegaz_ * np.sin(self.omegaz_ * t), np.cos(self.omegaz_ * t)]
])#state transition matrix
        sigma1 = 1
        sigma2 = 1
        sigma3= 1
        sigma4 = 1
        sigma5 = 1
        sigma6= 1

        self.rk.Q = np.array([[sigma1*(t**2),0,0,0,0,0],[0,sigma2,0,0,0,0],
                              [sigma3*(t**2),0,0,0,0,0],[0,sigma4,0,0,0,0],
                              [sigma5*(t**2),0,0,0,0,0],[0,sigma6,0,0,0,0]]) #process noise covariance matrix -> errore sul modello
        noise_variance = 0.001
        noise= np.random.normal(loc=0, scale=np.sqrt(noise_variance))
        self.rk.R = np.eye(3)*noise_variance #measurement noise covariance matrix ->errore sulla misurazione

        for i in range(200):
            z = np.array([self.centroid_pose_.pose.position.x+noise,self.centroid_pose_.pose.position.y+noise,self.centroid_pose_.pose.position.z+noise])
            #print(self.centroid_pose_.pose.position.z)
            #z = T_sphere_to_camera[3, 3] + noise
            self.rk.predict()
            #self.rk.update(z, self.Jacobian, self.Hx, R=self.rk.R)
            self.rk.update(z, self.Jacobian, self.Hx)


        self.posx_.append(self.centroid_pose_.pose.position.x)
        self.posy_.append(self.centroid_pose_.pose.position.y)
        self.posz_.append(self.centroid_pose_.pose.position.z)

        self.xsx_.append(self.rk.x[0])
        self.xsy_.append(self.rk.x[2])
        self.xsz_.append(self.rk.x[4])

        """
        if t > 50.00:
            self.xsx_ = asarray(self.xsx_)
            self.xsy_ = asarray(self.xsy_)
            self.xsz_ = asarray(self.xsz_)
            self.posx_ = asarray(self.posx_)
            self.posy_ = asarray(self.posy_)
            self.posz_ = asarray(self.posz_)

            # Primo grafico per x
            plt.figure()
            plt.plot(range(len(self.xsz_)), self.xsx_, label='EKF x', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posx_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.title('Confronto EKF vs Posizione Reale (x)')
            plt.legend()

            # Secondo grafico per y
            plt.figure()
            plt.plot(range(len(self.xsz_)), self.xsy_, label='EKF y', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posy_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.title('Confronto EKF vs Posizione Reale (y)')
            plt.legend()

            # Terzo grafico per z
            plt.figure()
            plt.plot(range(len(self.xsz_)), self.xsz_, label='EKF z', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posz_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.title('Confronto EKF vs Posizione Reale (z)')
            plt.legend()

            # Mostra tutti i grafici
            plt.show()
            
            plt.plot(range(len(self.xsz_)), self.xsx_, label='EKF x', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posx_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.legend()
            plt.figure()
            plt.plot(range(len(self.xsz_)), self.xsy_, label='EKF y', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posy_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.legend()
            plt.figure()
            plt.plot(range(len(self.xsz_)), self.xsz_, label='EKF z', color='b', marker='o')
            plt.plot(range(len(self.xsz_)), self.posz_, label='Real position', color='r', marker='x')
            plt.xlabel('Numero misurazioni nel tempo')
            plt.ylabel('Posizione')
            plt.legend()
            plt.show()
            
            # Creiamo un grafico 3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Dati calcolati
            ax.scatter(self.xsx_, self.xsy_, self.xsz_, c='b', marker='o', label='EKF')

            # Dati reali
            ax.scatter(self.posx_, self.posy_, self.posz_, c='r', marker='^', label='Real Position')

            # Etichette
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Confronto tra posizioni EKF e reali')

            # Legenda
            ax.legend()

            # Mostra il grafico
            plt.show()
            """
 

    def stopSimulation(self):
        self.sim.stopSimulation()


def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')

    stewart_platform_ekf = StewartPlatformEKF()
    
    #rate = 100# Hz
    #ros_rate = rospy.Rate(rate)

    while (t := stewart_platform_ekf.sim.getSimulationTime()) < 20:
        stewart_platform_ekf.update()

    #while not rospy.is_shutdown():
    #    stewart_platform_ekf.update()
     #   ros_rate.sleep()
    stewart_platform_ekf.stopSimulation()
    stewart_platform_ekf.plotData()
 

if __name__ == '__main__':
    main()
