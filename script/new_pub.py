#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import array, eye, asarray
from filterpy.kalman import ExtendedKalmanFilter
from numpy import array, eye, asarray
import tf
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from filterpy.common import Saver
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from StewartPlatform import *


from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class StewartPlatformEKF():
    def __init__(self):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        self.ekf_real_topic_name_ = rospy.get_param('~ekf_real_topic_name','/ekf/real/pose')
        self.ekf_est_topic_name_ = rospy.get_param('~ekf_est_topic_name','/ekf/real/pose')
        
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.pose_ekf_real_pub_ = rospy.Publisher(self.ekf_real_topic_name_, PoseStamped, queue_size=1)
        self.pose_ekf_estimate_pub_ = rospy.Publisher(self.ekf_est_topic_name_, PoseStamped, queue_size=1)
        
        self.sphere_matrix_ = 0


        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

        self.rk = ExtendedKalmanFilter(dim_x=4,dim_z=3)
        self.tempo_ = []
        self.xs_ = []  # Tutti i valori di x
        self.pos_ = []  # Posizioni reali z
        #self.s_ = Saver(self.rk)

                
        client = RemoteAPIClient()
        self.stewart_platform = StewartPlatform(client)
        self.sim = client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()


    def pose_callback(self,msg):
        self.sphere_pose_ = msg
        #print(self.sphere_pose_)
        #print(self.sphere_pose_)
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x,self.sphere_pose_.pose.orientation.y,self.sphere_pose_.pose.orientation.z,self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix( [self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)
        #print(self.sphere_matrix_)
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
            #print(T_matrix)
            #print('-------')
            return T_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print('No transform available %s - %s',source_frame, target_frame )

    # pos posizione reale, z2 posizione trovata  con il filtro
    def publishResult(self, pos, z2):
        print("--------------")
        pos_transformed = PoseStamped()
        pos_transformed.header.frame_id = 'world'
        pos_transformed.header.stamp = rospy.Time.now()
        pos_transformed.pose.position.x = pos[0]
        pos_transformed.pose.position.y = pos[1]
        pos_transformed.pose.position.z = pos[2]
        
        z2_transformed = PoseStamped()
        z2_transformed.header.frame_id = 'world'
        z2_transformed.header.stamp = rospy.Time.now()
        z2_transformed.pose.position.x = z2[0]
        z2_transformed.pose.position.y = z2[0]
        z2_transformed.pose.position.z = z2[0]
        
        self.pose_ekf_real_pub_.publish(pos_transformed)
        self.pose_ekf_estimate_pub_.publish(z2_transformed)


    def Hx(self, x):
        """ takes a state variable and returns the measurement
        that would correspond to that state.
        ybar è la posizione che ricavo dalla simulazione/ mean position of the feature in 3d space
        """
        T_sphere_to_world = self.get_transform('world','red_sphere')
        ybar = T_sphere_to_world[:3,3]
        alpha = x[0]
        tau= x[1]
        b = x[2]
        z0 = x[3]
        eig = np.array([1,1,1])
        n=3
        T_camera_to_world = self.get_transform('world', 'camera_link') #trasformation matrix camera world 
        rW = T_camera_to_world[:3,3] #translation vector
        Rrw = T_camera_to_world[:3,:3] #rotation matrix
        #print(Rrw.dot(eig))
        p1 = Rrw.dot(eig)*(z0-b*(np.cos(alpha)**(2*n))) + ybar -rW
        #print(np.shape(p1))
        return p1

    def add_measurement_noise(self,measurement, R):
        noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        return measurement + noise
    
    def Jacobian(self,m):
            """ compute Jacobian"""
            alpha = m[0]
            tau= m[1]
            b = m[2]
            z0 = m[3]
            eig =np.array([1,1,1])
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

        t= rospy.get_time() - self.start_t_ #time
        #print(f'Simulation time: {t:.2f} [s]')
        self.stewart_platform.example_ik(t)
        self.sim.step()

        #parametri modello
        z0=self.stewart_platform.z0_ #position of the liver at the exale
        tau = 1 #respiration frequency
        b = 3.09 #amplitude
        phi = 0 #phase
        n = 3 #gradient of the model
        alpha = np.pi*t/tau 

        #inizializzazione filtro
        self.rk.x = np.array([alpha, tau, b, z0]).T #state vector
        self.rk.F = np.eye(4) + np.array(([[0, 1, 0,0],[0, 0, 0, 0],[0, 0, 0, 0],[0,0,0,0]]))*t #prediction model
        phi_tau = phi_b = phi_z0 = 1
        self.rk.Q = ([[phi_tau  * (t**3) / 3, phi_tau * (t**2) / 2, 0, 0],
                        [phi_tau * (t**2) / 2, phi_tau * t, 0, 0],
                        [0, 0, phi_b * t, 0],
                        [0, 0, 0, phi_z0 * t]]) #process noise covariance 
        self.rk.R = np.eye(3)*(np.random.normal(0,0.001,1)**2)#rumore bianco gaussiano (media,ampiezza,numero_elementi), in alternativa si può generale con filterpy
        #measurement noise covariance 

        #iterazioni step filtro di kalman
        #numero iterazioni da definire, per ora 20
        for i in range(20): #????
            T_sphere_to_camera = self.get_transform('camera_link','red_sphere')
            if isinstance(T_sphere_to_camera, np.ndarray):
                z = T_sphere_to_camera[:3,3] #tutto il vettore di traslazione
                z = self.add_measurement_noise(z,self.rk.R) #aggiungo del rumore
                self.rk.update(z, self.Jacobian, self.Hx,R=self.rk.R)
                self.rk.predict()             
                #self.s_.save() --> nel test è utilizzata, serve?

        zz = self.get_transform('red_sphere','world')
        print(zz)
        print(zz[:, -1])
        zz = zz[:3,3]
        print(zz)
        # np.format_float_positional(scientific_notation, precision=10)
        self.publishResult(zz,zz)

        # if isinstance(zz, np.ndarray):
        #     zz = zz[:3,3] #vettore posizione
        #     self.pos_.append(zz) #lista di tutte le posizioni della pallina misurate dalla camera (no rumore)
        #     self.xs_.append(self.rk.x) #lista variabili di stato calcolate dal filtro
        
        
        # if t > 5.00:
        #     self.xs_ = asarray(self.xs_)                    
        #     self.pos_ = asarray(self.pos_)
        #     z2 = self.xs_[:,3] - self.xs_[:,2]*(np.cos(self.xs_[:,0])**(2*3)) 
        #     pos = self.pos_[:,2]

        #     self.publishResult(pos,z2)            
        
        # if t > 20.00:
        #     #self.s_.to_array()
        #     self.xs_ = asarray(self.xs_)
        #     self.pos_ = asarray(self.pos_)
        #     #print(np.shape(self.xs_))
            
        #     #il filtro mi restituisce le variabili di stato ma a me interssa lo spostamento lungo z della palla quindi sostituisco le varibiali nel modello?
        #     z2 = self.xs_[:,3] - self.xs_[:,2]*(np.cos(self.xs_[:,0])**(2*3)) 
        #     print()
        #     print(self.xs_[:,3])
        #     print(self.stewart_platform.z0_)
        #     #print(len(z2))
        #     print('*************************')
        #     #print(z2)

        #     plt.plot(range(len(z2)), z2, label='EKF', color='b', marker='o')
        #     pos = self.pos_[:,2]
        #     #print(pos) #posizione "reale"
        #     print('-----------------')
        #     #print(z2) #posizione trovata  con il filtro
        #     plt.plot(range(len(z2)), pos, label='Real position', color='r', marker='x')
        #     plt.xlabel('Numero misurazioni nel tempo')
        #     plt.ylabel('Posizione')
        #     plt.legend()
        #     plt.show()
        #     rospy.signal_shutdown('Plot completato')

    def stopSimulation(self):
        self.sim.stopSimulation()

#------------------------------------------------------------------

def main():
    rospy.init_node("stewart_platform_ekf")
    print('start')
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