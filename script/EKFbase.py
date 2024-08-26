#!/usr/bin/env python3
import numpy as np
import rospy
from numpy import asarray
from filterpy.kalman import ExtendedKalmanFilter
import tf
import matplotlib.pyplot as plt
from geometry_msgs.msg import PoseStamped
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from StewartPlatform import *


class EKFBase():
    def __init__(self,dim_x,dim_z):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.sphere_topic_name_ = rospy.get_param('~sphere_topic_name','/sphere/pose')
        self.centroid_topic_name_ = rospy.get_param('~centroid_topic_name','/output/centroid_pose')

 
        self.pose_sub_ = rospy.Subscriber(self.sphere_topic_name_, PoseStamped, self.pose_callback)
        self.centroid_sub_ = rospy.Subscriber(self.centroid_topic_name_, PoseStamped, self.centroid_callback)
        self.sphere_matrix_ = 0

        self.tf_listener_ = tf.TransformListener()
        self.br_ = tf.TransformBroadcaster()
        self.start_t_ = rospy.get_time()

   
        self.rk = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)

        self.xs_ = []
        self.pos_ = []
        self.centroid_pose_ = PoseStamped()

        # Inizializzazione della simulazione
        self.client = RemoteAPIClient()
        self.stewart_platform = StewartPlatform(self.client)
        self.sim = self.client.require('sim')
        self.sim.setStepping(True)
        self.sim.startSimulation()

    def pose_callback(self, msg):
        self.sphere_pose_ = msg
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x,
                                                           self.sphere_pose_.pose.orientation.y,
                                                           self.sphere_pose_.pose.orientation.z,
                                                           self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix([self.sphere_pose_.pose.position.x,
                                                              self.sphere_pose_.pose.position.y,
                                                              self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)

    def centroid_callback(self, msg):
        self.centroid_pose_ = msg

    def get_transform(self, source_frame, target_frame):
        try:
            (trans, rot) = self.tf_listener_.lookupTransform(target_frame, source_frame, rospy.Time(0))
            rot_matrix = tf.transformations.quaternion_matrix([rot[0], rot[1], rot[2], rot[3]])
            trasl_matrix = tf.transformations.translation_matrix([trans[0], trans[1], trans[2]])
            T_matrix = np.dot(trasl_matrix, rot_matrix)
            return T_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print(f'No transform available {source_frame} - {target_frame}')

    def plot_data(self):
        self.xs_ = asarray(self.xs_)
        self.pos_ = asarray(self.pos_)
        plt.plot(range(len(self.xs_)), self.xs_, label='EKF', color='b', marker='o')
        plt.plot(range(len(self.xs_)), self.pos_, label='Real position', color='r', marker='x')
        plt.xlabel('Numero misurazioni nel tempo')
        plt.ylabel('Posizione')
        plt.legend()
        plt.show()

    def stop_simulation(self):
        self.sim.stopSimulation()






































        