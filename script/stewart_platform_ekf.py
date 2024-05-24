#!/usr/bin/env python3
import numpy as np
import rospy

import tf

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion


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
        print(self.sphere_pose_)
        rot_matrix = tf.transformations.quaternion_matrix([self.sphere_pose_.pose.orientation.x,self.sphere_pose_.pose.orientation.y,self.sphere_pose_.pose.orientation.z,self.sphere_pose_.pose.orientation.w])
        trasl_matrix = tf.transformations.translation_matrix( [self.sphere_pose_.pose.position.x, self.sphere_pose_.pose.position.y, self.sphere_pose_.pose.position.z])
        self.sphere_matrix_ = np.dot(trasl_matrix, rot_matrix)
        print(self.sphere_matrix_)

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
            print(T_matrix)
            print('-------')
            return T_matrix

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print('No transform available')


    def update(self):
        T_sphere_to_world = self.get_transform('world','red_sphere')

        # self.sphere_matrix_

#------------------------------------------------------------------

def main():
    rospy.init_node("stewart_platform_ekf")

    stewart_platform_ekf = StewartPlatformEKF()
    
    rate = 100 # Hz
    ros_rate = rospy.Rate(rate)

    while not rospy.is_shutdown():
        # stewart_platform_ekf.update()
        ros_rate.sleep()

if __name__ == '__main__':
    main()
