#ifndef COLOR_SEG_H
#define COLOR_SEG_H

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>

class color_seg
{
private:
    // Ros handler
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    std::string fixed_frame_, optical_frame, cld_topic_name, image_topic_;
    tf::StampedTransform optical2map_;
    tf::TransformListener listener_;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyz_cld_ptr_;
    cv::Mat image_rgb_;

    sensor_msgs::PointCloud2 output_cloud_msg_;

    ros::Publisher cloud_pub, points_pub_, pose_output_pub_;
    ros::Subscriber cloud_sub, image_sub_;

    image_transport::ImageTransport it_;
    image_transport::Publisher rendered_image_pub_;

    cv::Mat hue_red(cv::Mat HSV);

    void createPoseArray(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr RGBImageToPointCloud(const cv::Mat &image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &original_cloud);

    cv::Mat pointCloudToRGBImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    void filterContour(cv::Mat &img, cv::Mat &mask);

    void PublishCentroid(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud);

    void PublishRenderedImage(image_transport::Publisher pub, const cv::Mat image, const std::string encoding, const std::string camera_frame);

    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &input);

protected:
    virtual ros::NodeHandle &getNodeHandle() { return nh_; }
    virtual ros::NodeHandle &getPrivateNodeHandle() { return private_nh_; }

public:
    color_seg(ros::NodeHandle &nh);

    ~color_seg() {}

    void init();
    void update();

};

int color_segmentation(int argc, char **argv);

#endif // COLOR_SEG_H
