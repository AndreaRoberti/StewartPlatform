#include <StewartPlatform/color_segmentation.h>

color_seg::color_seg(ros::NodeHandle &nh) : nh_(nh), private_nh_("~"),
                                            it_(nh),
                                            xyz_cld_ptr_(new pcl::PointCloud<pcl::PointXYZRGB>)
{
}

void color_seg::filterContour(cv::Mat &img, cv::Mat &mask)
{
    int largest_area = 0;
    int largest_contour_index = 0;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        double a = cv::contourArea(contours[i], false);
        if (a > largest_area)
        {
            largest_area = a;
            largest_contour_index = i;
        }
    }
    cv::Scalar color(255, 255, 255);
    cv::drawContours(mask, contours, largest_contour_index, color, cv::FILLED, 8, hierarchy);
}

void color_seg::PublishRenderedImage(image_transport::Publisher pub, const cv::Mat image, const std::string encoding, const std::string camera_frame)
{
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = camera_frame;
    const sensor_msgs::ImagePtr rendered_image_msg = cv_bridge::CvImage(header, encoding, image).toImageMsg();
    pub.publish(rendered_image_msg);
}

void color_seg::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{

    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8); // msg->encoding  sensor_msgs::image_encodings::BGR8) if BGR8 assertion fail for cvtColor. ...dno if 16
        image_rgb_ = cv_ptr->image;
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

void color_seg::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &input)
{
    pcl::PCLPointCloud2 pcl_pc2;             // struttura pc2 di pcl
    pcl_conversions::toPCL(*input, pcl_pc2); // conversione a pcl della pc2

    pcl_pc2.fields[3].datatype = sensor_msgs::PointField::FLOAT32;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    try
    {
        listener_.waitForTransform(optical_frame, fixed_frame_, input->header.stamp, ros::Duration(5.0));
        listener_.lookupTransform(fixed_frame_, optical_frame, input->header.stamp, optical2map_);

        pcl_ros::transformPointCloud(*cloud, *xyz_cld_ptr_, optical2map_);
        xyz_cld_ptr_->header.frame_id = fixed_frame_;
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
    }
}

// Function to convert a PCL point cloud to an OpenCV image
cv::Mat color_seg::pointCloudToRGBImage(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    cv::Mat image(cloud->height, cloud->width, CV_8UC3);
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        const pcl::PointXYZRGB &point = cloud->points[i];
        image.at<cv::Vec3b>(i / cloud->width, i % cloud->width) = cv::Vec3b(point.b, point.g, point.r);
    }
    return image;
}

// Function to convert an OpenCV image back to a PCL point cloud
pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_seg::RGBImageToPointCloud(const cv::Mat &image, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &original_cloud)
{
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    // cloud->width = image.cols;
    // cloud->height = image.rows;
    cloud->is_dense = false;
    // cloud->points.resize(cloud->width * cloud->height);

    for (size_t i = 0; i < original_cloud->points.size(); ++i)
    {
        // pcl::PointXYZRGB &point = cloud->points[i];
        pcl::PointXYZRGB point;
        cv::Vec3b color = image.at<cv::Vec3b>(i / image.cols, i % image.cols);
        if (color[0] != 0 && color[1] != 0 && color[2] != 0)
        {
            point.b = color[0];
            point.g = color[1];
            point.r = color[2];
            point.x = original_cloud->points[i].x;
            point.y = original_cloud->points[i].y;
            point.z = original_cloud->points[i].z;
            cloud->points.push_back(point);
        }
    }
    return cloud;
}

void color_seg::createPoseArray(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    geometry_msgs::PoseArray output_array;
    output_array.header.frame_id = fixed_frame_;
    output_array.header.stamp = ros::Time::now();
    output_array.poses.resize(cloud->points.size());

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        output_array.poses[i].position.x = cloud->points[i].x;
        output_array.poses[i].position.y = cloud->points[i].y;
        output_array.poses[i].position.z = cloud->points[i].z;
    }
    pose_output_pub_.publish(output_array);
}

cv::Mat color_seg::hue_red(cv::Mat HSV)
{
    cv::Mat hueMask_red_upper;
    inRange(HSV, cv::Scalar(160, 50, 90), cv::Scalar(180, 255, 255), hueMask_red_upper);
    cv::Mat hueMask_red_lower;
    inRange(HSV, cv::Scalar(0, 50, 90), cv::Scalar(10, 255, 255), hueMask_red_lower);

    return hueMask_red_lower | hueMask_red_upper;
}

void color_seg::init()
{

    private_nh_.param("fixed_frame", fixed_frame_, std::string("/world"));
    private_nh_.param("image_topic", image_topic_, std::string("/camera/color/image_rect_color"));
    private_nh_.param("optical_frame", optical_frame, std::string("/camera_color_optical_frame"));
    private_nh_.param("cld_topic_name", cld_topic_name, std::string("/camera/depth_registered/points"));

    cloud_sub = nh_.subscribe(cld_topic_name, 1, &color_seg::cloudCallback, this);
    image_sub_ = nh_.subscribe(image_topic_, 1, &color_seg::imageCallback, this);

    cloud_pub = nh_.advertise<sensor_msgs::PointCloud2>("/output_cloud", 1, this);
    points_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/output/points", 1, this);

    pose_output_pub_ = nh_.advertise<geometry_msgs::PoseArray>("/output/pose_array", 1, this);

    rendered_image_pub_ = it_.advertise("/output/image", 1);
}

void color_seg::PublishCentroid(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud)
{
    Eigen::Vector4f centroid, minp, maxp;
    // pcl::getMinMax3D(*cloud, minp, maxp);
    pcl::compute3DCentroid<pcl::PointXYZRGB>(*cloud, centroid);
    // centroid[0],  centroid[1], centroid[2]
}

void color_seg::update()
{
    if (!image_rgb_.empty() && xyz_cld_ptr_->size() > 0)
    {
        cv::Mat HSV;
        cv::cvtColor(image_rgb_, HSV, CV_BGR2HSV);

        cv::Mat hueMask_temp = hue_red(HSV);
        cv::Mat mask = cv::Mat::zeros(image_rgb_.size(), CV_8U);
        filterContour(hueMask_temp, mask);
        cv::Moments m = cv::moments(mask, false);
        cv::Point p_m = cv::Point(m.m10 / m.m00, m.m01 / m.m00);

        geometry_msgs::PointStamped point_out;
        point_out.header.frame_id = "camera_color_optical_frame";
        point_out.header.stamp = ros::Time::now();
        point_out.point.x = p_m.x;
        point_out.point.y = p_m.y;

        points_pub_.publish(point_out);

        cv::Mat red_segmented;

        cv::Mat rgb_image = pointCloudToRGBImage(xyz_cld_ptr_);
        cv::bitwise_and(image_rgb_, image_rgb_, red_segmented, mask);

        // Convert the segmented OpenCV image back to a PCL point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr red_cloud = RGBImageToPointCloud(red_segmented, xyz_cld_ptr_);

        pcl::toROSMsg(*red_cloud, output_cloud_msg_);
        output_cloud_msg_.header.frame_id = fixed_frame_;
        output_cloud_msg_.header.stamp = ros::Time::now();
        cloud_pub.publish(output_cloud_msg_);

        createPoseArray(red_cloud);

        PublishRenderedImage(rendered_image_pub_, mask, "mono8", "camera_color_optical_frame");
    }
}

// -----------------------------------------------------------------
int color_segmentation(int argc, char **argv)
{
    ros::init(argc, argv, "color_seg_node");
    ros::NodeHandle nh;
    color_seg color_seg(nh);
    color_seg.init();

    while (ros::ok())
    {
        ros::spinOnce();
        color_seg.update();
    }
    return 0;
}

int main(int argc, char **argv)
{
    return color_segmentation(argc, argv);
}