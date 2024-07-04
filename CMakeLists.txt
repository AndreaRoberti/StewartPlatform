cmake_minimum_required(VERSION 3.0.2)
project(stewart_platform)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_srvs
  geometry_msgs
  roscpp
  std_msgs
  tf
  tf2_ros
  tf2_eigen
  image_transport
  cv_bridge
  pcl_conversions
  pcl_ros
  pcl_msgs
  )
  
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS roscpp
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  # ${PCL_INCLUDE_DIRS} 
  # include
)

set(LIBRARIES
  ${catkin_LIBRARIES}
  ${Eigen3_LIBRARIES}
)



add_executable(coppelia_combiner src/coppelia_combiner.cpp)
target_link_libraries(coppelia_combiner  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(color_segmentation src/color_segmentation.cpp)
target_link_libraries(color_segmentation  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})