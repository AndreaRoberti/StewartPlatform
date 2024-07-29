# STEWART PLATFORM


## HOW TO 

- In one terminal run 

```bash
roscore
```
- Go to the CoppeliaSIM directory and run it by typing 

```bash
./coppeliaSim.sh 
```
- Open one of the simulated scene **ros_stewartPlatform.ttt** you can find in *scenes* folder

- After the source of the package : 

```bash
 source devel/setup.bash
```

You can run the launch file :

```bash
roslaunch stewart_platform stewart_color.launch 
```

- This launch file will start (**RUN**) the simulation inside coppelia, perform the color segmentation and PointCloud reconstruction from rgb and depth buffer.

- Remember to stop the simulation once you have done. 



## EXTENDED KALMAN FILTER

