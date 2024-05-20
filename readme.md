# AFEVS TASK

- Autonomous Focus Endoscope Visual Servoing Task

## Requirements

- Ubuntu 20.04

- ROS Noetic 

- smach_ros package : for FSM scripts

## Nodes

- SemanticSegmentation (c++) : for performing semantic segmentation on images, filter masks 

- Afevs_task (python): FSM for the entire procedure, based on smach_ros


## what is inside

- Semantic segmentation -> MASK object -> feature of the object
- Visual servoing with that feature 
- Velocity to Pose (velocity integrator)


# How to run

```bash
python3 scene_segmentation_node.py --name ars-gan_P2P --model pix2pix --direction AtoB --checkpoints_dir ./checkpoints/
```


