
# NeuromatchAcademy
This repository contains two sets of code, one related to the NMA 2020 tutorials and one related to the final project.
## Project description
**Objective:** In this project, we used the brain neural activity of mice to detect the expected behavior and the actual behavior of each trail defined in the paper [Steinmetz et al.](https://www.nature.com/articles/s41586-019-1787-x) (more info on task can be found below).
  1.We want to see if we can detect mouse responses based on its neural activity for each trial, also referred to as the response/actual behavior classification.  
  2.We want to see if we can detect each trial's expected action based on the mouse's neural activity, also referred to as the expected action classification.  
**Data:** [Steinmetz 2019](https://www.nature.com/articles/s41586-019-1787-x) data set, Spiking neural activity of different brain areas of mice  
**Input:** spiking rates for the visual and motor cortex of mice  
**Output:** classification accuracy on two different problems: 1-actual behavior detection 2-expected action detection  
**Method:** 1-D convolutional neural network
### More about the task:
In Steinmetz's task, a mouse was presented with two visual stimuli and had to determine which had higher contrast. To earn a water reward, the mouse had to turn the wheel in the opposite direction of the higher-contrast stimulus.  
Steinmetz et al. recorded from multiple brain areas over multiple recording sessions in different animals. All recordings were done on the left hemisphere, which corresponds to visual stimuli in the right visual field.
