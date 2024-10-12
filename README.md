# MacRoboMaster CV Team Challenge

This repository contains the code (challenge1.py) and final model checkpoint (final_mdeol_checkpoint.keras) for the Computer Vision challenge. 

## Code Explanation

The code uses a sequential Keras Tensorflow model with multiple convolution layers (and associated pooling layers) for predicting if an image contains a piece of armor or not. The images used to train the model were the ones provided. First, images were parsed and classified based on their associated .txt files. After this, images were processed by loading, resizing, and normalizing, where all values were then placed in a Tensorflow dataset. Then, the dataset was divided into a 80/20 training testing split and the model was compiled with Adam and a cross entropy loss. The model's final checkpoint was saved and exported as a file and the accuracies (training and testing) were plotted.

