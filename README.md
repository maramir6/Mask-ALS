# Mask-ALS
Mask RCNN implementation for Airborne LiDAR Sensor

Pytorch implementation of MaskRCNN model for tree crown segmentation based on Canopy Height Model efficiently computed from Airborne LiDAR Sensor. Some of the additional modular functions have been compiled on the flight in C and C++ code through the Numba librery for addiotional speed. The input is a Laz file, and the output is a Canopy Height Model and Shapefile with trees and some of their attributes.

