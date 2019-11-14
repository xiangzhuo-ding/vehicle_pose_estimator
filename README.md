# coms_4995

-  DenseBox Baidu
        a) single fully convolutional
        b) uses locaitons and scales of an image 
        c) public dataset(MALF, KITTI)
        b) careful hard negative mining techniques
        
-  R-CNN
        a) regional proposal methods to generate all the potential 
           bounding box canadidates
        b) CNN distinguish objects for every proposal 
        c) can't distiguish car from far away
   
-  Yolo
        a) predicts bounding boxes and class probabilities 
           directly from full images in one evaluation
 

What is negative mining techniques?
What is upsampleing?

libornovax master thesis code summary 

Input 
a) dimension are kept to 128X256 in order to proccess image in one 
batch
b) image pixels are converted to [-1, 1] from [0, 255]
        (v-128)/128

Generate the center of where the car lies.
When training we need to convert the bounding boxes so that each scale 
has the same amount of the training data.
Use a dilated convolution instead of adding extra pooling layer and 
deconvolving.


Complext Yolo - need point clouds

Densebox and Yolo does not exactly converts a image to a 3D space. 
