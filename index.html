Abstract—Fabric defect classification is traditionally achieved by human visual examination, which is inefficient and labor-intensive. Therefore, using intelligent and automated methods to solve this problem has become a hot research topic. With the increasing diversity of fabric defects, it is urgent to design effective methods to classify defects with a higher accuracy, which can contribute to ensuring the fabric products' quality. A fabric inspection system is a specialized computer vision system used to detect fabric defects for quality assurance. In this paper, a convolution neural networks algorithm was developed for an on-loom fabric defect inspection system by combining the techniques of image pre-processing, fabric motif determination, candidate defect map generation, and convolutional neural networks (CNNs).

Keywords—fabric defect estimation, tensorflow, convolutional neural network, classification

I.Introduction
In the textile industry, fabric defects represent an important problem in the quality control of textile manufacturing. Fabric defect detection and classification are the main methods used to ensure the quality of the fabric. Fabric defect classification is a vitally important process in the fabric quality evaluation, which can provide the defect information needed to adjust the machines and improve the processing technology. In the field of computer vision, the study of fabric defects has been a hot topic. Conventionally, fabric defects are evaluated by visual inspections of trained workers in accordance with human-made classification standards. Fabric defect classification remains a research issue and faces some difficulties due to the following three reasons. Firstly, new classes of fabric defects may be introduced with the growing application of fabric. Secondly, the similarities among different classes of fabric defects and the intraclass diversities of fabric defects make their discriminations challenging. Finally, different fibers, patterns and organizations of fabrics also make defect classification difficult.

II.Neural networks
A.Convolutional Neural Networks

Fig. 1. A simple convolutional neural networks

In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery. They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on their shared-weights architecture and translation invariance characteristics. They have applications in image and video recognition, recommender systems, image classification, medical image analysis, and natural language processing.

CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "fully-connectedness" of these networks makes them prone to overfitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme.

There are four main operations in the convolutional neural networks shown in Figure 1 above: Convolution, Non Linearity (ReLU), Pooling or Sub Sampling, Classification (Fully Connected Layer)

B.The Convolutional Neural Networks Step
When programming a CNN, the input is a tensor with shape (number of images) x (image width) x (image height) x (image depth). Then after passing through a convolutional layer, the image becomes abstracted to a feature map, with shape (number of images) x (feature map width) x (feature map height) x (feature map channels). A convolutional layer within a neural network should have the following attributes:

Convolutional kernels defined by a width and height (hyper-parameters). The number of input channels and output channels (hyper-parameter). The depth of the Convolution filter (the input channels) must be equal to the number channels (depth) of the input feature map.

Every image can be considered as a matrix of pixel values. Consider a 5 x 5 image whose pixel values are only 0 and 1 (note that for a grayscale image, pixel values range from 0 to 255, the green matrix below is a special case where pixel values are only 0 and 1).



Fig. 2. Pixel of image example

 

Fig. 3. filter example

 

Fig. 4. The Convolution operation. The output matrix is called Convolved Feature or Feature Map

Then, the Convolution of the 5 x 5 image and the 3 x 3 matrix can be computed as shown in the animation in Figure 4.

In CNN terminology, the 3×3 matrix is called a ‘filter’ or ‘kernel’ or ‘feature detector’ and the matrix formed by sliding the filter over the image and computing the dot product is called the ‘Convolved Feature’ or ‘Activation Map’ or the ‘Feature Map‘. It is important to note that filters acts as feature detectors from the original input image

C.Introducing Non Linearity (ReLU)
An additional operation called ReLU has been used after every Convolution operation in Figure 1 above. ReLU stands for Rectified Linear Unit and is a non-linear operation.



Fig. 5. The ReLU operation

ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. The purpose of ReLU is to introduce non-linearity in our ConvNet, since most of the real-world data we would want our ConvNet to learn would be non-linear (Convolution is a linear operation – element wise matrix multiplication and addition, so we account for non-linearity by introducing a non-linear function like ReLU).

 Fig. 6. The ReLU operation result on image

Other nonlinear functions such as tanh or sigmoid can also be used instead of ReLU, but ReLU has been found to perform better in most situations.

D.The Pooling Step
Spatial Pooling (also called subsampling or down sampling) reduces the dimensionality of each feature map but retains the most important information. Spatial Pooling can be of different types: Max, Average, Sum etc.

 

In case of Max Pooling, we define a spatial neighborhood (for example, a 2×2 window) and take the largest element from the rectified feature map within that window. Instead of taking the largest element we could also take the average (Average Pooling) or sum of all elements in that window. In practice, Max Pooling has been shown to work better.



Fig. 7. Max Pooling.

Slided our 2 x 2 window by 2 cells (also called ‘stride’) and take the maximum value in each region. As shown in Figure 7, this reduces the dimensionality of our feature map.

 

Fig. 8. Pooling applied to Rectified Feature Maps

In the network shown in Figure 8, pooling operation is applied separately to each feature map (notice that, due to this, we get three output maps from three input maps).



Fig. 9. Max Pooling

Figure 9 shows the effect of Pooling on the Rectified Feature Map we received after the ReLU operation in Figure 6 above. The function of Pooling is to progressively reduce the spatial size of the input representation

E.Fully Connected Layer
The Fully Connected layer is a traditional Multi-Layer Perceptron that uses a softmax activation function in the output layer. The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer.

 Fig. 10. Fully Connected Layer -each node is connected to every other node in the adjacent layer

The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset. For example, the image classification task we set out to perform has four possible outputs as shown in Figure 10.



Fig. 11. Visualizing Convolutional Neural Networks

III.Fabrıc defect dedectıon

Fig. 12. Classification of defect detection methods [10]

 

Fig. 13. Summary of detection success rates of previous methods in statistical approach [1]

A.Aitex Fabric Image Database


Fig. 14. Example of fabric defect [10]

The textile fabric database consists of 245 images of 7 different fabrics. There are 140 defect-free images, 20 for each type of fabric. With different types of defects, there are 105 images.

 

Images have a size of 4096×256 pixels. Defective images have been denominated as follows: nnnn_ddd_ff.png, where nnnn  is the image number, ddd is the defect code, and ff is the fabric code.

 

There is a mask of defect, denominated as: nnnn_ddd_ff_mask.png, where white pixels represent the defect area of the defective image.

 

B.Use Cnn for  Detect Detection
 Fig. 15. The architecture of the new convolutional neural network. [2]

 

 

C.Result of Cnn Model
 Fig. 16. Convolutional Neural Networks Model.

In application with limited training data, transfer learning has been considered as a good solution. Transferring the fixed weights will obtain a good performance, when the source and target domains are similar.

 

 Fig. 17. Result of Convolutional Neural Networks Model.

 



Fig. 18. Result of Convolutional Neural Networks Model.

Created cnn model to detect fabric defect in figure 16. After train calculated loss an accuracy for train and test dataset.  Train loss show as orange color and test loss show as blue color in figure 17. Train accuracy show as orange color and test accuracy show as blue color in figure 17.

Learning is firstly successful so train and test dataset accuracy is about 80% while epoch is 250. After epoch is 250, learning is slowly go up.

There is a dataset which is aitex fabric dataset for fabric detect. There is some article about fabric detection with convolution neural networks.

D.References
Ngan, H. Y. T., et al. (2011). "Automated fabric defect detection—A review." Image and Vision Computing 29(7): 442-458
Ouyang, W., et al. (2019). Fabric Defect Detection Using Activation Layer Embedded Convolutional Neural Network. 7: 70130-70140.
Schneider, D., et al. (2012). A vision based system for high precision online fabric defect detection, IEEE: 1494-1499.
Shuang, M., et al. (2018). "Automatic Fabric Defect Detection with a Multi-Scale Convolutional Denoising Autoencoder Network Model." Sensors (14248220) 18(4): 1064.
Sun, J., et al. (2019). "Surface Defects Detection Based on Adaptive Multiscale Image Collection and Convolutional Neural Networks." IEEE Transactions on Instrumentation and Measurement, Instrumentation and Measurement, IEEE Transactions on, IEEE Trans. Instrum. Meas. 68(12): 4787-4797.
Wei, B., et al. (2019). A new method using the convolutional neural network with compressive sensing for fabric defect classification based on small sample sizes. 89: 3539-3555.
Zhang, M., et al. (2019). Two-step Convolutional Neural Network for Image Defect Detection, Technical Committee on Control Theory, Chinese Association of Automation: 8525-8530.
Zhao, Y., et al. (2019). "A Visual Long-short-term Memory Based Integrated CNN Model for Fabric Defect Image Classification." Neurocomputing.
AITEX FABRIC IMAGE DATABASE
 a  public fabric image database for defect detection. Javier Silvestre-Blanes, Teresa Albero-Albero, Ignacio Miralles, Rubén Pérez-Llorens, Jorge Moreno AUTEX Research Journal, No. 4, 2019
 
