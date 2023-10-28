# CIFAR10

## The Problem
CIFAR-10 is a classic machine learning problem which is considered a good place for beginners of machine lerning to solve. Creating a simple model is relatively straightforward however a number of more advanced techniques can be implemented to enhance the accuracy of the model. Most state of the art models are able to achieve accuracies of over 95%. The dataset for CIFAR10 is freely available at https://www.cs.toronto.edu/~kriz/cifar.html

## PyTorch
This model was made using the PyTorch Deep Learning Framework developed by Meta. It makes use of Convolutional Nerual Networks which are the industry standard in image classification and object detection to reduce the number of inputs and tus computational load while simultaneously improving acuracy and taking advantage of patterns in the data. A number of more advanced techniques have been implemented here to bring the accuracy up, these techniques include:-

   ### Data Augumenttion:
By pre processing the images in the dataset through horizontal flips, vertical flips, or random cropping, the accuracy of the model can be imporved by forcing the model to learn the patterns behind the images and avoid overfitting.

   ### Dropout:
Dropout, as described in https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf, randomly sets the value of a set percentage of neurons in a fully connected layer to 0. By doing this, over reliance on certain neurons and thus over fitting can be reduced further enhancing the accuracy of the model. The original paper for dropout shows that the optimal rate for dropout is 5around p=0.5 provided the model is deep enought to learn patters even in the abscence of its neurons. This recommendation has been followed here.

   ### Adam Optimizer:
While stochastic gradient descent is normally used to train models, it requires fine tuning of its learning rate and momentum parameters to achieve maximum accuracy. Its more sophisticated counterpart the Adam optimiser is able to converge faster and achieve great accuracy without having to fine tune its various paramters as in Stochastic Gradient Descent. It is desribed in https://arxiv.org/abs/1412.6980

## Testing 
The model used here managed to achieve an accuracy of 86.08% on the test dataset of 10,000 images.

## Uses
Image classification is an extrememly broad field with a wide variety of techniques and uses in a number of other fields. It also acts as the base for a number of more advanced techniques such as object detection. Completing this project helped give me a better understanding of image classification and the technology behind it. It also helped me gain knowledge of Convolutional Neural Networks and associated innovations such as the ReLU activation funcition, dropout, the Adam optimiser, and data augumentation. The model itself was inspired by VGGnet which is described in https://arxiv.org/abs/1409.1556.
   


