# CIFAR10

CIFAR-10 is a classic machine learning problem which is considered a good place for beginners of machine lerning to solve. Creating a simple model is relatively straightforward however a number of more advanced techniques can be implemented to enhance the accuracy of the model. Most state of the art models are able to achieve accuracies of over 95%. The dataset for CIFAR10 is freely available at https://www.cs.toronto.edu/~kriz/cifar.html

This model was made using the PyTorch Deep Learning Framework. It makes use of the standard convolution technique to reduce the number of inputs and tus computational load while simultaneously improving acuracy and taking advantage of repeating patterns. A number of more advanced techniques have been implemented here to bring the accuracy up to a value of 83.78% on the test set, these techniques include:-

1. Data Augumenttion:
     By pre processing the images in the dataset through horizontal flips or random cropping, the accuracy of the model can be imporved by forcing the model to learn the patterns behind the objects and avoid overfitting.

2. Dropout:
     Dropout has been used in a number of places in the architecture of the model. It randomly sets the value of a set percentage of neurons in a fully connected layer 0. By doing this, over reliance on certain neurons and thus over fitting can be reduced further enhancing the accuracy of the model.


