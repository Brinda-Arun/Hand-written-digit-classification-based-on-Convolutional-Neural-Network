# Hand-written-digit-classification-based-on-Convolutional-Neural-Network
## MNIST CNN Classifier

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying the MNIST handwritten digit dataset. 
The model is trained using the MNIST dataset provided in CSV format. The repository also includes evaluation metrics such as accuracy, confusion matrix,
recall matrix, and F1 score.

### Files

- `mnist_cnn.py`: The main script that trains the CNN model, evaluates its performance on the test set, and displays the evaluation metrics.
- `CustomMNISTDataset.py`: A custom dataset class for loading the MNIST data from CSV files.
- `Data/mnist_train.csv`: CSV file containing the training data samples and labels.
- `Data/mnist_test.csv`: CSV file containing the test data samples and labels.
-  The test and train data set can be downloaded from the following URL :  https://pjreddie.com/projects/mnist-in-csv/

**Functionality **

1.Importing the necessary libraries:

torch, torch.nn, torch.optim, torchvision.transforms: PyTorch modules for building and training neural networks.
sklearn.metrics: Scikit-learn library for evaluating classification metrics.
Dataset and DataLoader from torch.utils.data: Classes for creating custom datasets and data loaders.
Checking the availability of a GPU and setting the device:

2.The code checks if a GPU (CUDA) is available and sets the device accordingly. If not, it falls back to using the CPU.
Defining the CNN model:

3.The CNN class is defined, which is a subclass of nn.Module.
The model consists of a series of convolutional layers, ReLU activation, max pooling, and fully connected layers.
The forward method defines the forward pass of the model.
Creating a custom dataset class:

4.The CustomMNISTDataset class is defined, which is a subclass of Dataset.
The class reads MNIST data from CSV files and preprocesses it.
The __len__ method returns the length of the dataset, and __getitem__ method retrieves a specific sample and label.
Loading and transforming the MNIST dataset:

5.The MNIST training and test datasets are loaded using the CustomMNISTDataset class.
A transformation is applied to normalize the image data.
Creating data loaders:

6.Data loaders are created for both the training and test datasets.
Data loaders handle batch loading and shuffling of the data during training and evaluation.
Creating the CNN model, loss function, and optimizer:

7.An instance of the CNN model is created and moved to the appropriate device (CPU or GPU).
The loss function (CrossEntropyLoss) and optimizer (Adam) are defined.
Training the model:

8.The model is trained for a specified number of epochs.
Within each epoch, the training data is loaded in batches.
The model performs a forward pass, computes the loss, backpropagates the gradients, and updates the model's parameters.
The loss is printed every 100 steps.
Evaluating the model:

9.The model is switched to evaluation mode using model.eval().
The test data is loaded, and predictions are made using the trained model.
The true labels and predicted labels are stored.
Accuracy is calculated using accuracy_score from sklearn.metrics.
Printing evaluation metrics:

10.The test accuracy is printed.
Confusion matrix and classification report (including recall, precision, and F1 score) are computed using confusion_matrix and classification_report from sklearn.metrics and then printed.


### Acknowledgments

The CNN architecture and training code in this repository are based on the MNIST classification example provided in the PyTorch documentation. 
The custom dataset class for loading MNIST data from CSV files is inspired by various examples available in the PyTorch community.

Please note that the code provided here is a simplified implementation for demonstration purposes and may not include all possible optimizations or 
advanced techniques. It is recommended to consult the official PyTorch documentation and relevant research papers for a comprehensive understanding of 
deep learning concepts and best practices.
