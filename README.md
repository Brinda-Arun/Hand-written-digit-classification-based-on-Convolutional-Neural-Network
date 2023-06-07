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

### Dependencies

The following dependencies are required to run the code:

- Python 3.x
- PyTorch
- scikit-learn

### Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mnist-cnn.git
   ```

2. Change into the repository directory:

   ```bash
   cd mnist-cnn
   ```

3. Install the dependencies:

   ```bash
   pip install torch scikit-learn
   ```

4. Run the script:

   ```bash
   python mnist_cnn.py
   ```

   The script will train the CNN model, print the loss during training, evaluate its performance on the test set, and display the accuracy,
   confusion matrix, recall matrix, and F1 score.



### Acknowledgments

The CNN architecture and training code in this repository are based on the MNIST classification example provided in the PyTorch documentation. 
The custom dataset class for loading MNIST data from CSV files is inspired by various examples available in the PyTorch community.

Please note that the code provided here is a simplified implementation for demonstration purposes and may not include all possible optimizations or 
advanced techniques. It is recommended to consult the official PyTorch documentation and relevant research papers for a comprehensive understanding of 
deep learning concepts and best practices.
