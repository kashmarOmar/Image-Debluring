# Image Deblurring Using Deep Learning
This project tackles the challenge of image deblurring, a fundamental problem in computer vision, through a convolutional autoencoder. The model is designed to restore sharpness to images affected by various types of blur, including motion blur, camera shake, and defocus blur.
Key Features

    Custom Autoencoder Architecture:
        Encoder: Extracts essential features using strided convolutions for downsampling.
        Decoder: Reconstructs high-quality images while preserving spatial hierarchies.
    End-to-End Trainability: Optimized with the Adam optimizer for efficient learning.
    Performance: Achieved significant accuracy improvements across training, validation, and testing datasets.

Methodology

The project employs a dataset of sharp and blurred images to train and evaluate the autoencoder. The results demonstrate consistent improvements in image clarity over 100 epochs, with notable reductions in mean squared error.
Results

    Training Accuracy: 77.29%
    Validation Accuracy: 79.13%
    Testing Accuracy: 78.50%

Technologies Used

    TensorFlow/Keras for model development
    Python for data preprocessing and visualization
    Kaggle Dataset for training and testing

Dataset

The dataset used in this project is sourced from Kaggle.
Future Work

Potential enhancements include extending the model to handle diverse blur types and optimizing its performance for real-world applications.
