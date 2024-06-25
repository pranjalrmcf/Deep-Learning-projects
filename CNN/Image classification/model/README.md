
# Image Classification with CNN

This project implements a Convolutional Neural Network (CNN) for image classification using PyTorch. The model is trained on a dataset of natural scene images, classifying them into six categories: buildings, forest, glacier, mountain, sea, and street.

## Project Structure

1. **Dependencies**: 
   - PyTorch
   - torchvision
   - OpenCV (cv2)
   - matplotlib
   - numpy
   - PIL

2. **Data Preparation**:
   - Images are transformed and normalized using `torchvision.transforms`
   - Dataset is loaded using `torchvision.datasets.ImageFolder`
   - Data is split into training and testing sets

3. **Model Architecture**:
   - Custom CNN class with:
     - 2 Convolutional layers
     - Batch normalization
     - ReLU activation
     - Max pooling
     - 5 Fully connected layers with dropout

4. **Training**:
   - Uses CrossEntropyLoss and SGD optimizer
   - Trains for 70 epochs
   - Tracks and prints training/testing loss and accuracy for each epoch

5. **Evaluation**:
   - Model performance is evaluated on a separate test dataset

6. **Visualization**:
   - Plots training and testing loss
   - Plots training and testing accuracy

7. **Model Persistence**:
   - Option to save the trained model weights
   - Option to load a pre-trained model

## Usage

1. Ensure all dependencies are installed
2. Prepare your dataset in the specified directory structure
3. Run the training script
4. Evaluate model performance using the provided visualization tools

## Results

- Final training and testing accuracy (89%, 82%)
- Loss and accuracy plots![Screenshot 2024-06-25 153423](https://github.com/pranjalrmcf/Deep-Learning-projects/assets/68674726/0a34c2c1-c2e7-424d-b734-eb2e552358c3)
- Any insights or observations from the training process

## Future Improvements

- Experimenting with different model architectures
- Implementing data augmentation techniques
- Fine-tuning hyperparameters
- Expanding to more classes or larger datasets

