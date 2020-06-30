# Traffic-Sign-Classification-using-CNN
Traffic signal Calssification using CNN in pytorch

**for this project I used Python and Pytorch to classify German Traffic Sign Dataset**
* Dataset Used: *German Traffic Sign Dataset.*
* Training Dataset has 39202 Training images classified into 43 classes.
* Test Dataset has 12630 images without classes.
* Test Dataset Image's labels can be found in the Test.csv file.

Dataset is available at: [GTSRB - German Traffic Sign Recognition Benchmark](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

I used Convolution Neural Network Model to Classify the Dataset.<br>
**I was able to reach**<br>
**99.83% accuarcy for the validation dataset**<br>
**97.35% accuarcy for the test dataset**<br>

**Process Used:**
* Load the data
* Apply Transformation
* Dataset visualization
* Split the Dataset into Training and validationa Datasets
* Define the model
* Train the model using Training Dataset
* Evaluate the model using validation Dataset 
* Test The model using Test Dataset

**Environment:**
Kaggle- gpu runtime environment. 

## Load the Data
Loaded the dataset using ImageFolder from torchvision.<br>
It loads the image dataset which consist of classes with each class images with in separate folder.<br>
Tarining Data is in Train directory.<br>
Testing Data is in Test directory.<br>


## Apply transformation
transfromation applied on the dataset.<br>

* **Resize** to get (32, 32) image size<br>
* **ToTensor** to transform the inpity image data into pytroch tensor.<br>
* **Normalize** the data.<br>


## Split the Dataset into Training and validationa Datasets

**Training and Validation Datasets**<br>
**Split the dataset into training and validation dataset**<br>
**I split the Tarining Dataset in 28000 training images and 9209 validation images**<br>
**Test Dataset is given**<br>

**Training set** - used to train the model i.e. compute the loss and adjust the weights of the model using gradient descent.<br>
**Validation set** - used to evaluate the model while training, adjust hyperparameters (learning rate etc.) and pick the best version of the model.<br>
**Test set** - used to compare different models, or different types of modeling approaches, and report the final accuracy of the model.<br>
Since there's no predefined validation set, we can set aside a small portion (9209 images) of the training set to be used as the validation set. We'll use the random_split helper method from PyTorch to do this. To ensure that we always create the same validation set, we'll also set a seed for the random number generator.<br>


## Dataset visualization
**Count Per Class:**<br>
![Count Per Class](https://drive.google.com/file/d/1rWrshpDbewc-SM5H2WGiC5oy2D_iwJ-c/view?usp=sharing)

**Single Image form the Training Dataset**<br>



**Batch Size: 200**<br>
Images in a batch:<br>


## Define the model:

**Design And implement a Deep Learning Model that learns to classify German Traffic Sign from the Training Dataset**<br>

I used CNN(Convolutional Neural Network) to classify the images in the dataset.<br><br>
**A Convolutional Neural Network** (ConvNet/CNN) is a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other.<br><br>
**Batch Normalization** is a technique for improving the speed, performance, and stability of artificial neural networks.<br><br>
**MyCnnModel(
  (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1))
  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
  (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (la1): Linear(in_features=2304, out_features=512, bias=True)
  (bn4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (la2): Linear(in_features=512, out_features=128, bias=True)
  (bn5): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (la3): Linear(in_features=128, out_features=43, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)**


## Train The Model
