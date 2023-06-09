# UCL-ELEC0135-assignment
## UCL ELEC0135 final assignment by Mujie Xu 19027325

This readme gives the structure of the code and the procedures about how to train, validate and test the models.

This assignment is to solve a past machine learning competition from Kaggle named: Cassava Leaf Disease Classification.
Link: https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview

## Three models are employed to solve this image classification task, as followed:

-- Pretrained ResNet-18; resnet.py

-- Pretrained VGG-19; vgg.py

-- Customized Resnet-18, model trained from scratch; resnet.py

All the code are built with PyTorch as the framework of the Deep Learning framework. 
Code are done and run using Nvidia GPU to accelerate.

The GPU information: 

![GPU_Info.PNG](https://github.com/mujiexu2/UCL-ELEC0135/blob/main/IMG/GPU%20Info.png)
## Python libraries used
- pytorch
- numpy
- matplotlib
- pillow
- csv
- pathlib
- logging
- shutil
- random
- os
- pandas
- time


For pytorch, Please check official website to install it correctly.
<https://pytorch.org/get-started/locally/>
For all libraries, they can be installed manually. Or, run command: pip install -r requirements.txt to on python 
virtual environment to configure relevant libraries.

## Project Organization Details
-- UCL-ELEC0135

&emsp; -- resnet.py

&emsp; -- resnet_customized.py

&emsp; -- vgg.py

&emsp; -- inference.py

&emsp; -- augmentation.py

&emsp; -- main.py

## Program Run Instruction

### Program Result Instruction
While running each model, it will generate five files: 

-- Logging for whole running process

-- Validation result for each epoch, total 20 epochs produce 20 results

-- Test result 

-- Validation result accuracy list in csv file, 20 accuracy included

-- Plot: 

&emsp;&emsp;    -- Validation accuracy vs. epoch no.

&emsp;&emsp;    -- Training accuracy vs. epoch no.

&emsp;&emsp;    -- Training loss vs. batch count

### Training and Validation
You can "python resnet/vgg" to run ResNet-18 and VGG-19 models to do the task. Both model has the configuration, about
setting the epoch number to 20, batch size to 64, learning rate with three settings: 0.1, 0.01, 0.001.
For "resnet,py", the ResNet-18 model, it contains two settings: pretrained/customized models.

-- If utilizing the pretrained ResNet-18 model, please make sure the code within run(), 
            be like: 
![resnet-18_pretrained_setting.PNG](https://github.com/mujiexu2/UCL-ELEC0135/blob/main/IMG/resnet-18_pretrained_setting.png)

-- If utilizing the customized ResNet-18 model, please make sure the code within run(), 
            be like:
![resnet-18_customized_setting.PNG](https://github.com/mujiexu2/UCL-ELEC0135/blob/main/IMG/resnet-18_customized_setting.png)

&emsp;&emsp; -- Within resnet.py, resnet_customized.py is imported with providing the customized, model trained from 
    random weights (self-built).

-- Both customized and pretrained ResNet-18 work fine with the three learning rate, while 0.001 gives the best result 
    with the pretrained ResNet-18.

For "vgg.py", the VGG-19 model, only pretrained VGG-19 model is utilized.

-- Learning rate can only be set to 0.001, while other learning rates provide extremely low training and validation accuracy.

### Testing
After running the training and validation file, which main.py does, it is critical to run inference.py. 

--Inference.py does applying the best model within the 20 epochs onto the test dataset, VGG-19 and Resnet-18 model.

&emsp;&emsp; -- User needs to pick the model(.pth file) from the "model" file via checking the plot and the validation 
accuracy list. Find the model path of the 20 epochs with the highest validation accuracy, and execute the model 
inference task.

&emsp;&emsp; -- The test accuracy, timing, and confusion matrix will be shown

&emsp;&emsp; -- If inferenced VGG model, commented all resnet model path and code, and change any related model path to 
    model_path_vgg

&emsp;&emsp; -- If inferenced Resnet model, commented all vgg model path and code, and change any related model path to 
    model_path_resnet

### Others:

-- Augmentation.py： User can apply the data augmentation towards the training images to see transformations





