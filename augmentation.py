import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

# define the data augmentation transformations
trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_path= Path.cwd()/ 'cassava-leaf-disease-classification/amls_valid/2843402791.jpg'
# load the original image
img_train = Image.open(image_path)

# apply the data augmentation to the original image
aug_img = trans_train(img_train)

# convert the augmented image back to a PIL image
aug_img_pil = transforms.ToPILImage()(aug_img)

# display the original and augmented images side by side
img_train.show()
aug_img_pil.show()

