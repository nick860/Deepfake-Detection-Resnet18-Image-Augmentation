import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
def generate_augmented_images(data_dir, output_dir, num_samples_per_class=100):
  """
  Generates augmented images from a given dataset directory.

  Args:
      data_dir (str): Path to the directory containing the original images.
      output_dir (str): Path to the directory where the augmented images will be saved.
      num_samples_per_class (int, optional): The number of augmented images to generate per class. Defaults to 100.
  """

  # Define data transformations for augmentation
  data_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
  data_transforms_2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ], p=0.5),
    transforms.ToTensor()
    ]  )

  # Load the dataset
  train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=700, shuffle=False)

  print(enumerate(train_loader))
  # now for every class in the dataset, we will generate augmented images and put it in the output directory
  for class_idx, (imgs, labels) in enumerate(train_loader):
        print(class_idx)
        class_name = train_dataset.classes[class_idx]
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(num_samples_per_class):
            
            img = train_loader.dataset[i+700*class_idx][0]
            # for every image create a new 3 augmented image, and save it in the output directory
            img = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img.save(os.path.join(class_dir, f'{i}.png'))
            img2 = data_transforms_2(img)
            img2 = Image.fromarray((img2.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img2.save(os.path.join(class_dir, f'{i}_2.png'))
            img3 = data_transforms_2(img)
            img3 = Image.fromarray((img3.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            img3.save(os.path.join(class_dir, f'{i}_3.png'))
    
if __name__ == '__main__':

  data_dir = os.path.join(os.getcwd(), 'whichfaceisreal') # get the path of the current directory
  output_dir = os.path.join(os.getcwd(), 'whichfaceisreal2//train') # get the path of the current directory
  num_samples_per_class = 700  # Adjust the number of augmented images per class

  generate_augmented_images(data_dir, output_dir, num_samples_per_class)
