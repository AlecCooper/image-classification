import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import PIL

class DataSet(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        image_list = os.listdir(root_dir)

        class_dict = {}

        # Filter out any random system files
        filtered_image_list = []
        index = 0
        for filename in image_list:
            if ".jpg" in filename or ".jpeg" in filename:
                filtered_image_list.append(filename)

                item = filename[:filename.find("_")]

                if not item in class_dict:
                    class_dict[item] = index
                    index+=1

        self.image_list = filtered_image_list
        self.class_dict = class_dict
        self.encode_dim = index

    # Calc length of the dataset
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index): 

        # Open the image  
        image_name = os.path.join(self.root_dir, self.image_list[index])
        image = PIL.Image.open(image_name)
        
        item = self.image_list[index][:self.image_list[index].find("_")]
        index = self.class_dict[item]
        label = torch.zeros(self.encode_dim, dtype=torch.float32)
        label[index] = 1

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return (image, label)
