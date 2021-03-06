import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image


tag_mapper = {
    'A':0.,
    'B':1.,
    'C':2.
}


class MyDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None):
        with open(data_list) as f:
            self.filenames = f.readlines()[1:]## first one is head
        self.data_dir = data_dir
        self.transform = transform

        
    def __getitem__(self, idx):
        now = self.filenames[idx]
        name, tag = now.split(',')
              
        tag = int(tag_mapper[tag.strip()])
        #print(name, tag)
        #print(os.path.join(self.data_dir, name))
        #img = cv2.imread(os.path.join(self.data_dir, name).strip())
        #img = torch.Tensor(img)
        image = Image.open(self.data_dir+name).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(tag, dtype=torch.long), name

    def __len__(self):
        return len(self.filenames)


def collate_picture(samples):
    #batchsize*channel*h*w
    imgs = []
    tags = []
    names = []
    for sample in samples:
        imgs.append(sample[0])
        tags.append(sample[1])
        names.append(sample[2])
    batch = {'imgs': imgs, 'tags': tags, 'names': names}
    return batch











