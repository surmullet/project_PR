import torch
import cv2
from torch.utils.data import Dataset
import os

class DataForKnn(Dataset):
    def __init__(self, root, transform = None, databaseSet = False):
        self.CLASS_NAMES = [
            "Speed limit (20km per h)", "Speed limit (30km per h)", "Speed limit (50km per h)",
            "Speed limit (60km per h)",
            "Speed limit (70km per h)", "Speed limit (80km per h)", "End of speed limit (80km per h)",
            "Speed limit (100km per h)",
            "Speed limit (120km per h)", "No passing", "No passing for vehicles over 3.5 metric tons",
            "Right-of-way at next intersection",
            "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
            "No entry", "General caution",
            "Dangerous curve to the left", "Dangerous curve to the right", "Double curve", "Bumpy road",
            "Slippery road",
            "Road narrows on the right", "Road work", "Traffic signals", "Pedestrians",
            "Children crossing",
            "Bicycles crossing", "Beware of ice-snow", "Wild animals crossing",
            "End of all speed and passing limits",
            "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
            "Go straight or left",
            "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
            "End of no-passing zone for trucks", 'Background'
        ]

        self.transform = transform
        self.label2idx = {}

        for idx, class_name in enumerate(self.CLASS_NAMES):
            self.label2idx[class_name] = idx

        self.datas = []

        if databaseSet:
            root = os.path.join(root, 'TrainData')
        else:
            root = os.path.join(root, 'ValidData')

        for cate in os.listdir(root):
            label = self.label2idx[cate]
            path_cate = os.path.join(root, cate)
            for image in os.listdir(path_cate):
                path_image = os.path.join(path_cate, image)
                self.datas.append([path_image, label])


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        path_image, label = self.datas[index]

        image = cv2.imread(path_image)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label)