import random
from torch.utils.data import Dataset
import torch
import cv2
import os

class Traffic_sign_dataset(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        random.seed(42)
        self.transform = transform

        if is_train:
            root = os.path.join(root, 'TrainData')
            k = 135
        else:
            root = os.path.join(root, 'ValidData')
            k = 50

        categories = os.listdir(root)

        self.datas = []

        for idx1, cate1 in enumerate(categories):
            for idx2, cate2 in enumerate(categories):

                if cate1 == cate2:
                    files1 = sorted(os.listdir(os.path.join(root, cate1)))
                    files2 = sorted(os.listdir(os.path.join(root, cate2)))

                    for idx_file1, file1 in enumerate(files1):
                        for idx_file2, file2 in enumerate(files2):
                            if idx_file1 <= idx_file2:
                                path_file1 = os.path.join(root, cate1, file1)
                                path_file2 = os.path.join(root, cate2, file2)

                                self.datas.append([path_file1, path_file2, 0])

                elif cate1 != cate2 and idx1 < idx2:
                    files1 = random.choices(os.listdir(os.path.join(root, cate1)), k = k)
                    files2 = random.choices(os.listdir(os.path.join(root, cate2)), k = k)

                    for file1, file2 in zip(files1, files2):
                        path_file1 = os.path.join(root, cate1, file1)
                        path_file2 = os.path.join(root, cate2, file2)

                        self.datas.append([path_file1, path_file2, 1])

        random.shuffle(self.datas)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        path_file1, path_file2, label = self.datas[index]

        image1 = cv2.imread(path_file1)
        image2 = cv2.imread(path_file2)

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), torch.tensor(label, dtype=torch.float32)