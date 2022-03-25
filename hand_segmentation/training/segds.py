import torch
import glob

import PIL.Image as Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import transforms


class SegDataset(Dataset):

    def __init__(self, parentDir, imageDir, maskDir):
        self.imageList = glob.glob(parentDir + '/' + imageDir + '/*')
        self.imageList.sort()
        self.maskList = glob.glob(parentDir + '/' + maskDir + '/*')
        self.maskList.sort()

    def __getitem__(self, index):
        preprocess = transforms.Compose([
            transforms.Resize((384, 288), 2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        X = Image.open(self.imageList[index]).convert('RGB')
        X = preprocess(X)

        trfresize = transforms.Resize((384, 288), 2)
        trftensor = transforms.ToTensor()

        yimg = Image.open(self.maskList[index]).convert('L')
        y1 = trftensor(trfresize(yimg))
        y1 = y1.type(torch.BoolTensor)
        y2 = torch.bitwise_not(y1)
        y = torch.cat([y2, y1], dim=0)

        return X, y

    def __len__(self):
        return len(self.imageList)


# TTR is Train Test Ratio
def trainTestSplit(dataset, TTR):
    trainDataset = torch.utils.data.Subset(dataset, range(0, int(TTR * len(dataset))))
    valDataset = torch.utils.data.Subset(dataset, range(int(TTR * len(dataset)), len(dataset)))
    return trainDataset, valDataset


