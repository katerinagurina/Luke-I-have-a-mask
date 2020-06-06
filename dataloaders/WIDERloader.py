#by Andrei Erofeev
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
import os
from dataloaders.transforms import transform, resize
import torchvision.transforms.functional as TF

device = ('cuda' if torch.cuda.is_available else 'cpu')

class WIDERLoader(Dataset):
    def __init__(self, list_file, img_path, mode='train', cut_size=None):
        super(WIDERLoader, self).__init__()
        self.img_path = img_path
        self.mode = mode
        self.cut_size = cut_size
        self.fnames = []
        self.targets = []
        self.boxes = []
        #self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        line_i = 0
        while line_i < len(lines):
            self.fnames.append(lines[line_i])
            line_i += 1
            num_faces = int(lines[line_i])
            if num_faces == 0:
                line_i += 1
            line_i += 1
            box = []
            label = []
            for i_face in range(num_faces):
                line = lines[line_i].strip().split()
                x = float(line[0])
                y = float(line[1])
                w = float(line[2])
                h = float(line[3])
                #c = int(line[4])
                line_i += 1
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h, 2])
                #label.append(c)
            self.boxes.append(box)
            #self.labels.append(label)

        self.num_samples = len(self.boxes)
        if self.cut_size is not None:
            self.num_samples = self.cut_size
            self.fnames = self.fnames[:self.cut_size]
            self.boxes = self.boxes[:self.cut_size]
            #self.labels = self.labels[:self.cut_size]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = os.path.join(self.img_path, self.fnames[index])
        image_path = image_path.rstrip()
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        img, boxes = transform(img, self.boxes[index], box_mode='xy')
        #img = TF.to_tensor(img)
        #boxes = self.boxes[index]

        #targets = {}
        #targets['boxes'] = torch.tensor(boxes).to(device)
        #targets['labels'] = torch.tensor(self.labels[index]).to(device)
        return torch.from_numpy(np.asarray(img)).to(device), torch.tensor(boxes, dtype = torch.float32).to(device)


def collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        d = {}
        d['boxes'] = sample[1]['boxes']
        d['labels'] = sample[1]['labels']
        targets.append(d)
        # targets.append(sample[1])
    return imgs, targets