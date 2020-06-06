#by Ekaterina Gurina

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
import os
import scipy
import scipy.io
import pandas as pd
from dataloaders.transforms import transform, resize

device = ('cuda' if torch.cuda.is_available else 'cpu')

class MAFALoader(Dataset):
    def __init__(self, path_to_data, path_to_target, transform=transform, mode='train', only_multiple_faces = True,
                 cut_size = None):
        self.mode = mode
        mat = scipy.io.loadmat(path_to_target)
        if mode == 'train':
            column_name = 'label_train'
            imgName = 'imgName'
        else:
            column_name = 'LabelTest'
            imgName = 'name'
        df_labels = pd.DataFrame(mat[column_name][0]).rename({imgName: 'imgName'}, axis=1)
        if only_multiple_faces:
            df_labels = df_labels[df_labels['label'].map(len) > 1]
            df_labels = df_labels.reset_index()
        if cut_size is not None:
            df_labels = df_labels.iloc[:cut_size, :]
        self.df_labels = df_labels
        #self.fnames = os.listdir(path_to_data)
        self.fnames = [item for sublist in self.df_labels['imgName'].to_list() for item in sublist]
        self.path_to_data = path_to_data
        self.transform = transform

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.path_to_data, self.fnames[index])
        image = Image.open(image_path)
        image = image.convert('RGB')
        boxes = self.df_labels[self.df_labels['imgName'] == self.fnames[index]]['label'].values[0]
        lables = []
        init_sizes = (image.height, image.width)

        for b in boxes:
            if self.mode == 'train':
                if b[12] == -1: b[12] = 2
                if b[13] == -1: b[13] = 1
                good_position = (b[12] == 1 or b[12] == 2) and b[13] == 3
                bad_position = (b[12] == 1 or b[12] == 2) and (b[13] == 2 or b[13] == 1)
                true_label = 0 if good_position else (2 if bad_position else 1)
                lables.append(true_label)
            else:
                true_label = b[4] - 1 if b[4] != -1 else 0
                lables.append(true_label)

        # for b in boxes:
        #     if b[12] == -1:
        #         b[12] = 2
        #     if b[13] == -1:
        #         b[13] = 2
        #     good_position = (b[12] == 1 or b[12] == 2) and b[13] == 3
        #     bad_position = (b[12] == 1 or b[12] == 2) and (b[13] == 2 or b[13] == 1)
        #     true_label = 0 if good_position else (2 if bad_position else 1)
        #     lables.append(true_label)
        boxes = [list(b[:4]) for b in boxes]
        if self.transform:
            image, boxes = self.transform(image, boxes)
        boxes = torch.cat((boxes, torch.tensor(lables, dtype=torch.float32).unsqueeze(1)), dim = 1)
        # targets = {}
        # targets['boxes'] = torch.tensor(boxes).to(device).type(torch.float32)
        # targets['labels'] = torch.tensor(labels).to(device).type(torch.float32)
        return torch.from_numpy(np.asarray(image)).to(device), torch.tensor(boxes, dtype=torch.float32).to(device), init_sizes

#by Andrei Erofeev
def collate_fn(batch):
    #targets = []
    annots = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        #d = {}
        #d['boxes'] = sample[1]['boxes']
        #d['labels'] = sample[1]['labels']
        #targets.append(sample[1])
        annots.append(sample[1])

    widths = [int(s.shape[1]) for s in imgs]
    heights = [int(s.shape[2]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, 3, max_width, max_height).to(device)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :, :int(img.shape[1]), :int(img.shape[2])] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
    # print(annot_padded.shape)
    if max_num_annots > 0:
        for idx, annot in enumerate(annots):
            # print(annot.shape)
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    return padded_imgs.to(device), annot_padded.to(device)#torch.FloatTensor(imgs), torch.FloatTensor(targets)