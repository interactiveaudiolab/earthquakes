from torch.utils.data import Dataset
import os
import obspy
import pickle
import numpy as np
import copy

class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        item_one = self.dataset[i]
        item_two = self.dataset[np.random.randint(0, len(self.dataset))]
        data = np.vstack([item_one[0], item_two[0]])
        label = np.vstack([item_one[1], item_two[1]])
        return data, label

    def __len__(self):
        return len(self.dataset)

class EarthquakeDataset(Dataset):
    def __init__(self, folder, transforms='demean', length=20000, split=''):
        self.folder = folder
        self.files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.p' in x])
        labels = []
        for fname in self.files:
            label = fname.split('_')[-1][:-2]
            labels.append(label)
        self.labels = sorted(list(set(labels)))
        self.length = length
        self.transforms = transforms.split(':')
        self.split = split.split(':')
        self.split = self.filter_files(split)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            earthquake = pickle.load(f)
        label = earthquake['label']
        sac = earthquake['data']
        index = self.labels.index(label)
        one_hot = np.zeros(len(self.labels))
        one_hot[index] = 1

        sac.data = self.get_target_length_and_transpose(sac.data, self.length)
        data = self.transform(sac, self.transforms)
        return data, one_hot

    def __len__(self):
        return len(self.files)

    def toggle_split(self):
        tmp = self.files
        self.files = self.split
        self.split = tmp

    def filter_files(self, split):
        split_files = []
        for fname in self.files:
            with open(fname, 'rb') as f:
                earthquake = pickle.load(f)
            if earthquake['name'] in split:
                split_files.append(fname)
                self.files.remove(fname)
        return split_files

    def get_target_length_and_transpose(self, data, target_length):
        length = data.shape[-1]
        if target_length == 'full':
            target_length = length
        if length > target_length:
            offset = np.random.randint(0, length - target_length)
        else:
            offset = 0

        pad_length = max(target_length - length, 0)
        pad_tuple = [(0, 0) for k in range(len(data.shape))]
        pad_tuple[0] = (0, pad_length)
        data = np.pad(data, pad_tuple, mode='constant')
        data = data[offset:offset + target_length]
        return data

    @staticmethod
    def transform(sac, transforms):
        #cut AFTER filtering
        data = []
        if 'demean' in transforms:
            sac.detrend(type='demean')
        if 'raw' in transforms:
            data.append(sac.data)
        if 'bandpass' in transforms:
            sac_copy = copy.deepcopy(sac)
            sac_copy.filter('bandpass', freqmin=2, freqmax=8, corners=4, zerophase=True)
            data.append(sac_copy.data)
        if 'lowpass' in transforms:
            sac_copy = copy.deepcopy(sac)
            sac_copy.filter('lowpass', freq=2)
            data.append(sac_copy.data)

        return np.stack(data, axis=-1).T