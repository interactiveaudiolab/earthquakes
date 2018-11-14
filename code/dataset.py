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

        coin_flip = np.random.random()
        if coin_flip > .5:
            while np.argmax(item_two['label']) == np.argmax(item_one['label']):
                item_two = self.dataset[np.random.randint(0, len(self.dataset))]
        else:
            while np.argmax(item_two['label']) != np.argmax(item_one['label']):
                item_two = self.dataset[np.random.randint(0, len(self.dataset))]

        data = np.vstack([item_one['data'], item_two['data']])
        label = np.vstack([item_one['label'], item_two['label']])
        return {'data': data, 'label': label}

    def __len__(self):
        return len(self.dataset)

#class Augmenter(Dataset):


class EarthquakeDataset(Dataset):
    def __init__(self, folder, transforms='demean', length=20000, split='', filter_labels=('positive', 'negative')):
        self.folder = folder
        self.files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.p' in x])
        labels = []
        for fname in self.files:
            label = fname.split('_')[-1][:-2]
            if label in filter_labels:
                labels.append(label)
            else:
                self.files.remove(fname)
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

        data = self.transform(sac, self.transforms)
        data = self.get_target_length_and_transpose(data, self.length)

        return {'data': data, 'label': one_hot}

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
        pad_tuple[1] = (0, pad_length)
        data = np.pad(data, pad_tuple, mode='constant')
        data = data[:, offset:offset+target_length]
        return data

    @staticmethod
    def transform(sac, transforms):
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
        data = np.stack(data, axis=0)
        if 'whiten' in transforms:
            data -= data.mean()
            data /= data.std() + 1e-6
        return data