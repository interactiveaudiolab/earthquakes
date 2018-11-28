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
        weights = np.vstack([item_one['weight'], item_two['weight']])
        return {'data': data, 'label': label, 'weight': weights}

    def __len__(self):
        return len(self.dataset)


class EarthquakeDataset(Dataset):
    def __init__(self, folder, transforms='demean', length=20000, split='', augmentations='', filter_labels=('negative', 'positive')):
        self.folder = folder
        self.files = sorted([os.path.join(folder, x) for x in os.listdir(folder) if '.p' in x])
        labels = filter_labels
        for fname in self.files:
            label = fname.split('_')[-1][:-2]
            if label not in filter_labels:
                self.files.remove(fname)
        self.labels = sorted(list(set(labels)))
        self.length = length
        self.transforms = transforms.split(':')
        self.augmentations = augmentations.split(':')
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

        sacs = self.pre_transform(sac, self.transforms)
        data = [self.get_surface_window(s) for s in sacs]
        data = np.stack(data, axis=0)

        data = self.get_target_length_and_transpose(data, self.length)
        weight = 1.0
        data = self.augment(data, self.augmentations)
        data = self.post_transform(data, self.transforms)

        return {'data': data, 'label': one_hot, 'weight': weight}

    def __len__(self):
        return len(self.files)

    def get_surface_window(self, sac):
        velocities = [5.0, 2.5]

        start = int(sac.stats.sac['dist'] / velocities[0])
        stop = int(sac.stats.sac['dist'] / velocities[1])

        t = sac.stats.starttime
        sac.trim(t + start, t + stop, nearest_sample=False)
        return sac.data

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
    def pre_transform(sac, transforms):
        data = []
        if 'demean' in transforms:
            sac.detrend(type='demean')
        if 'raw' in transforms:
            data.append(sac)
        if 'bandpass' in transforms:
            sac_copy = copy.deepcopy(sac)
            sac_copy.filter('bandpass', freqmin=2, freqmax=8, corners=4, zerophase=True)
            data.append(sac_copy)
        if 'lowpass' in transforms:
            sac_copy = copy.deepcopy(sac)
            sac_copy.filter('lowpass', freq=2)
            data.append(sac_copy)
        return data

    @staticmethod
    def post_transform(data, transforms):
        if 'whiten' in transforms:
            data -= data.mean()
            data /= data.std() + 1e-6
        return data

    @staticmethod
    def augment(data, augmentations):
        if 'amplitude' in augmentations:
            start_gain, end_gain = [np.random.random(), np.random.random()]
            amplitude_mod = np.linspace(start_gain, end_gain, num=data.shape[-1])
            data *= amplitude_mod

        if 'noise' in augmentations:
            std = data.std() * np.random.uniform(1, 2)
            mean = data.mean()
            noise = np.random.normal(loc=mean, scale=std, size=data.shape)
            coin_flip = np.random.random()
            if coin_flip > .5:
                data += noise
            
        return data