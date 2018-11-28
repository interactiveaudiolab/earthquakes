import obspy
import numpy as np
import os
from tqdm import trange
import pickle
from utils import save_file

folders = [os.path.join('/data/raw/trigger', x) for x in os.listdir('/data/raw/trigger')]

def read_surface_window(f):
    f = open(f)
    times = [s.rstrip().replace('\t', ' ').split(' ') for s in f.readlines()]
    d = {}
    for time in times:
        d[time[0].split('BH')[0]] = (float(time[1]), float(time[2]))
    f.close()
    return d


def load_data(directory, label, target_directory):
    if not os.path.exists(directory):
        return []
    earthquake_files = [x for x in os.listdir(directory) if '.SAC' in x]

    progress_bar = trange(len(earthquake_files))
    if len(earthquake_files) == 0:
        return []

    velocities = [5.0, 2.5]
    for i in progress_bar:
        earthquake_file = earthquake_files[i]
        progress_bar.set_description(os.path.join(directory, earthquake_file))
        earthquake = obspy.read(os.path.join(directory, earthquake_file))[0]
        earthquake.resample(sampling_rate=20, window='hanning', no_filter=True, strict_length=False)
        data_dict = {}

        data_dict['data'] = earthquake
        data_dict['label'] = label
        data_dict['name'] = directory.split('/')[-2]

        save_file(os.path.join(target_directory, '%s_%s_%s.p' % (label, data_dict['name'], earthquake_file)), data_dict)

target_directory = '/data/prepared/trigger'
os.makedirs(target_directory, exist_ok=True)
for f in folders:
    load_data(os.path.join(f, 'positive'), 'positive', target_directory)
    load_data(os.path.join(f, 'negative'), 'negative', target_directory)
    #load_data(os.path.join(f, 'chaos'), 'chaos', target_directory)
