import obspy
import numpy as np
import os
from tqdm import trange
import pickle
from utils import save_file

folders = [os.path.join('/data/raw', x) for x in os.listdir('/data/raw')]

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

    surface_window = read_surface_window(os.path.join(directory, 'surface_window.txt'))
    for i in progress_bar:
        earthquake_file = earthquake_files[i]
        progress_bar.set_description(os.path.join(directory, earthquake_file))
        earthquake = obspy.read(os.path.join(directory, earthquake_file))[0]
        sample_rate = earthquake.stats.sampling_rate
        times, data = earthquake.times(), earthquake.data
        start, stop = surface_window[earthquake_file.split('BH')[0]]
        start, stop = int(sample_rate * start), int(sample_rate * stop)

        data = np.array(data)[start:stop]
        data_dict = {}
        if data.shape[0] == (stop - start):
            earthquake.data = data
            data_dict['data'] = earthquake
            data_dict['label'] = label
            data_dict['name'] = directory.split('/')[-2]
            save_file(os.path.join(target_directory, '%s_%s.p' % (earthquake_file, label)), data_dict)

target_directory = '/data/prepared'
os.makedirs(target_directory, exist_ok=True)
for f in folders:
    load_data(os.path.join(f, 'positive'), 'positive', target_directory)
    load_data(os.path.join(f, 'negative'), 'negative', target_directory)
    load_data(os.path.join(f, 'chaos'), 'chaos', target_directory)