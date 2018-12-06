import obspy
import numpy as np
import os
from tqdm import trange
import pickle
from utils import save_file, parallel_process
from multiprocessing import cpu_count

num_cpu = cpu_count() - 2
data_dir = 'tremor'

folders = [os.path.join('/data/raw/', data_dir, x) for x in os.listdir(os.path.join('/data/raw/', data_dir))]

def read_surface_window(f):
    f = open(f)
    times = [s.rstrip().replace('\t', ' ').split(' ') for s in f.readlines()]
    d = {}
    for time in times:
        d[time[0].split('BH')[0]] = (float(time[1]), float(time[2]))
    f.close()
    return d

def process_file(earthquake_file, directory, target_directory, label):
    earthquake = obspy.read(os.path.join(directory, earthquake_file))[0]
    earthquake.resample(sampling_rate=20, window='hanning', no_filter=True, strict_length=False)
    data_dict = {}

    data_dict['data'] = earthquake
    data_dict['label'] = label
    data_dict['name'] = directory.split('/')[-2]

    save_file(os.path.join(target_directory, '%s_%s_%s.p' % (label, data_dict['name'], earthquake_file)), data_dict)

def load_data(directory, label, target_directory):
    if not os.path.exists(directory):
        return []
    earthquake_files = [{'earthquake_file': x, 'directory': directory, 'target_directory': target_directory, 'label': label} for x in os.listdir(directory) if '.SAC' in x]

    if len(earthquake_files) == 0:
        return []

    parallel_process(earthquake_files, process_file, use_kwargs=True, n_jobs=num_cpu)        

target_directory = os.path.join('/data/prepared', data_dir)
os.makedirs(target_directory, exist_ok=True)
for f in folders:
    load_data(os.path.join(f, 'positive'), 'positive', target_directory)
    load_data(os.path.join(f, 'negative'), 'negative', target_directory)
    #load_data(os.path.join(f, 'chaos'), 'chaos', target_directory)
