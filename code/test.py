from dataset import EarthquakeDataset
import torch
from tqdm import trange, tqdm
import warnings
import pprint
import os
from torch.utils.data import DataLoader
from parser import construct_parser
import json
from networks import utils
from argparse import Namespace
import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)
torch.manual_seed(0)
warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.monitor_interval = 0

parser = construct_parser()
args = parser.parse_args()

with open(os.path.join(args.output_directory, 'args.json'), 'r') as f:
    args = json.load(f)

args = Namespace(**args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = EarthquakeDataset(folder=args.dataset_directory,
                            transforms=args.transforms,
                            length=args.length,
                            split=args.split)
dataset.toggle_split()
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        drop_last=True)

model, _, _ = utils.load_model(args.output_directory, device_target='cuda')
model.to(device)

progress_bar = trange(len(dataloader))
epoch_loss = []
embeddings = []
labels = []
for (data, label) in dataloader:
    data = data.to(device).requires_grad_().float()
    label = label.to(device).float()
    output = model(data).squeeze(1)
    embeddings.append(output.cpu().data.numpy())
    labels.append(label.cpu().data.numpy())
    progress_bar.update(1)

embeddings = np.vstack(embeddings)
labels = np.vstack(labels)

with open(os.path.join(args.output_directory, 'output.p'), 'wb') as f:
    data_dict = {'embeddings': embeddings, 'labels': labels}
    pickle.dump(data_dict, f)

utils.visualize_embedding(embeddings, labels, os.path.join(args.output_directory, 'viz.png'))