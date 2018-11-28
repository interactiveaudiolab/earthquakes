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
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pycm import ConfusionMatrix

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
                            augmentations='',
                            length=args.length,
                            split=args.split)

model, _, _ = utils.load_model(args.output_directory, device_target='cuda')
model.to(device)
model.eval()

def get_embeddings(dataset, model):
    dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        drop_last=False)
    
    epoch_loss = []
    embeddings = []
    labels = []
    for data_dict in dataloader:
        data = data_dict['data'].to(device).requires_grad_().float()
        label = data_dict['label'].to(device).float()
        output = model(data).squeeze(1)
        embeddings.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())
    return np.vstack(embeddings), np.vstack(labels)

def train_svm(embeddings, labels):
    print(embeddings.shape, labels.shape)
    #svc = SVC(kernel='rbf')
    svc = KNeighborsClassifier()
    svc.fit(embeddings, np.argmax(labels, axis=-1))
    return svc

print('Getting train embeddings')
embeddings, labels = get_embeddings(dataset, model)
plt.clf()
pca = utils.visualize_embedding(embeddings, labels, os.path.join(args.output_directory, 'tr_viz.png'))

print('Training KNN')
svc = train_svm(embeddings, labels)

print('Predicting on train embeddings with KNN')
predictions = svc.predict(embeddings)
ground_truth = np.argmax(labels, axis=-1)

cm = ConfusionMatrix(predict_vector=predictions, actual_vector=ground_truth)
print(str(cm))

print('Getting test embeddings')
dataset.toggle_split()
embeddings, labels = get_embeddings(dataset, model)
plt.clf()
utils.visualize_embedding(embeddings, labels, os.path.join(args.output_directory, 'tt_viz.png'), pca=pca)

print('Predicting on test embeddings with KNN')
predictions = svc.predict(embeddings)
ground_truth = np.argmax(labels, axis=-1)

cm = ConfusionMatrix(predict_vector=predictions, actual_vector=ground_truth)

print('Results')
print(str(cm))

with open(os.path.join(args.output_directory, 'results.txt'), 'w') as f:
    f.write(str(cm))