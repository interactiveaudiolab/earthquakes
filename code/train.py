from dataset import SiameseDataset, EarthquakeDataset
import torch
from tensorboardX import SummaryWriter
import argparse
from tqdm import trange, tqdm
import numpy as np
import warnings
import pprint
import os
import subprocess
from torch.utils.data import DataLoader, sampler, ConcatDataset
from parser import construct_parser
import json
import networks
from networks import utils
import loss

pp = pprint.PrettyPrinter(indent=4)
torch.manual_seed(0)
warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.monitor_interval = 0

parser = construct_parser()
args = parser.parse_args()

with open(os.path.join(args.output_directory, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if os.path.isdir(os.path.join(args.output_directory, 'checkpoints')):
    if args.overwrite:
        print('Deleting existing directory: ', args.output_directory)
        subprocess.call(['rm','-rf', args.output_directory])
        print("Found existing directory. Sleeping for 5 seconds after deleting existing directory!")

writer = SummaryWriter(log_dir=args.output_directory)
args.log_dir = writer.file_writer.get_logdir()

os.makedirs(os.path.join(args.output_directory, 'checkpoints'), exist_ok=True)

dataset = EarthquakeDataset(folder=args.dataset_directory,
                            transforms=args.transforms,
                            length=args.length,
                            split=args.split)

if args.training_strategy == 'siamese':
    dataset = SiameseDataset(dataset=dataset)

if args.sample_strategy == 'sequential':
    sample_strategy = sampler.SequentialSampler(dataset)
elif args.sample_strategy == 'random':
    sample_strategy = sampler.RandomSampler(dataset)

dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        sampler=sample_strategy,
                        drop_last=True)

model = networks.FullyConnected(input_size=args.length)
model = model.to(device)
utils.show_model(model)


parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters,
                            lr=args.learning_rate)

epochs = trange(args.num_epochs)
loss_function = loss.ContrastiveLoss(margin=2.0)
n_iter = 0
for epoch in epochs:
    progress_bar = trange(len(dataloader))
    epoch_loss = []
    for (data, label) in dataloader:
        data = data.to(device).requires_grad_().float()
        label = label.to(device).float()

        output = model(data)
        _loss = loss_function(output, label)

        _loss.backward()
        optimizer.step()

        progress_bar.set_description(str(_loss))
        progress_bar.update(1)
        writer.add_scalar('iter_loss/scalar', _loss.item(), n_iter)
        epoch_loss.append(_loss.item())
        n_iter += 1

    _loss = np.mean(epoch_loss)
    writer.add_scalar('epoch_loss/scalar', _loss, epoch)

    utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, False, os.path.join(args.log_dir, 'checkpoints', 'latest.h5'))