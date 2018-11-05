import numpy as np
import torch
import shutil
import os
import json
from networks import FullyConnected
import inspect

def show_model(model):
    print(model)
    num_parameters = 0
    for p in model.parameters():
        if p.requires_grad:
            num_parameters += np.cumprod(p.size())[-1]
    print('Number of parameters: %d' % num_parameters)

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-3] + '_best.h5')

def load_class_from_params(params, class_func):
    arguments = inspect.getfullargspec(class_func).args[1:]
    if 'input_size' not in params and 'input_size' in arguments:
        params['input_size'] = params['length']
    filtered_params = {p: params[p] for p in params if p in arguments}
    return class_func(**filtered_params)

def load_model(run_directory, device_target='cuda'):
    with open(os.path.join(run_directory, 'args.json'), 'r') as f:
        args = json.load(f)

    model = None
    device = None

    if 'spatial' not in run_directory:
        saved_model_path = os.path.join(run_directory, 'checkpoints/latest.h5')
        device = torch.device('cuda', 1) if device_target == 'cuda' else torch.device('cpu')
        class_func = FullyConnected
        model = load_class_from_params(args, class_func).to(device)

        model.eval()
        checkpoint = torch.load(saved_model_path)
        model.load_state_dict(checkpoint['state_dict'])

    return model, args, device

def visualize_embedding(embeddings, labels, output_file):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    pca.fit(embeddings)
    output = pca.transform(embeddings)
    colors = np.argmax(labels, axis=-1)
    plt.style.use('classic')
    plt.scatter(output[:, 0], output[:, 1], c=colors)
    plt.xlabel('PCA0')
    plt.ylabel('PCA1')
    plt.title('Visualization of learned embedding space')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_file)