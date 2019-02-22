import argparse

def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str,
        help='Where the files produced during training go (e.g. model checkpoints).'
    )
    parser.add_argument("--dataset_directory", type=str,
        help='Path to directory containing the dataset to train from.'
    )
    parser.add_argument("--model_type", type=str,
        help="What type of model ot use. Options are 'fc', which is a fully connected network and 'conv' which is a convolutional neural network."
    )
    parser.add_argument("--batch_size", default=40, type=int,
        help="Batch size for training the model. Defaults to 40"
    )
    parser.add_argument("--num_workers", default=1, type=int,
        help="Number of workers to use for generating the dataset. Defaults to 1. Increase for more training efficiency."
    )
    parser.add_argument("--num_epochs", default=40, type=int,
        help="Number of epochs (passes through the data) to train for. Defaults to 40."
    )
    parser.add_argument("--transforms", default='demean:raw', type=str,
        help="Transforms to apply to the data, separated by colons (e.g. demean:raw). Options are demean, raw, lowpass, bandpass, and whiten. Can be combined using colons. lowpass, bandpass, raw are applied to the original data and is concatenated together. See dataset.py for details."
    )

    parser.add_argument("--augmentations", default='', type=str,
        help="Augment the data on the fly. Two options: 'amplitude' and 'noise'. Amplitude makes the input get louder or quieter over the course of the seismogram. Noise adds a bit of a Gaussian noise to the input. Both are done randomly."
    )
    parser.add_argument("--length", default=100000, type=int,
        help="The seismogram is cropped randomly during training. This decides the number of samples to use."
    )
    parser.add_argument("--sample_strategy", default='sequential', type=str,
        help="How to go through the training data each epoch. Options are sequential and random."
    )
    parser.add_argument("--training_strategy", default='siamese', type=str,
        help="Training strategy for the model. Either use siamese training or don't. Probably don't change this from 'siamese', which it defaults to."
    )
    parser.add_argument("--split", default='', type=str,
        help="This takes in a string that identifies the earthquake that will be used for testing. The string should be the same as the name of the folder containing the earthquake."
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float,
        help="Learning rate for training. Defaults to 2e-5."
    )
    parser.add_argument("--embedding_size", default=10, type=int,
        help="Embedding size for the model. Each earthquake seismogram is projected to an embedding space of this size. Defaults to 10."
    )
    parser.add_argument("--weight_decay", default=0, type=float,
        help="L2 regularization of the model. Adjust to be higher if model is overfitting."
    )
    parser.add_argument("--loss_function", choices=['dpcl', 'cl'], default='dpcl', type=str,
        help="Loss function to train model with. Can be 'dpcl', deep clustering loss, or 'cl', contrastive loss. Contrastive loss is not tested and may not work."
    )
    return parser

def construct_data_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str, required=True, help='Processed earthquake files get stored here in pickle files.')
    parser.add_argument("--dataset_directory", type=str, required=True, help='Raw earthquake files get stored here. The folder should have subfolders for each earthquake. Those subfolders should contain subfolders with SAC files with the subfolder name being the label of the earthquake (e.g. /folder/earthquake_name/positive, /folder/earthquake_name/negative')
    parser.add_argument("--accepted_labels", type=str, default='positive:negative', help='Labels to use for training, delineated by : (e.g. positive:negative:chaos). Defaults to positive:negative.')
    return parser