import argparse

def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", type=str)
    parser.add_argument("--dataset_directory", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--num_epochs", default=40, type=int)
    parser.add_argument("--transforms", default='demean:raw', type=str)
    parser.add_argument("--length", default=100000, type=int)
    parser.add_argument("--sample_strategy", default='sequential', type=str)
    parser.add_argument("--training_strategy", default='siamese', type=str)
    parser.add_argument("--split", default='', type=str)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--embedding_size", default=10, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--loss_function", choices=['dpcl', 'cl'], default='dpcl', type=str)
    return parser