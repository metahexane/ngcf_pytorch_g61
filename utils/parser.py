'''
Pytorch Implementation of Neural Graph Collaborative Filtering (NGCF) (https://doi.org/10.1145/3331184.3331267)

This file contains the parser function

authors: Mohammed Yusuf Noor, Muhammed Imran Özyar, Calin Vasile Simon
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='Gowella',
                        help='Dataset name: Amazond-book, Gowella, ml-100k')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Store model to path.')
    parser.add_argument('--n_epochs', type=int, default=400,
                        help='Number of epoch.')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='l2 reg.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='number of embeddings.')
    parser.add_argument('--layers', type=str, default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--node_dropout', type=float, default=0.,
                        help='Graph Node dropout.')
    parser.add_argument('--mess_dropout', type=float, default=0.1,
                        help='Message dropout.')
    parser.add_argument('--k', type=str, default=20,
                        help='k order of metric evaluation (e.g. NDCG@k)')
    parser.add_argument('--eval_N', type=int, default=1,
                        help='Evaluate every N epochs')
    parser.add_argument('--save_results', type=int, default=1,
                        help='Save model and results')

    return parser.parse_args()
