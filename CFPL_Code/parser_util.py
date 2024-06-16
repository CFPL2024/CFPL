# coding=utf-8
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataFile',
                        type=str,
                        help='path to dataset')

    parser.add_argument('--fileLabel',
                        type=str,
                        help='path to data label')
    
    parser.add_argument('--extendLabel',
                        type=str)
    
    parser.add_argument('--k',
                        type=int,
                        default=0)

    parser.add_argument('--fileModelSave',
                        type=str,
                        help='path to save model')
    
    parser.add_argument('--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)
    
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('--numFreeze',
                        type=int,
                        help='number of freezed layers in pretrained model, default=12',
                        default=0)

    parser.add_argument('--numDevice',
                        type=int,
                        help='id of gpu ',
                        default=0)
    
    parser.add_argument('--episodeTrain',
                        type=int,
                        help='number of tasks per epoch in training process',
                        default=200)

    parser.add_argument('--episodeVal',
                        type=int,
                        help='number of tasks per epoch in valid process',
                        default=200)

    parser.add_argument('--episodeTest',
                        type=int,
                        help='number of tasks per epoch in testing process',
                        default=100)
    
    parser.add_argument('--numCount',
                        type=int,
                        help='number of intent ',
                        default=10)

    parser.add_argument('--warmup_steps',
                        type=int,
                        help='num of warmup_steps',
                        default=100)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='ratio of decay',
                        default=0.2)

    parser.add_argument('--dropout_rate',
                        type=float,
                        help='ratio of dropout',
                        default=0.1)
    
    parser.add_argument('--lamda',
                        type=float,
                        help='ratio of sim loss',
                        default=0.1)
    
    parser.add_argument('--beta',
                        type=float,
                        default=0.01)
    
    parser.add_argument('--alpha',
                        type=float,
                        default=0.8)
    
    parser.add_argument('--temperature',
                        type=float,
                        default=0.05
                        )
    
    parser.add_argument('--balanced_scalar',
                        type=float,
                        default=0.9)
    
    return parser
