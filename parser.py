#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# creator: TAN Zusheng

import argparse
def args_parser():
    '''
    Developed by: TAN Zusheng
    This is the argparse function for model and train/test mode selection
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="1k_data", help="data path of the training data")
    parser.add_argument("--mode", type=str, default="train", help="train is training mode and test is testing mode")
    parser.add_argument("--model", type=str, default="cnn_lstm", help="selecting model")
    parser.add_argument("--epochs", type=int, default=200, help="model training epochs")
    
    args = parser.parse_args()
    return args
