# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pdb
import torch
import time
import numpy


def main(args):
    for i in range(1, 6):
        cmd = "python train.py --data_dir %s --train_anno_file %s --val_anno_file %s --batch-size 4 --model %s " \
              "--lr 1e-4 --warmup-epochs 5 --epochs 300 --output %s --experiment %s" % \
              (os.path.join(args.data_dir, "images"), os.path.join(args.data_dir, "labels", "train_fold%d.txt" % i),
               os.path.join(args.data_dir, "labels", "val_fold%d.txt" % i), args.model, args.output_dir, "fold%d" % i)
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='uniformer_small_IL', help='name of method')
    parser.add_argument('--data_dir', type=str, default='~/datasets/LLD-MMRI2023/data/classification_dataset/', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default="~/training/test_output/lld_mmri2023/baseline", help="output dir")

    args = parser.parse_args()

    main(args)
