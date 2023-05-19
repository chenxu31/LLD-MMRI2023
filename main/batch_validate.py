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
        cmd = "python validate.py --data_dir %s --val_anno_file %s --model %s --results-dir %s " \
              "--score-dir %s --checkpoint %s" % \
              (os.path.join(args.data_dir, "images"), os.path.join(args.data_dir, "labels", "val_fold%d.txt" % i),
               args.model, os.path.join(args.output_dir, "val_fold%d" % i), os.path.join(args.output_dir, "val_fold%d" % i),
               os.path.join(args.checkpoint_dir, "fold%d" % i, "model_best.pth.tar"))
        os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='uniformer_small_IL', help='name of method')
    parser.add_argument('--data_dir', type=str, default='~/datasets/LLD-MMRI2023/data/classification_dataset/', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default="~/training/test_output/lld_mmri2023/baseline", help="output dir")
    parser.add_argument('--checkpoint_dir', type=str, default="~/training/checkpoints/lld_mmri2023/baseline/", help="training start epoch")

    args = parser.parse_args()

    main(args)
