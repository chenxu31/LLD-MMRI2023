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


    acc_list = numpy.zeros((5,), numpy.float32)
    f1_list = numpy.zeros((5,), numpy.float32)
    recall_list = numpy.zeros((5,), numpy.float32)
    precision_list = numpy.zeros((5,), numpy.float32)
    kappa_list = numpy.zeros((5,), numpy.float32)
    for i in range(1, 6):
        with open(os.path.join(args.output_dir, "val_fold%d" % i, "results.txt"), "r") as f:
            for line in f.readlines():
                if line.find("acc:") >= 0:
                    acc_list[i - 1] = float(line[4:].strip())
                elif line.find("f1:") >= 0:
                    f1_list[i - 1] = float(line[3:].strip())
                elif line.find("recall:") >= 0:
                    recall_list[i - 1] = float(line[7:].strip())
                elif line.find("precision:") >= 0:
                    precision_list[i - 1] = float(line[10:].strip())
                elif line.find("kappa:") >= 0:
                    kappa_list[i - 1] = float(line[6:].strip())

    msg = "acc:%f/%f\nf1:%f/%f\nrecall:%f/%f\nprecision:%f/%f\nkappa:%f/%f" % \
          (acc_list.mean(), acc_list.std(), f1_list.mean(), f1_list.std(), recall_list.mean(), recall_list.std(),
           precision_list.mean(), precision_list.std(), kappa_list.mean(), kappa_list.std())
    print(msg)

    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', type=str, default='uniformer_small_IL', help='name of method')
    parser.add_argument('--data_dir', type=str, default='~/datasets/LLD-MMRI2023/data/classification_dataset/', help='path of the dataset')
    parser.add_argument('--output_dir', type=str, default="~/training/test_output/lld_mmri2023/baseline", help="output dir")
    parser.add_argument('--checkpoint_dir', type=str, default="~/training/checkpoints/lld_mmri2023/baseline/", help="training start epoch")

    args = parser.parse_args()

    main(args)
