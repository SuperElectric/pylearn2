#! /usr/bin/env python
"""
This script loads the instance recognition datasset and creates a ZCA_Dataset
with it, and visualizes both the ZCA'ed and un-ZCA'ed images.
"""

from __future__ import print_function
import sys
from pylearn2.config import yaml_parse
from pylearn2.space import *

def main():

    print("loading zca_dataset")
    zca_dataset = yaml_parse.load(
        """!obj:pylearn2.datasets.zca_dataset.ZCA_Dataset {
            preprocessed_dataset: !pkl: "${PYLEARN2_DATA_PATH}/norb_small/instance_recognition/small_norb_02_00_train.pkl",
            preprocessor: !pkl: "${PYLEARN2_DATA_PATH}/norb_small/instance_recognition/small_norb_02_00_preprocessor.pkl",
            axes: ['c', 0, 1, 'b']
    }""")
    print("loaded zca_dataset")

    space = CompositeSpace((Conv2DSpace(shape=(96, 96), num_channels=1),
                            # ZCA_Dataset converts to one-hot by default
                            VectorSpace(dim=50)))
    sources = ("features", "targets")

    for elem in zca_dataset.iterator(mode='sequential',
                                     batch_size=128,  # from cifar-10.yaml
                                     data_specs=(space, sources)):
        for thing in elem:
            print(thing.shape)
        # print(elem)
        sys.exit(0)

if __name__ == '__main__':
    main()
