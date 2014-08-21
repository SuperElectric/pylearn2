#! /usr/bin/env python

import argparse
import os.path
import numpy
import theano
from pylearn2.utils import serial


def _parse_args():
    parser = argparse.ArgumentParser("Converts weixun's .npz files to "
                                     "something readable by viewer.")
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help="Weixun's .npz file, containing 'S' and 'L'")

    parser.add_argument('-d',
                        '--dataset',
                        default=os.path.join('norb_small',
                                             'instance_recognition',
                                             'small_norb_00_00_train.pkl'),
                        help=("path of the instance dataset used to compute "
                              "the .npz file. The elements in the "
                              "dataset need not be in the same order as the "
                              "corresponding elements in the .npz file. "
                              "(Correspondence will be established by "
                              "matching labels.)"))
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help=("Output .npz file, suitable for use with "
                              "view_confusion_matrix.py"))

    result = parser.parse_args()

    # check --input
    assert os.path.isfile(result.input)
    assert os.path.splitext(result.input)[1] == '.npz'

    # check --dataset, turn it into a realtive path
    assert os.path.isfile(result.dataset)
    assert os.path.splitext(result.dataset)[1] == '.pkl'
    data_dir = os.environ['PYLEARN2_DATA_PATH']
    assert result.dataset.startswith(data_dir)
    result.dataset = os.path.relpath(result.dataset, start=data_dir)

    return result


def main():
    args = _parse_args()
    npz_dict = numpy.load(args.input)
    for key in ('S', 'L'):
        assert key in npz_dict, "Couldn't find key '%s' in npz file"

    softmaxes = npz_dict['S']
    assert (softmaxes >= 0.0).all()
    assert (softmaxes <= 1.0).all()
    assert softmaxes.shape[1] == 50

    labels = npz_dict['L']
    assert (labels >= 0).all()
    assert labels.shape[1] == 5

    assert softmaxes.shape[0] == labels.shape[0]

    labels = numpy.asarray(labels, dtype='int32')
    softmaxes = numpy.asarray(softmaxes, dtype=theano.config.floatX)

    # Checks that the dataset labels are one-to-one and onto with weixun's
    # labels
    dataset_abspath = os.path.join(os.environ['PYLEARN2_DATA_PATH'],
                                   args.dataset)
    dataset = serial.load(dataset_abspath)

    def get_unique_labels(labels):
        result = frozenset(tuple(label) for label in labels)
        assert len(result) == labels.shape[0]
        return result

    dataset_unique_labels = get_unique_labels(dataset.y)
    weixun_unique_labels = get_unique_labels(labels)
    assert dataset_unique_labels == weixun_unique_labels

    numpy.savez(args.output,
                softmaxes=softmaxes,
                norb_labels=labels,
                dataset_path=args.dataset)

if __name__ == '__main__':
    main()
