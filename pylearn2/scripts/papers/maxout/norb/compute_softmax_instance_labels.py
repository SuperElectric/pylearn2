#! /usr/bin/env python
"""
A script for saving the softmax label vectors computed by a model for a
image dataset.
"""

import argparse, os, sys
import numpy, theano
from pylearn2.models.mlp import MLP
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
# from pylearn2.datasets.zca_dataset import ZCA_Dataset


def main():
    floatX = numpy.dtype(theano.config.floatX)

    def parse_args():

        def check_output(output):
            output_dir = os.path.dirname(output)
            if not os.path.isdir(output_dir):
                print 'Directory "%s" does not exist. Exiting.' % output_dir
                sys.exit(1)

            output_ext = os.path.splitext(result.output)[1]
            if output_ext != '.npz':
                print ('Expected --output to have extension ".npz", got "%s"' %
                       output_ext)
                sys.exit(1)

        def get_output_path(model_path, dataset_path):
            def get_basename(path):
                split = os.path.split
                splitext = os.path.splitext
                return splitext(split(path)[1])[0]

            model_directory = os.path.dirname(os.path.abspath(model_path))
            output_filename = '%s_%s.npz' % (get_basename(model_path),
                                             get_basename(dataset_path))
            return os.path.join(model_directory, output_filename)

        parser = argparse.ArgumentParser(
            description=("Computes the soft label vectors for a dataset and "
                         "saves them as a matrix."))

        this_script_dir = os.path.split(os.path.abspath(__file__))[0]

        parser.add_argument('--model',
                            '-m',
                            default=os.path.join(this_script_dir,
                                                 'norb_small.pkl'),
                            help='The .pkl file of the trained model')

        default_dataset_path = os.path.join(os.environ['PYLEARN2_DATA_PATH'],
                                            'norb_small',
                                            'instance_recognition',
                                            'small_norb_02_00_test.pkl')

        parser.add_argument('--dataset',
                            '-d',
                            default=default_dataset_path,
                            help=('The .pkl file of an instance recognition '
                                  'SmallNORB dataset.'))

        parser.add_argument('--output',
                            '-o',
                            default='',
                            help=("The path of the .npz file to save to. "
                                  "Must end in '.npz'. Default: "
                                  "<model_filename>_<dataset_filename>.npz, "
                                  "in the same directory as --model"))

        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=100,
                            help=("The batch size to use when classifying."))

        result = parser.parse_args()

        if result.output == '':
            result.output = get_output_path(result.model, result.dataset)

        check_output(result.output)
        return result

    def load_small_norb_instance_dataset(dataset_path):
        def get_preprocessor_path(dataset_path):
            base_path, extension = os.path.splitext(dataset_path)
            assert extension == '.pkl'
            assert any(base_path.endswith(x) for x in ('train', 'test'))

            if base_path.endswith('train'):
                base_path = base_path[:-5]
            elif base_path.endswith('test'):
                base_path = base_path[:-4]

            return base_path + 'preprocessor.pkl'

        return yaml_parse.load(
            """!obj:pylearn2.datasets.zca_dataset.ZCA_Dataset {
            preprocessed_dataset: !pkl: "%s",
            preprocessor: !pkl: "%s",
            axes: ['c', 0, 1, 'b']
            }""" % (dataset_path, get_preprocessor_path(dataset_path)))

    def get_model_function(model, batch_size):
        """
        Returns an evaluatable function that numerically performs a model's
        fprop method.
        """
        input_space = model.input_space
        input_symbol = input_space.make_theano_batch(name='features',
                                                     dtype=floatX,
                                                     batch_size=batch_size)
        output_symbol = model.fprop(input_symbol)
        return theano.function([input_symbol, ], output_symbol)

    def convert_from_onehot(onehot_labels):
        """
        Converts a BxL matrix (B = batch size, L = # of possible labels)
        of one-hot label vectors, and returns a Bx1 matrix where each
        row just contains the index of the nonzero element in that row.

        For binary one-hot labels (one 1, 0s elsewhere), use binary=True.
        For continuous one-hot labels (e.g. the output of a softmax), use
        binary=False.
        """
        nonzero_indices = onehot_labels.nonzero()
        assert len(nonzero_indices) == len(onehot_labels.shape)
        assert all(len(i) == onehot_labels.shape[0]
                   for i in nonzero_indices)
        return nonzero_indices[-1]

    # def get_error_rate(label, expected_label):
    #     error_mask = (label != expected_label).any(axis=1)
    #     num_errors = numpy.count_nonzero(error_mask)
    #     print "num_correct: ", num_errors
    #     print "label.shape[0]: ", label.shape[0]
    #     result = float(num_errors) / label.shape[0]
    #     print "result: ", result
    #     return result

    args = parse_args()
    test_set = load_small_norb_instance_dataset(args.dataset)
    model = serial.load(args.model)

    # This is just a sanity check. It's not necessarily true; it's just
    # expected to be true in the current use case.
    assert isinstance(model, MLP)

    # batch_size = test_set.X.shape[0]  # this causes a memory error
    batch_size = args.batch_size

    data_specs = (CompositeSpace((model.input_space,
                                  VectorSpace(dim=test_set.y.shape[1]))),
                  ('features', 'targets'))

    model_function = get_model_function(model, batch_size)
    all_labels = numpy.zeros(test_set.y.shape, dtype=floatX)
    all_expected_labels = numpy.zeros([test_set.y.shape[0]], dtype=int)
    # num_errors = 0
    num_data = 0

    # computed_labels = numpy.zeros(test_set.y.shape, floatX)

    for batch_number, (image, expected_label) in \
            enumerate(test_set.iterator(mode='sequential',
                                        batch_size=batch_size,
                                        data_specs=data_specs,
                                        return_tuple=True)):

        label = model_function(image)

        start_index = batch_number * batch_size
        end_index = min(start_index + batch_size, test_set.y.shape[0])
        all_labels[start_index:end_index, :] = label

        expected_label = convert_from_onehot(expected_label)
        all_expected_labels[start_index:end_index] = expected_label

        # soft_confusion_matrix[expected_label[:, 0], :] += label
        # # confusion_matrix += numpy.einsum('ij,ik->jk', label, label)
        # # computed_labels[start_index:end_index, :] = label
        # # print "image, expected label, and label shapes: ", \
        # #       image.shape, expected_label.shape, label.shape

        # label = convert_from_onehot(label, binary=False)
        # hard_confusion_matrix[tuple(expected_label[:, 0]),
        #                       tuple(label[:, 0])] += 1.0
        # num_errors += numpy.count_nonzero(label != expected_label)
        num_data += label.shape[0]
        print "Processed %g %% of %d images" % \
              (100.0 * float(num_data) / test_set.y.shape[0],
               test_set.y.shape[0])

    numpy.savez(args.output,
                labels=all_labels,
                ground_truth=all_expected_labels)
    # soft_confusion_matrix /= float(num_data)
    # hard_confusion_matrix /= float(num_data)


if __name__ == '__main__':
    main()
