#! /usr/bin/env python
"""
A script for saving the softmax label vectors computed by a model for a
image dataset.
"""

import argparse, os, sys, time
import numpy, theano
from pylearn2.models.mlp import MLP
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils import serial
from pylearn2.scripts.papers.maxout.norb import load_norb_instance_dataset


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

        # Batch size. I chose the default value below to be the empirically
        # observed optimal value for small_norb, read from a non-solid-state
        # drive, processed on a GTX 780. If your hardware differs, try playing
        # with this value to maximize your throughput.
        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=30,  # Best for mkg's machine. YMMV.
                            help=("The batch size to use when classifying. "
                                  "Play with this to find the value that "
                                  "gives you the maximum throughput on your "
                                  "machine. Bigger is not necessarily "
                                  "faster."))

        result = parser.parse_args()

        if result.output == '':
            result.output = get_output_path(result.model, result.dataset)

        check_output(result.output)
        return result

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

    args = parse_args()
    # test_set = load_norb_instance_dataset(args.dataset)
    test_set = serial.load(args.dataset)

    model = serial.load(args.model)

    # This is just a sanity check. It's not necessarily true; it's just
    # expected to be true in the current use case.
    assert isinstance(model, MLP)

    batch_size = args.batch_size
    print "test_set.y.shape: ", test_set.y.shape
    data_specs = (CompositeSpace((model.input_space,
                                  VectorSpace(dim=test_set.y.shape[1]))),
                  ('features', 'targets'))

    model_function = get_model_function(model, batch_size)
    all_computed_ids = numpy.zeros((test_set.y.shape[0], 50), dtype=floatX)
    # all_expected_ids = numpy.zeros([test_set.y.shape[0]], dtype=int)
    num_data = 0

    start_time = time.time()
    for batch_number, (image, object_id) in \
        enumerate(test_set.iterator(mode='sequential',
                                    batch_size=batch_size,
                                    data_specs=data_specs,
                                    return_tuple=True)):

        computed_id = model_function(image)

        start_index = batch_number * batch_size
        end_index = min(start_index + batch_size, test_set.y.shape[0])
        all_computed_ids[start_index:end_index, :] = computed_id

        num_data += computed_id.shape[0]

        percent = 100.0 * float(num_data) / test_set.y.shape[0]
        time_elapsed = time.time() - start_time
        fps = float(num_data) / time_elapsed
        print "Processed %.01f %% of %d images, fps = %g" % \
              (percent, test_set.y.shape[0], fps)

    numpy.savez(args.output,
                softmaxes=all_computed_ids,
                norb_labels=test_set.y,
                dataset_path=args.dataset)

    print "saved output to '%s'" % args.output

if __name__ == '__main__':
    main()
