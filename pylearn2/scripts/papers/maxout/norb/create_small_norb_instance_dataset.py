#! /usr/bin/env python
"""
This script makes an instance recognition dataset out of the small NORB
dataset, by doing the following:

1) Loads both 'test' and 'train' subsets of the NORB datasets together.

2) Splits dataset into new train and test subsets with disjoint views of the
   same objects.

3) Contrast-normalizes and approximately whitens these datasets.
"""

from __future__ import print_function
import sys, os, time, argparse
import numpy
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import serial, string_utils, safe_zip
from pylearn2.datasets import preprocessing
from pylearn2.datasets.new_norb import NORB
from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)


def parse_args():
    """
    Parses the command-line arguments, returns them in a dict, keyed by
    argument name.
    """
    parser = argparse.ArgumentParser(description=
                                     "Preprocesses the small NORB dataset "
                                     "into DenseDesignMatrices suitable "
                                     "for instance recognition "
                                     "experiments.")
    parser.add_argument("-a",
                        "--azimuth_ratio",
                        type=int,
                        required=False,
                        default=2,
                        metavar='A',
                        help=("Use every A'th azimuth as a testing "
                              "instance. To use all azimuths for "
                              "training, enter 0."))
    parser.add_argument("-e",
                        "--elevation_ratio",
                        type=int,
                        required=False,
                        default=0,
                        metavar='E',
                        help=("Use every E'th elevation as a testing "
                              "instance. To use all azimuths for "
                              "training, enter 0."))

    parser.add_argument("-i",
                        "--which_image",
                        type=str,
                        choices=('left', 'right'),
                        default="left",
                        help="Which stereo image to use")

    return parser.parse_args()


def get_output_paths(args):

    def make_output_dir():
        """
        Creates the output directory to save the preprocessed dataset to. Also
        puts a README file in there.

        Returns the full path to the output directory.
        """

        data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/norb_small/')
        output_dir = os.path.join(data_dir, 'instance_recognition')
        serial.mkdir(output_dir)
        with open(os.path.join(output_dir, 'README'), 'w') as readme_file:
            readme_file.write("""
The files in this directory were created by the "%s" script in
pylearn2/scripts/papers/maxout/. Each run of this script generates
the following files:

  README
  train_M_N.pkl
  train_M_N.npy
  test_M_N.pkl
  test_M_N.npy
  preprocessor_M_N.pkl

The digits M and N refer to the "--azimuth-ratio" and "--elevation-ratio"

As you can see, each dataset is stored in a pair of files. The .pkl file
stores the object data, but the image data is stored separately in a .npy file.
This is because pickle is less efficient than numpy when serializing /
deserializing large numpy arrays.

The ZCA whitening preprocessor takes a long time to compute, and therefore is
stored for later use as preprocessor_M_N.pkl.
""" % os.path.split(__file__)[1])

        return output_dir

    output_dir = make_output_dir()

    output_filenames = 'small_norb_%s_%02d_%02d' % (args.which_image,
                                                    args.azimuth_ratio,
                                                    args.elevation_ratio)
    output_paths = tuple(os.path.join(output_dir, f)
                         for f in output_filenames)

    if any(os.path.isfile(x) for x in output_paths):
        for output_path in output_paths:
            if os.path.isfile(output_path):
                print("%s already exists." % output_path)

        print("Exiting...")
        sys.exit(1)

    return output_paths


def split_into_unpreprocessed_datasets(norb, args):
    # Selects one of the two stereo images.
    images = norb.get_topological_view(single_tensor=True)
    image_shape = images.shape[2:]
    images = images[:, 0 if args.which_image == 'left' else 1, ...]
    images = images.reshape(images.shape[0], -1)

    # Gets rowmasks that select training and testing set rows.

    def get_testing_rowmask(norb_dataset,
                            azimuth_ratio,
                            elevation_ratio):
        """
        Returns a row mask that selects the testing set from the merged data.
        """

        azimuth_index, elevation_index = (norb_dataset.label_name_to_index[n]
                                          for n in ('azimuth', 'elevation'))

        assert not None in (azimuth_index, elevation_index)

        labels = norb_dataset.y

        result = numpy.ones(labels.shape[0], dtype='bool')

        # azimuth labels are spaced by 2
        azimuth_modulo = azimuth_ratio * 2

        # elevation labels are integers from 0 to 9, so no need to convert
        elevation_modulo = elevation_ratio

        if azimuth_modulo > 0:
            azimuths = labels[:, azimuth_index]
            result = numpy.logical_and(result, azimuths % azimuth_modulo == 0)

        if elevation_modulo > 0:
            elevations = labels[:, elevation_index]
            result = numpy.logical_and(result,
                                       elevations % elevation_modulo == 0)

        return result

    testing_rowmask = get_testing_rowmask(norb,
                                          args.azimuth_ratio,
                                          args.elevation_ratio)
    training_rowmask = numpy.logical_not(testing_rowmask)

    view_converter = DefaultViewConverter(shape=image_shape)

    # Splits images into training and testing sets
    return tuple(DenseDesignMatrix(X=images[r, :],
                                   y=norb.y[r, :],
                                   view_converter=view_converter)
                 for r in (training_rowmask, testing_rowmask))


def main():

    args = parse_args()
    norb = NORB(which_norb='small', which_set='both')

    datasets = split_into_unpreprocessed_datasets(norb, args)

    # Subtracts each image's mean intensity. Scale of 55.0 taken from
    # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
    for dataset in datasets:
        dataset.X = global_contrast_normalize(dataset.X, scale=55.0)

    # Prepares the output directory and checks against the existence of
    # output files. We do this before ZCA'ing, to make sure we trigger any
    # IOErrors now rather than later.
    output_paths = get_output_paths(args)

    zca = preprocessing.ZCA()

    print("ZCA'ing training set...")
    start_time = time.time()
    datasets[0].apply_preprocessor(preprocessor=zca, can_fit=True)
    print("...done (%g seconds)." % (time.time() - start_time))

    print("ZCA'ing testing set...")
    start_time = time.time()
    datasets[1].apply_preprocessor(preprocessor=zca, can_fit=False)
    print("...done (%g seconds)." % (time.time() - start_time))

    for dataset, output_path in safe_zip(datasets, output_paths):
        serial.save(output_path, dataset)
        npy_path = os.path.splitext(output_path)[0]
        print("saved %s, %s" % (os.path.split(output_path)[1],
                                os.path.split(npy_path)[1]))

if __name__ == '__main__':
    main()
