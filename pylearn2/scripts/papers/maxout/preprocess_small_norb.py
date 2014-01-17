#! /usr/bin/env python
"""
This script makes an instance recognition dataset out of the Small NORB
dataset, by doing the following:

1) Merges SmallNORB's "train" and "test" datasets.
2) Replaces category-instance label pairs with unique instance labels.
3) Splits dataset into train and test datasets with interleaved azimuths.
4) Contrast-normalizes and approximately whitens the database.
"""

from __future__ import print_function
import sys, argparse, numpy, os
from numpy import logical_and
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import serial, string_utils
from pylearn2.datasets import preprocessing
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


def parse_args():
    """
    Parses the command-line arguments, returns them in a dict, keyed by
    argument name.
    """
    parser = argparse.ArgumentParser(description=
                                     "Preprocesses the SmallNORB dataset "
                                     "into DenseDesignMatrices suitable "
                                     "for instance recognition "
                                     "experiements.")
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

    return parser.parse_args(sys.argv[1:])


def get_new_labels(labels):
    """
    Given a NxM matrix of NORB labels, returns a Nx1 matrix of new labels
    that assigns a unique integer for each (category, instance) pair.
    """
    result = numpy.zeros((labels.shape[0], 1), dtype='int')
    category_index, instance_index = (SmallNORB.label_type_to_index[n]
                                      for n in ('category', 'instance'))

    num_categories, num_instances = (SmallNORB.num_labels_by_type[x]
                                     for x in (category_index,
                                               instance_index))

    new_label = 0
    examples_per_label = 0
    for category in xrange(num_categories):
        for instance in xrange(num_instances):
            row_mask = logical_and(labels[:, category_index] == category,
                                   labels[:, instance_index] == instance)
            if new_label == 0:
                examples_per_label = len(numpy.nonzero(row_mask))
                assert examples_per_label != 0
            else:
                assert len(numpy.nonzero(row_mask)) == examples_per_label

            result[row_mask] = new_label
            new_label = new_label + 1

    return result


def get_testing_rowmask(labels, azimuth_ratio, elevation_ratio):
    """
    Returns a row mask that selects the testing set from the merged data.
    """

    azimuth_index, elevation_index = (SmallNORB.label_type_to_index[n]
                                      for n in ('azimuth', 'elevation'))

    result = numpy.ones(labels.shape[0], dtype='bool')

    # azimuth labels are spaced by 2
    azimuth_modulo = azimuth_ratio * 2

    # elevation labels are integers from 0 to 9, so no need to convert
    elevation_modulo = elevation_ratio

    if azimuth_modulo > 0:
        azimuths = labels[:, azimuth_index]
        result = logical_and(result, azimuths % azimuth_modulo == 0)

    if elevation_modulo > 0:
        elevations = labels[:, elevation_index]
        result = logical_and(result, elevations % elevation_modulo == 0)

    return result


def load_instance_datasets(azimuth_ratio, elevation_ratio):
    """
    Repackages the NORB database as an instance-recognition database with 50
    distinct instances. Returns two DenseDesignMatrix instances; the training
    and testing sets.

    The original NORB database uses distinct objects for training and testing.
    This would be meaningless in an instance recognition database. Therefore,
    this function merges the two datasets, then for each object, it selects
    every <azimuth_ratio>'th azimuth and <elevation_ratio>'th elevation image
    to be in the testing set.
    """

    print("Reading SmallNORB")
    train = SmallNORB('train', True)
    test = SmallNORB('test', True)
    print("read SmallNORB")

    images = numpy.vstack((train.X, test.X))
    labels = numpy.vstack((train.y, test.y))

    new_labels = get_new_labels(labels)
    test_mask = get_testing_rowmask(labels, azimuth_ratio, elevation_ratio)
    train_mask = numpy.logical_not(test_mask)

    view_converter = train.view_converter
    train = DenseDesignMatrix(X=images[train_mask, :],
                              y=new_labels[train_mask, :],
                              view_converter=view_converter)
    test = DenseDesignMatrix(X=images[test_mask, :],
                             y=new_labels[test_mask, :],
                             view_converter=view_converter)

    print("split dataset into %d training examples, %d testing examples." % \
          (train.X.shape[0], test.X.shape[0]))

    return (train, test)


def make_output_dir():
    """
    Creates the output directory to save the preprocessed dataset to. Also puts
    a README file in there.

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
""")

    return output_dir


def main():
    """ Top-level main() funtion. """

    args = parse_args()
    training_set, testing_set = load_instance_datasets(args.azimuth_ratio,
                                                       args.elevation_ratio)

    for dataset in (training_set, testing_set):
        # Subtracts each image's mean intensity. Scale of 55.0 taken from
        # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
        dataset.X = global_contrast_normalize(dataset.X, scale=55.0)

    preprocessor = preprocessing.ZCA()
    print("ZCA'ing training_set")
    training_set.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
    print("ZCA'ing testing set")
    testing_set.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
    print("finished preprocessing")
    output_dir = make_output_dir()

    prefix = 'small_norb_%02d_%02d' % (args.azimuth_ratio,
                                       args.elevation_ratio)

    for dataset, name in zip((training_set, testing_set), ('train', 'test')):

        # The filename, minus the suffix
        basename = "%s_%s" % (prefix, name)

        basepath = os.path.join(output_dir, basename)
        dataset.use_design_loc(basepath + '.npy')
        serial.save(basepath + '.pkl', dataset)
        print("saved %s, %s" % tuple(os.path.split(basepath)[1] + suffix
                                for suffix in ('.npy', '.pkl')))

    preprocessor_filename = prefix +"_preprocessor.pkl"
    serial.save(os.path.join(output_dir, preprocessor_filename),
                preprocessor)
    print("saved %s" % preprocessor_filename)

if __name__ == '__main__':
    main()
