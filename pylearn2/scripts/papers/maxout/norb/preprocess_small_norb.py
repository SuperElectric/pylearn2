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
import sys, argparse, os, time
import numpy, theano
from numpy import logical_and
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import serial, string_utils, safe_zip
from pylearn2.datasets import preprocessing
from pylearn2.datasets.norb import SmallNORB, Norb
from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)


def parse_args():
    """
    Parses the command-line arguments, returns them in a dict, keyed by
    argument name.
    """
    parser = argparse.ArgumentParser(description=
                                     "Preprocesses the SmallNORB dataset "
                                     "into DenseDesignMatrices suitable "
                                     "for instance recognition "
                                     "experiments.")

    ArgumentTypeError = argparse.ArgumentTypeError

    def positive_integer(arg):
        """
        Checks that an argument is a positive integer.
        """

        result = int(arg)
        if result != float(arg):
            raise ArgumentTypeError("%g is not an integer" % arg)

        if result < 1:
            raise ArgumentTypeError("%d is not greater than zero." % arg)

        return result

    parser.add_argument("-n",
                        "--which_norb",
                        choices=['small', 'big'],
                        required=True,
                        help=("'big' for NORB or 'small' for Small NORB"))

    parser.add_argument("-a",
                        "--azimuth_spacing",
                        type=positive_integer,
                        required=False,
                        default=2,
                        metavar='A',
                        help=("Use every A'th azimuth as a training "
                              "instance. To use all azimuths for "
                              "training, enter 1."))
    parser.add_argument("-e",
                        "--elevation_spacing",
                        type=positive_integer,
                        required=False,
                        default=1,
                        metavar='E',
                        help=("Use every E'th elevation as a training "
                              "instance. To use all elevations for "
                              "training, enter 1."))

    parser.add_argument("-l",
                        "--lighting_spacing",
                        type=positive_integer,
                        required=False,
                        default=1,
                        metavar='I',
                        help=("Use every I'th illumination as a training "
                              "instance. To use all illuminations for "
                              "training, enter 1."))

    def between_0_and_1(arg):
        """
        Checks that an argument is a float between 0.0 and 1.0 inclusive.
        """
        result = float(arg)

        if result < 0.0 or result > 1.0:
            raise ArgumentTypeError("%g was not between 0.0 and 1.0 "
                                    "inclusive." % arg)

        return result

    parser.add_argument("-b",
                        "--training_blanks",
                        type=between_0_and_1,
                        required=False,
                        default=None,
                        metavar='B',
                        help=("The fraction of all 'blank' images to use "
                              "in the training set. Not applicable if "
                              "--which_norb == 'small'."))

    # parser.add_argument("--equal_sizes",
    #                     type=bool,
    #                     required=False,
    #                     default=True,
    #                     help=("If True, ensures that the training set and "
    #                           "test set are of equal sizes. If False, the "
    #                           "test set is N-size_of_training_set, where N "
    #                           "is the total number of examples in the "
    #                           "dataset."))

    result = parser.parse_args(sys.argv[1:])

    if result.which_norb == 'small' and result.training_blanks is not None:
        print ("--training_blanks is only applicable for the big NORB "
               "dataset; i.e. when --which_norb is 'big'")
        sys.exit(1)

    return result


def get_training_rowmask(labels,
                         label_type_to_index,
                         elevation_spacing,
                         azimuth_spacing,
                         lighting_spacing,
                         training_blanks = None):
    """
    Returns a row mask that selects the testing set from the merged data.
    """

    label_names = ("elevation", "azimuth", "lighting")
    label_indices = tuple(label_type_to_index[n] for n in label_names)
    label_modulos = (elevation_spacing, azimuth_spacing * 2, lighting_spacing)

    num_labels = labels.shape[0]

    row_mask = numpy.ones(num_labels, dtype=bool)

    for index, modulo in safe_zip(label_indices, label_modulos):
        label_column = labels[:, index]
        row_mask = numpy.logical_and(row_mask, label_column % modulo == 0)

    if training_blanks is not None:
        rng = numpy.random.RandomState(seed=12345)
        blanks_mask = rng.random_sample(num_labels) < training_blanks
        row_mask = numpy.logical_and(row_mask, blanks_mask)

    return row_mask

    # azimuth_index, elevation_index = (SmallNORB.label_type_to_index[n]
    #                                   for n in ('azimuth', 'elevation'))

    # result = numpy.ones(labels.shape[0], dtype='bool')

    # # azimuth labels are spaced by 2
    # azimuth_modulo = azimuth_spacing * 2

    # # elevation labels are integers from 0 to 8, so no need to convert
    # elevation_modulo = elevation_spacing

    # # if azimuth_modulo > 0:
    # azimuths = labels[:, azimuth_index]
    # result = logical_and(result, azimuths % azimuth_modulo == 0)

    # # if elevation_modulo > 0:
    # elevations = labels[:, elevation_index]
    # result = logical_and(result, elevations % elevation_modulo == 0)

    # return result


def load_instance_datasets(which_norb,
                           which_image,
                           elevation_spacing,
                           azimuth_spacing,
                           lighting_spacing,
                           training_blanks):
    """
    Repackages the NORB database training and testing sets by merging them,
    retaining just the left or just the right stereo images, then splitting
    them into two by interleaved azimuth and elevation angles.

    Parameters
    ----------

    which_norb : 'big' or 'small'.

    which_image : int
    Must be 0 or 1. Selects whether to use left or right images, respectively.

    azimuth_spacing : int
    If azimuth_spacing=N, then every Nth azimuth will be used in the training
    set, and the other images will be used in testing set.

    elevation_spacing : int
    If elevation_spacing=N, then every Nth elevation will be used in the
    training set, and the other images will be used in testing set.

    equal_sizes : bool
    If True, then the test set will be constrained to be the same size as the
    training set. If False, then the test set size will be
    N - <size of training set>, which is often bigger than the training set.

    Returns
    -------
    (test_set, training_set): tuple of DenseDesignMatrix'es
    """

    if which_norb not in ('big', 'small'):
        raise ValueError("which_norb must be either 'big' or 'small', not '%s'"
                         % str(which_norb))

    if which_image not in (0, 1):
        raise ValueError("which_image must be 0 or 1, but was %d" %
                         which_image)

    NorbType = SmallNORB if which_norb == 'small' else Norb

    print("Reading %s NORB dataset" % which_norb)
    train = NorbType('train', True)
    test = NorbType('test', True)
    print("read dataset")

    label_type_to_index = train.label_type_to_index

    # Select just the first image of the stereo image pairs.
    image_shape = NorbType.original_image_shape + (1, )
    num_pixels = numpy.prod(image_shape)

    for db in (train, test):
        assert db.X.shape[1] == num_pixels * 2  # * 2, because of stereo pairs

    # Use just one of the two stereo images
    col_start = which_image * num_pixels
    col_end = col_start + num_pixels
    train.X = train.X[:, col_start:col_end]
    test.X = test.X[:, col_start:col_end]

    print("train.X, .y: %s, %s" % (str(train.X.shape), str(train.y.shape)))

    print("Combining %s NORB's training and testing sets..." % which_norb)
    images = numpy.vstack((train.X, test.X))
    labels = numpy.vstack((train.y, test.y))
    print("...done.")

    # print("train.X.dtype: %s" % train.X.dtype)
    # print("images.dtype: %s" % images.dtype)

    # assert str(images.dtype) == theano.config.floatX

    # Free some memory
    del train
    del test

    # new_labels = get_object_ids(labels)
    train_mask = get_training_rowmask(labels,
                                      label_type_to_index,
                                      azimuth_spacing,
                                      elevation_spacing,
                                      lighting_spacing,
                                      training_blanks)
    test_mask = numpy.logical_not(train_mask)

    # def randomly_reduce_size(mask, desired_num_trues):
    #     """
    #     Randomly picks Trues in test_mask to switch to false, until only
    #     desired_num_trues Trues remain.

    #     returns: test_mask with desired_num_trues Trues.
    #     """

    #     true_indices = numpy.nonzero(mask)[0]

    #     numpy.random.shuffle(true_indices)  # shuffles in-place
    #     indices_to_flip = true_indices[desired_num_trues:]
    #     result = numpy.copy(mask)
    #     result[indices_to_flip] = False
    #     assert numpy.count_nonzero(result) == desired_num_trues, \
    #            ("numpy.count_nonzero(result): %d, desired_num_trues: %d" %
    #             (numpy.count_nonzero(result), desired_num_trues))
    #     return result

    # if equal_sizes:
    #     test_mask = randomly_reduce_size(test_mask,
    #                                      numpy.count_nonzero(train_mask))

    view_converter = DefaultViewConverter(shape=image_shape)
    train, test = (DenseDesignMatrix(X=images[row_mask, :],
                                     y=labels[row_mask, :],
                                     view_converter=view_converter)
                   for row_mask in (train_mask, test_mask))

    print("split dataset into %s training examples, %s testing examples." %
          (str(train.X.shape), str(test.X.shape)))

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
""" % os.path.split(__file__)[1])

    return output_dir


def main():
    """ Top-level main() funtion. """

    args = parse_args()

    training_set, testing_set = load_instance_datasets(args.which_norb,
                                                       0,  # which_image
                                                       args.elevation_spacing,
                                                       args.azimuth_spacing,
                                                       args.lighting_spacing,
                                                       args.training_blanks)


    print("applying GCN...")

    for dataset in (training_set, testing_set):
        print("converting from %s to %s" % (dataset.X.dtype, theano.config.floatX))
        dataset.X = numpy.asarray(dataset.X, dtype=theano.config.floatX)

        # Subtracts each image's mean intensity. Scale of 55.0 taken from
        # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
        # dataset.X = global_contrast_normalize(dataset.X, scale=55.0)
        print("GCN'ing %s dataset" % str(dataset.X.shape))
        global_contrast_normalize(dataset.X, scale=55.0, in_place=True)

    print("GCN done. Max pixel value: %g" % dataset.X.max())

    preprocessor = preprocessing.ZCA()

    output_dir = make_output_dir()
    prefix = '%s_norb_E%02d_A%02d_L%02d' % (args.which_norb,
                                            args.elevation_spacing,
                                            args.azimuth_spacing,
                                            args.lighting_spacing)
    if args.which_norb == 'big':
        prefix += '_B%0.2f' % args.training_blanks

    print("ZCA'ing training set")
    training_set.apply_preprocessor(preprocessor=preprocessor, can_fit=True)
    print("ZCA'ing testing set")
    t1 = time.time()
    testing_set.apply_preprocessor(preprocessor=preprocessor, can_fit=False)
    t2 = time.time()
    print("ZCA of testing set took %g secs" % (t2 - t1))

    # prefix = 'small_norb_%02d_%02d' % (args.azimuth_spacing,
    #                                    args.elevation_spacing)


    # for ds in (training_set, testing_set):
    #     print("ds.y.shape: ", ds.y.shape)

    # Saves testing & training datasets
    for dataset, name in zip((training_set, testing_set), ('train', 'test')):

        # The filename, minus the suffix
        basename = "%s_%s" % (prefix, name)

        basepath = os.path.join(output_dir, basename)
        dataset.use_design_loc(basepath + '.npy')
        serial.save(basepath + '.pkl', dataset)
        print("saved %s, %s" % tuple(os.path.split(basepath)[1] + suffix
                                     for suffix in ('.npy', '.pkl')))

    # Garbage-collect training_set, testing_set.
    #
    # Del doesn't necessarily free memory in Python, but numpy manages its own
    # memory, and del(M) does seem to free M's memory when M is a numpy array.
    #
    # See: http://stackoverflow.com/a/16300177/399397
    del training_set, testing_set

    # print("Saving preprocessor with:\n"
    #       "  P_     of shape %s\n"
    #       "  inv_P_ of shape %s\n"
    #       "  mean_  of shape %s" %
    #       tuple(x.shape for x in (preprocessor.P_,
    #                               preprocessor.inv_P_,
    #                               preprocessor.mean_)))

    # Saves preprocessor
    preprocessor_basepath = os.path.join(output_dir, prefix + "_preprocessor")
    preprocessor.set_matrices_save_path(preprocessor_basepath + '.npz')
    serial.save(preprocessor_basepath + '.pkl', preprocessor)

    for suffix in ('.npz', '.pkl'):
        print("saved %s" % (preprocessor_basepath + suffix))

if __name__ == '__main__':
    main()
