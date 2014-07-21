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
import sys, os, time, argparse, copy
import numpy
import theano
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

    parser.add_argument("-n",
                        "--which_norb",
                        choices=('big', 'small'),
                        required=True,
                        help="Which NORB dataset to use")

    return parser.parse_args()


def get_output_paths(args):
    """
    Returns the output file paths, in the following order:

      training set .pkl file
      training set images .npy file
      training set labels .npy file
      testing set .pkl file
      testing set images .npy file
      testing set labels .npy file
      preprocessor .pkl file
      preprocessor .npz file

    Creates the output directory if necessary. If not, tests to see if it's
    writeable.
    """

    def make_output_dir():
        """
        Creates the output directory to save the preprocessed dataset to. If it
        already exists, this just tests to make sure it's writeable. Also
        puts a README file in there.

        Returns the full path to the output directory.
        """

        dir_template = ('${PYLEARN2_DATA_PATH}/%s/' %
                        ('norb' if args.which_norb == 'big'
                         else 'norb_small'))
        data_dir = string_utils.preprocess(dir_template)

        output_dir = os.path.join(data_dir, 'instance_recognition')
        if os.path.isdir(output_dir):
            if not os.access(output_dir, os.W_OK):
                print("Output directory %s is write-protected." % output_dir)
                print("Exiting.")
                sys.exit(1)
        else:
            serial.mkdir(output_dir)

        with open(os.path.join(output_dir, 'README'), 'w') as readme_file:
            readme_file.write("""
The files in this directory were created by the "%s" script in
pylearn2/scripts/papers/maxout/. Each run of this script generates
the following files:

  README
  train_M_N.pkl
  train_M_N_images.npy
  train_M_N_labels.npy
  test_M_N_images.pkl
  test_M_N_labels.npy
  preprocessor_M_N.pkl
  preprocessor_M_N.npz

The digits M and N refer to the "--azimuth-ratio" and "--elevation-ratio"

As you can see, each dataset is stored in a pair of files. The .pkl file
stores the object data, but the image data is stored separately in a .npy file.
This is because pickle is less efficient than numpy when serializing /
deserializing large numpy arrays.

The ZCA whitening preprocessor takes a long time to compute, and therefore is
stored for later use as preprocessor_M_N.pkl.
""" % os.path.split(__file__)[1])

        return output_dir  # ends make_output_dir()

    output_dir = make_output_dir()

    filename_prefix = ('%snorb_%s_%02d_%02d_gcn_zca_' %
                       ('small_' if args.which_norb == 'small' else '',
                        args.which_image,
                        args.azimuth_ratio,
                        args.elevation_ratio))

    path_prefix = os.path.join(output_dir, filename_prefix)

    result = []
    for set_name in ('train', 'test'):
        # The .pkl file
        result.append(path_prefix + set_name + '.pkl')

        # The images and labels memmap files (.npz)
        for memmap_name in ('images', 'labels'):
            result.append(path_prefix + set_name + "_" + memmap_name + '.npy')

    # The preprocessor's .pkl and .npz files
    result.extend([path_prefix + 'preprocessor' + extension
                   for extension in ('.pkl', '.npz')])

    return result


def split_into_unpreprocessed_datasets(norb, args):
    """
    Returns (training set, testing set) as a tuple of DenseDesignMatrix'es
    """

    # Selects one of the two stereo images.
    images = norb.get_topological_view(single_tensor=False)
    images = images[0 if args.which_image == 'left' else 1]
    image_shape = images.shape[1:]
    images = images.reshape(images.shape[0], -1)

    labels = norb.y

    # Gets rowmasks that select training and testing set rows.
    def get_testing_rowmask(norb_dataset,
                            azimuth_ratio,
                            elevation_ratio):
        """
        Returns a row mask that selects the testing set from the merged data.
        If there are blank images, these are split according to the same ratio
        as the non-blank images.
        """

        (category_index,
         azimuth_index,
         elevation_index) = (norb_dataset.label_name_to_index[n]
                             for n in ('category', 'azimuth', 'elevation'))

        assert not None in (azimuth_index, elevation_index)

        blank_rowmask = (norb_dataset.y[:, category_index] == 5)
        blank_row_indices = numpy.nonzero(blank_rowmask)
        assert(len(blank_row_indices) == 1)
        blank_row_indices = blank_row_indices[0]
        print("num. of blanks: %d" % len(blank_row_indices))

        # Excludes blanks from the azimuth and elevation filters
        result = numpy.logical_not(blank_rowmask)
        assert numpy.logical_not(result[blank_row_indices]).all()

        # Azimuth labels are spaced by 2
        azimuth_modulo = azimuth_ratio * 2

        # Elevation labels are integers from 0 to 9, so no need to convert
        elevation_modulo = elevation_ratio

        if azimuth_modulo > 0:
            azimuths = labels[:, azimuth_index]
            result = numpy.logical_and(result,
                                       (azimuths % azimuth_modulo) == 0)

        if elevation_modulo > 0:
            elevations = labels[:, elevation_index]
            result = numpy.logical_and(result,
                                       (elevations % elevation_modulo) == 0)

        # testing_fraction: the fraction of the dataset that is testing data.
        num_nonblank = norb_dataset.y.shape[0] - len(blank_row_indices)
        num_nonblank_testing = numpy.count_nonzero(result)
        testing_fraction = num_nonblank_testing / float(num_nonblank)
        assert testing_fraction >= 0
        assert testing_fraction <= 1.0

        # Include <testing_fraction> of the blank images for the testing data.
        rng = numpy.random.RandomState(seed=1234)
        rng.shuffle(blank_row_indices)  # in-place
        num_testing_blanks = int(len(blank_row_indices) * testing_fraction)
        print("Including %d blank images in testing set, %d in training set" %
              (num_testing_blanks,
               len(blank_row_indices) - num_testing_blanks))
        blank_row_indices = blank_row_indices[:num_testing_blanks]

        # all blank indices should be False
        assert numpy.logical_not(result[blank_row_indices]).all()
        result[blank_row_indices] = True

        return result

    testing_rowmask = get_testing_rowmask(norb,
                                          args.azimuth_ratio,
                                          args.elevation_ratio)
    training_rowmask = numpy.logical_not(testing_rowmask)

    mono_view_converter = DefaultViewConverter(shape=image_shape)

    all_output_paths = get_output_paths(args)
    (training_images_path,
     training_labels_path,
     testing_images_path,
     testing_labels_path) = tuple(all_output_paths[i] for i in (1, 2, 4, 5))

    # Splits images into training and testing sets
    # start_time = time.time()
    result = []

    def get_memmap(path, shape, dtype):
        mode = 'r+' if os.path.isfile(path) else 'w+'
        return numpy.lib.format.open_memmap(filename=path,
                                            mode=mode,
                                            dtype=dtype,
                                            shape=shape)

    def human_readable_memory_size(size, precision=2):
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
        siffix_index = 0
        while size > 1024:
            siffix_index += 1  # increment the index of the suffix
            size = size / 1024.0  # apply the division
        return "%.*f %s" % (precision, size, suffixes[siffix_index])

    for (rowmask,
         images_path,
         labels_path,
         set_name) in safe_zip((training_rowmask, testing_rowmask),
                               (training_images_path, testing_images_path),
                               (training_labels_path, testing_labels_path),
                               ('train', 'test')):

        num_rows = numpy.count_nonzero(rowmask)
        images_shape = (num_rows, numpy.prod(image_shape))
        images_dtype = numpy.dtype(theano.config.floatX)
        labels_shape = (num_rows, labels.shape[1])
        labels_dtype = norb.y.dtype

        num_bytes = numpy.sum([numpy.prod(s) * numpy.dtype(d).itemsize
                               for s, d
                               in safe_zip((images_shape, labels_shape),
                                           (images_dtype, labels_dtype))])

        print("allocating image and label memmaps for %sing set (%s total)" %
              (set_name, human_readable_memory_size(num_bytes)))

        X, y = tuple(get_memmap(path, shape, dtype)
                     for path, shape, dtype
                     in safe_zip((images_path, labels_path),
                                 (images_shape, labels_shape),
                                 (images_dtype, labels_dtype)))

        X[...] = images[rowmask, :]
        assert isinstance(X, numpy.memmap), "type(X) = %s" % type(X)

        X /= 255.0  # at least for zca, this makes no difference.

        y[...] = labels[rowmask, :]

        dataset = copy.copy(norb)  # shallow copy
        dataset.X = X
        dataset.y = y
        dataset.view_converter = copy.deepcopy(mono_view_converter)
        result.append(dataset)

    return result


def get_zca_training_set(training_set):
    """
    Returns a design matrix containing one image per short label in
    training_set. If there are blank images, then the returned matrix
    will contain N of them, where N is the average number of images
    per object.
    """

    # bin images according to short label
    labels = training_set.y[:, :5]
    bins = {}
    for row_index, (datum, label_memmap) in enumerate(safe_zip(training_set.X,
                                                               labels)):
        label = tuple(label_memmap)
        if label not in bins:
            bins[label] = [row_index]
        else:
            bins[label].append(row_index)

    row_indices_of_blanks = None
    blank_label = None
    for label, row_indices in bins.iteritems():
        if blank_label is not None:
            # Makes sure that all blank labels are the same (i.e. ended up in
            # the same bin).
            assert label[0] != 5
        elif label[0] == 5:
            row_indices_of_blanks = row_indices
            blank_label = label
            # print("BL: %s" % str(blank_label))

    row_indices = []

    if blank_label is not None:
        # Removes bin of blank images, if there is one.
        del bins[blank_label]

        # Computes avg number of images per object, puts that many blank images
        # in row_indices.
        num_objects = len(frozenset(tuple(x) for x in labels[:, :2]))
        num_object_images = sum(len(b) for b in bins)
        avg_images_per_object = num_object_images / float(num_objects)

        rng = numpy.random.RandomState(seed=9876)
        rng.shuffle(row_indices_of_blanks)
        assert len(row_indices_of_blanks) > avg_images_per_object, \
            ("len(row_indices_of_blanks): %d, avg_images_per_object: %d" %
             (len(row_indices_of_blanks), avg_images_per_object))
        row_indices.extend(row_indices_of_blanks[:int(avg_images_per_object)])

    # Collects one row index for each distinct short label in training_set
    for bin_row_indices in bins.itervalues():
        row_indices.append(bin_row_indices[0])

    return training_set.X[tuple(row_indices), :]


def main():

    args = parse_args()
    norb = NORB(which_norb=args.which_norb,
                which_set='both',
                image_dtype=theano.config.floatX)

    # (training set, testing set)
    datasets = split_into_unpreprocessed_datasets(norb, args)

    # Subtracts each image's mean intensity. Scale of 55.0 taken from
    # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
    for dataset in datasets:
        global_contrast_normalize(dataset.X, scale=55.0, in_place=True)

    # Prepares the output directory and checks against the existence of
    # output files. We do this before ZCA'ing, to make sure we trigger any
    # IOErrors now rather than later.
    output_paths = get_output_paths(args)
    (training_pkl_path,
     training_images_path,
     training_labels_path,
     testing_pkl_path,
     testing_images_path,
     testing_labels_path,
     pp_pkl_path,
     pp_npz_path) = output_paths

    if os.path.isfile(pp_pkl_path) and os.path.isfile(pp_npz_path):
        zca = serial.load(pp_pkl_path)
    else:
        zca = preprocessing.ZCA(use_memmap_workspace=True)
        zca_training_set = get_zca_training_set(datasets[0])

        print("Computing ZCA components using %d images out the %d training "
              "images" % (zca_training_set.shape[0], datasets[0].y.shape[0]))
        start_time = time.time()
        zca.fit(zca_training_set)
        print("\t...done (%g seconds)." % (time.time() - start_time))

        zca.set_matrices_save_path(pp_npz_path)
        serial.save(pp_pkl_path, zca)

        print("saved preprocessor to:")
        for pp_path in (pp_pkl_path, pp_npz_path):
            print("\t%s" % os.path.split(pp_path)[1])

    print("ZCA'ing training set...")
    start_time = time.time()
    pre_X = datasets[0].X.copy()
    datasets[0].apply_preprocessor(preprocessor=zca, can_fit=False)
    abs_diffs = numpy.abs(pre_X.flatten() - datasets[0].X.flatten())
    avg_change = (abs_diffs.sum() / numpy.prod(pre_X.shape))
    print("\tavg pixel value change = %g" % avg_change)
    print("\tmin, max change = %g, %g" % (abs_diffs.min(), abs_diffs.max()))
    assert not (pre_X == datasets[0].X).all()
    print("\t...done (%g seconds)." % (time.time() - start_time))

    print("ZCA'ing testing set...")
    start_time = time.time()
    datasets[1].apply_preprocessor(preprocessor=zca, can_fit=False)
    print("\t...done (%g seconds)." % (time.time() - start_time))

    print("Saving to %s:" % os.path.split(training_pkl_path)[0])

    for (dataset,
         pkl_path,
         images_path,
         labels_path) in safe_zip(datasets,
                                  (training_pkl_path, testing_pkl_path),
                                  (training_images_path, testing_images_path),
                                  (training_labels_path, testing_labels_path)):
        serial.save(pkl_path, dataset)
        for path in (pkl_path, images_path, labels_path):
            print("\t%s" % os.path.split(path)[1])

if __name__ == '__main__':
    main()
