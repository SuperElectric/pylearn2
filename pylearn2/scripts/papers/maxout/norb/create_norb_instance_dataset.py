#! /usr/bin/env python
"""
This script makes an instance recognition dataset out of the small NORB
dataset, by doing the following:

1) Loads both 'test' and 'train' subsets of the NORB datasets together.

2) Splits dataset into new train and test subsets with disjoint views of the
   same objects.

3) Saves the split datasets individually

4) Optionally preprocesses them, and saves the preprocessed versions, along with the preprocessor, if applicable.

For example, if you call:
  ./create_norb_instance_dataset -a 2 -e 1 -i left -n small -p gcn-zca

You'd create the following files, in
{PYLEARN2_DATA_PATH}/norb_small/instance_recognition/small_norb_left_02_01/:

  raw_train.pkl
  raw_train_images.npy
  raw_train_labels.npy

  raw_test.pkl
  raw_test_images.npy
  raw_test_labels.npy

  gcn-zca_preprocessor.pkl

  gcn-zca_train.pkl
  gcn-zca_train_images.npy
  gcn-zca_train_labels.npy

  gcn-zca_test.pkl
  gcn-zca_test_images.npy
  gcn-zca_test_labels.npy


"""

from __future__ import print_function
import sys, os, time, argparse, copy
import numpy
import theano
#from numpy import logical_and, logical_not
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import serial, string_utils, safe_zip
from pylearn2.datasets import preprocessing
from pylearn2.datasets.new_norb import NORB
from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)
from pylearn2.scripts.papers.maxout.norb import human_readable_memory_size


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
                        help=("Use every A'th azimuth as a testing instance. "
                              "To use all azimuths, enter 1. To use none, "
                              "enter 0."))

    parser.add_argument("-e",
                        "--elevation_ratio",
                        type=int,
                        required=False,
                        default=1,
                        metavar='E',
                        help=("Use every E'th elevation as a testing "
                              "instance. To use all azimuths, enter 1. To use "
                              "none, enter 0."))

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

    parser.add_argument("-p",
                        "--preprocessor",
                        choices=('gcn-zca', 'lcn', 'none'),
                        required=True,
                        help="Which preprocessor to use, if any")

    result = parser.parse_args()

    if (result.azimuth_ratio == 0) != (result.elevation_ratio == 0):
        print ("--azimuth-ratio and --elevation-ratio must either both be "
               "positive, or both be zero. Exiting...")
        sys.exit(1)

    if result.azimuth_ratio == 1 and \
       result.elevation_ratio == 1 and \
       result.preprocessor == 'gcn-zca':
        print ("Both --azimuth-ratio and --elevation-ratio are 1, making the "
               "training set empty. The gcn-zca --preprocessor requires a "
               "non-empty training set. Exiting...")
        sys.exit(1)

    return result


def get_output_dir(args):
    """
    Returns the path to the output directory. Creates any missing directories
    in the path, as needed.
    """
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
    assert os.path.isdir(data_dir)

    dirs = [data_dir,
            'norb' if args.which_norb == 'big' else 'norb_small',
            'instance_recognition',
            '%s_%02d_%02d' % (args.which_image,
                              args.azimuth_ratio,
                              args.elevation_ratio)]
    return os.path.join(*dirs)


# def get_output_paths(args):
#     """
#     Returns the output file paths, in the following order:

#       training set .pkl file
#       training set images .npy file
#       training set labels .npy file
#       testing set .pkl file
#       testing set images .npy file
#       testing set labels .npy file
#       preprocessor .pkl file
#       preprocessor .npz file

#     Creates the output directory if necessary. If not, tests to see if it's
#     writeable.
#     """

#     def make_output_dir():
#         """
#         Creates the output directory to save the preprocessed dataset to. If it
#         already exists, this just tests to make sure it's writeable. Also
#         puts a README file in there.

#         Returns the full path to the output directory.
#         """

#         dir_template = ('${PYLEARN2_DATA_PATH}/%s/' %
#                         ('norb' if args.which_norb == 'big'
#                          else 'norb_small'))
#         data_dir = string_utils.preprocess(dir_template)

#         output_dir = os.path.join(data_dir, 'instance_recognition')
#         if os.path.isdir(output_dir):
#             if not os.access(output_dir, os.W_OK):
#                 print("Output directory %s is write-protected." % output_dir)
#                 print("Exiting.")
#                 sys.exit(1)
#         else:
#             serial.mkdir(output_dir)

#         with open(os.path.join(output_dir, 'README'), 'w') as readme_file:
#             readme_file.write("""
# The files in this directory were created by the "%s" script in
# pylearn2/scripts/papers/maxout/. Each run of this script generates
# the following files:

#   README
#   train_M_N.pkl
#   train_M_N_images.npy
#   train_M_N_labels.npy
#   test_M_N_images.pkl
#   test_M_N_labels.npy
#   preprocessor_M_N.pkl
#   preprocessor_M_N.npz

# The digits M and N refer to the "--azimuth-ratio" and "--elevation-ratio"

# As you can see, each dataset is stored in a pair of files. The .pkl file
# stores the object data, but the image data is stored separately in a .npy file.
# This is because pickle is less efficient than numpy when serializing /
# deserializing large numpy arrays.

# The ZCA whitening preprocessor takes a long time to compute, and therefore is
# stored for later use as preprocessor_M_N.pkl.
# """ % os.path.split(__file__)[1])

#         return output_dir  # ends make_output_dir()

#     output_dir = make_output_dir()

#     filename_prefix = ('%snorb_%s_%02d_%02d_gcn_zca_' %
#                        ('small_' if args.which_norb == 'small' else '',
#                         args.which_image,
#                         args.azimuth_ratio,
#                         args.elevation_ratio))

#     path_prefix = os.path.join(output_dir, filename_prefix)

#     result = []
#     for set_name in ('train', 'test'):
#         # The .pkl file
#         result.append(path_prefix + set_name + '.pkl')

#         # The images and labels memmap files (.npz)
#         for memmap_name in ('images', 'labels'):
#             result.append(path_prefix + set_name + "_" + memmap_name + '.npy')

#     # The preprocessor's .pkl and .npz files
#     result.extend([path_prefix + 'preprocessor' + extension
#                    for extension in ('.pkl', '.npz')])

#     return result


def get_raw_datasets(norb,
                     azimuth_ratio,
                     elevation_ratio,
                     which_image,
                     output_dir):
    """
    Splits norb into two DenseDesignMatrices, saves them to output_dir,
    and returns them.

    If they already exist in output_dir, this just loads and returns them.

    Parameters:
    -----------
    norb : pylearn2.datasets.new_norb.NORB
      A NORB dataset, instantiated with which_set='both'.

    azimuth_ratio : int
      See --azimuth_ratio argument.

    elevation_ratio : int
      See --elevation_ratio argument.

    which_image : str
      See --which_image argument.

    output_dir : str
      The path to the output directory to save these datasets to.

    Returns : training_set, testing_set
    """

    assert azimuth_ratio >= 0
    assert elevation_ratio >= 0
    assert which_image in ('left', 'right')

    set_names = ['train', 'test']
    output_paths = [os.path.join(output_dir, 'raw_%s.pkl' % s)
                    for s in set_names]

    # If the files already exist, no need to create them again.
    if all(os.path.isfile(output_path) for output_path in output_paths):
        return [serial.load(output_path) for output_path in output_paths]

    # Selects one of the two stereo images.
    images = norb.get_topological_view(single_tensor=False)
    images = images[0 if which_image == 'left' else 1]
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

        all_blanks_rowmask = (norb_dataset.y[:, category_index] == 5)

        def get_angle_rowmask():
            """
            Returns the row indices corresponding to camera views that
            satisfy both azimuth_ratio and elevation_ratio.

            Blank images are excluded, since their camera angles are undefined.
            """

            assert (azimuth_ratio == 0) == (elevation_ratio == 0)

            # special case when user chooses to use all data for training
            if azimuth_ratio == 0 and elevation_ratio == 0:
                return numpy.zeros(norb_dataset.y.shape[0], dtype=bool)
            else:
                assert azimuth_ratio > 0 and elevation_ratio > 0, \
                    ("Illegal arguments. This should've been caught in "
                     "parse_args.")

            # azimuth labels are spaced by 2, hence the "* 2"
            azimuths = labels[:, azimuth_index]
            azimuth_rowmask = ((azimuths % (azimuth_ratio * 2) == 0))

            elevations = labels[:, elevation_index]
            elevation_rowmask = ((elevations % elevation_ratio) == 0)
            logical_and, logical_not = (numpy.logical_and, numpy.logical_not)
            return logical_and(logical_not(all_blanks_rowmask),
                               logical_and(azimuth_rowmask, elevation_rowmask))

        result = get_angle_rowmask()

        def add_testing_blanks(result):
            """
            Divides up the blank images in the same ratio as the nonblank
            images, and add some random selection of blank images into result.
            """

            # testing_fraction: the fraction of the dataset that is testing
            # data.
            blank_row_indices = numpy.nonzero(all_blanks_rowmask)[0]
            num_nonblank = norb_dataset.y.shape[0] - len(blank_row_indices)
            num_nonblank_testing = numpy.count_nonzero(result)
            testing_fraction = num_nonblank_testing / float(num_nonblank)
            assert testing_fraction >= 0
            assert testing_fraction <= 1.0

            # Include <testing_fraction> of the blank images for the testing
            # data.
            rng = numpy.random.RandomState(seed=1234)
            rng.shuffle(blank_row_indices)  # in-place
            num_testing_blanks = int(len(blank_row_indices) * testing_fraction)
            testing_blank_row_indices = blank_row_indices[:num_testing_blanks]

            # Add in the testing blanks
            result[testing_blank_row_indices] = True

        add_testing_blanks(result)

        # Print the # of training vs testing
        num_total = norb_dataset.y.shape[0]
        num_testing = numpy.count_nonzero(result)
        num_training = num_total - num_testing
        testing_fraction = (num_testing / float(num_total))
        training_fraction = 1.0 - testing_fraction
        print ("%d (%d%%) training images, %d (%d%%) testing images." %
               (num_training,
                int(training_fraction * 100),
                num_testing,
                int(testing_fraction * 100)))

        return result

    testing_rowmask = get_testing_rowmask(norb,
                                          azimuth_ratio,
                                          elevation_ratio)
    training_rowmask = numpy.logical_not(testing_rowmask)

    rowmasks = [training_rowmask, testing_rowmask]

    image_paths = [os.path.join(output_dir, 'raw_%s_images.npy' % s)
                   for s in set_names]

    label_paths = [os.path.join(output_dir, 'raw_%s_labels.npy' % s)
                   for s in set_names]

    result = []

    def get_memmap(path, shape, dtype):
        mode = 'r+' if os.path.isfile(path) else 'w+'
        result = numpy.memmap(filename=path,
                              mode=mode,
                              dtype=dtype,
                              shape=shape)

        if result.shape != shape:
            raise IOError("Asked numpy for a %s memmap, but got a %s "
                          "memmap." % (str(shape), str(result.shape)))

        return result

    mono_view_converter = DefaultViewConverter(shape=image_shape)

    for (rowmask,
         images_path,
         labels_path,
         set_name) in safe_zip(rowmasks,
                               image_paths,
                               label_paths,
                               set_names):

        print("Set: %s" % set_name)

        num_rows = numpy.count_nonzero(rowmask)

        if num_rows == 0:
            print("  Skipping allocation of empty %sing set" % set_name)
            result.append(None)
        else:
            images_shape = (num_rows, numpy.prod(image_shape))
            images_dtype = numpy.dtype(theano.config.floatX)
            labels_shape = (num_rows, labels.shape[1])
            labels_dtype = norb.y.dtype

            print("  images_shape: %s" % str(images_shape))

            num_bytes = numpy.sum([numpy.prod(s) * numpy.dtype(d).itemsize
                                   for s, d
                                   in safe_zip((images_shape, labels_shape),
                                               (images_dtype, labels_dtype))])

            print("  allocating image and label memmaps for %sing set "
                  "(%d rows, %s total)" %
                  (set_name, num_rows, human_readable_memory_size(num_bytes)))

            X, y = tuple(get_memmap(path, shape, dtype)
                         for path, shape, dtype
                         in safe_zip((images_path, labels_path),
                                     (images_shape, labels_shape),
                                     (images_dtype, labels_dtype)))

            # print("%s: X.shape: %s images[rowmask, :].shape: %s" %
            #       (set_name, str(X.shape), str(images[rowmask, :].shape)))
            X[...] = images[rowmask, :]
            assert isinstance(X, numpy.memmap), "type(X) = %s" % type(X)

            # The expected convention for floating-point pixels is that
            # they are in the range [0.0, 1.0]
            X /= 255.0

            y[...] = labels[rowmask, :]

            dataset = copy.copy(norb)  # shallow copy
            dataset.X = X
            dataset.y = y
            dataset.view_converter = copy.deepcopy(mono_view_converter)
            result.append(dataset)

    for output_dataset, output_path in safe_zip(result, output_paths):
        if output_dataset is not None:
            serial.save(output_path, output_dataset)

    return result


def preprocess_lcn(datasets):
    raise NotImplementedError()


def preprocess_gcn_zca(datasets, preprocessor_path):
    assert datasets[0] is not None
    assert len(datasets) == 2

    # GCN: Subtracts each image's mean intensity. Scale of 55.0 taken from
    # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
    for dataset in datasets:
        if dataset is not None:
            global_contrast_normalize(dataset.X, scale=55.0, in_place=True)

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
        for (row_index,
             (datum, label_memmap)) in enumerate(safe_zip(training_set.X,
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
                # Makes sure that all blank labels are the same (i.e. ended up
                # in the same bin).
                assert label[0] != 5
            elif label[0] == 5:
                row_indices_of_blanks = row_indices
                blank_label = label
                # print("BL: %s" % str(blank_label))

        row_indices = []

        if blank_label is not None:
            # Removes bin of blank images, if there is one.
            del bins[blank_label]

            # Computes avg number of images per object, puts that many blank
            # images in row_indices.
            num_objects = len(frozenset(tuple(x) for x in labels[:, :2]))
            num_object_images = sum(len(b) for b in bins)
            avg_images_per_object = num_object_images / float(num_objects)

            rng = numpy.random.RandomState(seed=9876)
            rng.shuffle(row_indices_of_blanks)  # in-place

            assert len(row_indices_of_blanks) > avg_images_per_object, \
                ("len(row_indices_of_blanks): %d, avg_images_per_object: %d" %
                 (len(row_indices_of_blanks), avg_images_per_object))

            avg_images_per_object = int(avg_images_per_object)
            row_indices.extend(row_indices_of_blanks[:avg_images_per_object])

        # Collects one row index for each distinct short label in training_set
        for bin_row_indices in bins.itervalues():
            row_indices.append(bin_row_indices[0])

        return training_set.X[tuple(row_indices), :]

    preprocessor_npz_path = os.path.splitext(preprocessor_path)[0] + '.npz'

    # Load an existing ZCA preprocessor, or create one if none exists.
    if os.path.isfile(preprocessor_path) and \
       os.path.isfile(preprocessor_npz_path):
        zca = serial.load(preprocessor_path)
        print("Loaded existing preprocessor %s" % preprocessor_path)
    elif datasets[0] is not None:
        zca = preprocessing.ZCA(use_memmap_workspace=True)
        zca_training_set = get_zca_training_set(datasets[0])

        print("Computing ZCA components using %d images out the %d "
              "training images" %
              (zca_training_set.shape[0], datasets[0].y.shape[0]))
        start_time = time.time()
        zca.fit(zca_training_set)
        print("\t...done (%g seconds)." % (time.time() - start_time))

        zca.set_matrices_save_path(preprocessor_npz_path)
        serial.save(preprocessor_path, zca)

        print("saved preprocessor to:")
        for pp_path in (preprocessor_path, preprocessor_npz_path):
            print("\t%s" % os.path.split(pp_path)[1])

    # ZCA the training set
    print("ZCA'ing training set...")
    start_time = time.time()
    datasets[0].apply_preprocessor(preprocessor=zca,
                                   can_fit=False)  # sic; fit() called above.
    print("\t...done (%g seconds)." % (time.time() - start_time))

    # ZCA the testing set
    if datasets[1] is not None:
        print("ZCA'ing testing set...")
        start_time = time.time()
        datasets[1].apply_preprocessor(preprocessor=zca, can_fit=False)
        print("\t...done (%g seconds)." % (time.time() - start_time))


def preprocess_and_save_datasets(raw_datasets, preprocessor_name, output_dir):
    """
    Takes a pair of raw datasets, makes deep copies of them, and preprocesses
    the copies in-place. The datasets are saved in canonically-named paths.
    If the preprocessor is data-dependent, it too is saved.

    Parameters:
    -----------
    raw_datasets: list or tuple
      A list/tuple of two datasets. Either can be None (though not both).

    preprocessor_name: str
      The --preprocessor argument.
    """
    assert preprocessor_name != 'none'
    assert len(raw_datasets) == 2
    for raw_dataset in raw_datasets:
        assert raw_dataset is None or isinstance(raw_dataset, NORB)

    # The prefix shared by all output files' full paths.
    base_path = os.path.join(output_dir, preprocessor_name)

    preprocessed_dataset_paths = [base_path + '_%s.pkl' % set_name
                                  for set_name in ('train', 'test')]

    if all(os.path.isfile(p) for p in preprocessed_dataset_paths):
        print("Both preprocessed datasets already exist. Exiting without "
              "touching them.")
        return

    def make_dataset_to_preprocess(raw_dataset, output_path):
        """
        Returns a deep copy of raw_dataset.

        Parameters:
        -----------

        raw_dataset: NORB
          The set to be deep-copied. Can be None, in which case this function
          does nothing, returning None.

        output_path:
          The path that the set will be saved under. This will be used
          to name the copy's memmap files. For example, if path is foo/bar.pkl,
          the memmaps will be foo/bar_images.npy and foo/bar_labels.npy.
        """
        assert set_name in ('train', 'test')

        if raw_dataset is None:
            return None

        assert isinstance(raw_dataset, NORB)

        def deep_copy(raw_dataset, memmap_paths):
            """
            Returns a deep copy of raw_dataset. Saves the new X and y
            memmaps to memmap_paths.
            """

            def copy_memmap(memmap_to_copy, memmap_path):
                """
                Makes a copy of a memmap to memmap_path.

                If the memmap file already exists, opens it in read-only mode,
                and asserts that its contents are already identical to
                memmap_to_copy.
                """
                mode = 'r' if os.path.isfile(memmap_path) else 'w+'
                result = numpy.memmap(filename=memmap_path,
                                      mode=mode,
                                      shape=memmap_to_copy.shape,
                                      dtype=memmap_to_copy.dtype)

                if mode == 'r':
                    assert numpy.all(result == memmap_to_copy)
                else:
                    result[...] = memmap_to_copy

                return result

            shallow_copy = copy.copy(raw_dataset)
            shallow_copy.X = None
            shallow_copy.y = None
            result = copy.deepcopy(shallow_copy)
            result.X = copy_memmap(raw_dataset.X, memmap_paths[0])
            result.y = copy_memmap(raw_dataset.y, memmap_paths[1])

            return result

        basepath = os.path.splitext(output_path)[0]
        memmap_paths = [basepath + "_%s.npy" % tensor_name
                        for tensor_name in ('images', 'labels')]

        result_dataset = deep_copy(raw_dataset, memmap_paths)
        result_path = basepath + ".pkl"
        return result_dataset, result_path

    preprocessed_datasets = [make_dataset_to_preprocess(r, d)
                             for r, d in safe_zip(raw_datasets,
                                                  preprocessed_dataset_paths)]

    if preprocessor_name == 'gcn-zca':
        preprocessor_path = base_path + ".pkl"
        preprocess_gcn_zca(preprocessed_datasets, preprocessor_path)
    elif preprocessor_name == 'lcn':
        preprocess_lcn(preprocessed_datasets)

    for d, p in safe_zip(preprocessed_datasets, preprocessed_dataset_paths):
        serial.save(d, p)


def main():

    args = parse_args()
    norb = NORB(which_norb=args.which_norb,
                which_set='both',
                image_dtype=theano.config.floatX)

    output_dir = get_output_dir(args)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)  # also creates any missing parent dirs.

    # Splits norb into training and testing sets.
    # Saves them to disk, then returns them.
    # If the sets already exist on disk, this just loads and returns them.
    raw_datasets = get_raw_datasets(norb,
                                    args.azimuth_ratio,
                                    args.elevation_ratio,
                                    args.which_image,
                                    output_dir)

    if raw_datasets[0] is None and args.preprocessor == 'gcn_zca':
        raise ValueError("Training set was empty, but preprocessor was "
                         "gcn_zca. This should've been caught in parse_args.")

    # Preprocesses, saves datasets, unless they already exist.
    # Also saves preprocessor, if it's fittable.
    if args.preprocessor != 'none':
        preprocess_and_save_datasets(raw_datasets,
                                     args.preprocessor,
                                     output_dir)


    # # Subtracts each image's mean intensity. Scale of 55.0 taken from
    # # pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py
    # for dataset in datasets:
    #     if dataset is not None:
    #         global_contrast_normalize(dataset.X, scale=55.0, in_place=True)

    # # Prepares the output directory and checks against the existence of
    # # output files. We do this before ZCA'ing, to make sure we trigger any
    # # IOErrors now rather than later.
    # output_paths = get_output_paths(args)
    # (training_pkl_path,
    #  training_images_path,
    #  training_labels_path,
    #  testing_pkl_path,
    #  testing_images_path,
    #  testing_labels_path,
    #  pp_pkl_path,
    #  pp_npz_path) = output_paths

    # # Create / load a ZCA preprocessor if the training set is nonzero.
    # if datasets[0] is not None:

    #     # Load or create a ZCA preprocessor.
    #     if os.path.isfile(pp_pkl_path) and os.path.isfile(pp_npz_path):
    #         zca = serial.load(pp_pkl_path)
    #     elif datasets[0] is not None:
    #         zca = preprocessing.ZCA(use_memmap_workspace=True)
    #         zca_training_set = get_zca_training_set(datasets[0])

    #         print("Computing ZCA components using %d images out the %d "
    #               "training images" %
    #               (zca_training_set.shape[0], datasets[0].y.shape[0]))
    #         start_time = time.time()
    #         zca.fit(zca_training_set)
    #         print("\t...done (%g seconds)." % (time.time() - start_time))

    #         zca.set_matrices_save_path(pp_npz_path)
    #         serial.save(pp_pkl_path, zca)

    #         print("saved preprocessor to:")
    #         for pp_path in (pp_pkl_path, pp_npz_path):
    #             print("\t%s" % os.path.split(pp_path)[1])

    #     # ZCA the training set
    #     print("ZCA'ing training set...")
    #     start_time = time.time()
    #     datasets[0].apply_preprocessor(preprocessor=zca, can_fit=False)
    #     print("\t...done (%g seconds)." % (time.time() - start_time))

    #     # ZCA the testing set
    #     if datasets[1] is not None:
    #         print("ZCA'ing testing set...")
    #         start_time = time.time()
    #         datasets[1].apply_preprocessor(preprocessor=zca, can_fit=False)
    #         print("\t...done (%g seconds)." % (time.time() - start_time))

    # print("Saving to %s:" % os.path.split(training_pkl_path)[0])

    # for (dataset,
    #      pkl_path,
    #      images_path,
    #      labels_path) in safe_zip(datasets,
    #                               (training_pkl_path, testing_pkl_path),
    #                               (training_images_path, testing_images_path),
    #                               (training_labels_path, testing_labels_path)):
    #     if dataset is not None:
    #         serial.save(pkl_path, dataset)
    #         for path in (pkl_path, images_path, labels_path):
    #             print("\t%s" % os.path.split(path)[1])

if __name__ == '__main__':
    main()
