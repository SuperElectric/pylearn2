"""
Defines the pylearn2.scripts.papers.maxout.norb module.
"""

import os, pickle, sys, time, copy
import numpy
from pylearn2.blocks import Block
from pylearn2.utils import serial
from pylearn2.datasets.preprocessing import CentralWindow, Pipeline
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


def human_readable_memory_size(size, precision=2):
    if size < 0:
        raise ValueError("Size must be non-negative (was %g)." % size)

    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffix_index = 0
    while size > 1024:
        suffix_index += 1  # increment the index of the suffix
        size = size / 1024.0  # apply the division
    return "%.*f %s" % (precision, size, suffixes[suffix_index])


def human_readable_time_duration(seconds):
    hours = seconds // 60
    seconds = seconds % 60
    days = hours // 24
    hours = hours % 24

    result = "%g s"
    if hours > 0:
        result = "%d h " + result
    if days > 0:
        result = "%d d " + result

    return result


def object_id_to_category_and_instance(object_ids):
    if numpy.isscalar(object_ids):
        object_ids = numpy.array(object_ids)

    assert len(object_ids.shape) in (0, 1)

    categories = object_ids // 10
    instances = object_ids % 10  # TODO: is instance for blank really 0?

    return categories, instances


def norb_labels_to_object_ids(norb_labels, label_name_to_index):
    """
    Converts norb labels to unique object IDs.

    Parameters:
    -----------

    norb_labels : numpy.ndarray
      See pylearn2.datasets.NORB.y.
      A NxM matrix of ints, where each row is a NORB label vector.
      M == 5 for small norb, M == 11 for big NORB.

    label_name_to_index : dict
      See pylearn2.datasets.NORB.label_name_to_index.
      Maps label name strings to their column indices in norb_labels.

    returns : numpy.ndarray
      A vector of ints, where each row contains a single int, identifying
      the object identity. Ranges from 0 to 49 for small NORB, 0 to 50 for
      big NORB (50 is for "blank" images).
    """
    if len(norb_labels.shape) == 1:
        norb_labels = norb_labels[numpy.newaxis, :]

    assert norb_labels.shape[1] in (5, 11)

    categories = norb_labels[:, label_name_to_index['category']]
    instances = norb_labels[:, label_name_to_index['instance']]

    # This gives wrong object_ids for 'blank'k images, which have negative
    # instance labels.
    object_ids = categories * 10 + instances

    # Correct the 'blank' images' object_ids to be 50.
    object_ids[categories == 5] = 50

    # unique_ids = frozenset(object_ids)
    # expected_num_ids = 50 if norb_labels.shape[1] == 5 else 51
    # assert len(unique_ids) == expected_num_ids, \
    #     ("Only found %d unique objects in "
    #      "dataset; expected all %d objects."
    #      % (len(unique_ids), expected_num_ids))

    return object_ids
    # return object_ids[:, numpy.newaxis]


class CropBlock(Block):

    def __init__(self, dataset, crop_shape):
        assert len(crop_shape) == 2
        assert isinstance(dataset, DenseDesignMatrix)

        self._dataset_view_converter = dataset.view_converter
        self._cropped_view_converter = copy.copy(self._dataset_view_converter)
        self._cropped_view_converter.shape = \
            (tuple(crop_shape) + (self._dataset_view_converter[-1], ))
        # self._dataset = dataset
        self._crop_shape = crop_shape

    def __call__(self, batch):
        """
        Crops a batch.

        Parameters:
        -----------

        batch: theano matrix
          A symbolic matrix, representing a batch of rows from a dense
          design matrix.

        returs: theano matrix
          The matrix_batch, cropped.
        """

        # view_converter = self._dataset.view_converter
        topo_batch = self._dataset_view_converter.design_mat_to_topo_view(batch)

        axes = self._dataset_view_converter.axes
        needs_transpose = not axes[1:3] == (0, 1)

        if needs_transpose:
            axes_order = tuple(axes.index(a) for a in ('c', 0, 1, 'b'))
            batch = batch.transpose(axes_order)

        offset = (arr.shape[1:3] - self._crop_shape) // 2
        batch = batch[:,
                      offset[0]:offset[0] + self._crop_shape[0],
                      offset[1]:offset[1] + self._crop_shape[1],
                      :]

        if needs_transpose:
            reverse_axes_order = tuple(('c', 0, 1, 'b').index(a)
                                       for a in self._axis)
            batch = batch.transpose(reverse_axes_order)

        return cropped_view_converter.topo_view_to_design_mat(batch)


def load_norb_instance_dataset(dataset_path,
                               label_format="obj_id",
                               crop_shape=None,
                               axes=None):
    """
    Loads a NORB (big or small) instance dataset.

    Optionally replaces the dataset's y (NORB label vectors) to object ID
    integers, or to object ID one-hot vectors. Small NORB maps to 50 classes (1
    per object). Big NORB maps to 51 classes (50 objects + "blank" images).

    Optionally crops the NORB images to a central window.

    returns: dataset

    Parameters:
    -----------

    dataset_path : string
      path to instance dataset's .pkl file

    label_format : str
      choices: "norb", "obj_id", "obj_onehot"
      "norb": leave the label vectors untouched, as NORB label vectors
      "obj_id": convert the label vectors to object ID scalars.
      "obj_onehot": convert the label vectors to object ID one-hot vectors.

    crop_shape : tuple
      A tuple of two ints. Specifies the shape of a central window to crop
      the images to.
    """
    if axes is not None:
        assert tuple(axes) in (('c', 0, 1, 'b'),
                               ('b', 0, 1, 'c'))

    def load_instance_dataset(dataset_path):

        result = serial.load(dataset_path)
        assert len(result.y.shape) == 2
        assert result.y.shape[1] in (5, 11), ("Expected 5 or 11 columns, "
                                              "got %d" % result.y.shape[1])

        if label_format in ('obj_id', 'obj_onehot'):
            result.y = norb_labels_to_object_ids(result.y,
                                                 result.label_name_to_index)
            if label_format == 'obj_onehot':
                result.convert_to_one_hot()

        # No need to update result.view_converter; it only deals with images,
        # not labels.

        return result

    dataset = load_instance_dataset(dataset_path)

    # def get_preprocessor_path(dataset_path):
    #     directory, filename = os.path.split(dataset_path)
    #     basename, extension = os.path.splitext(filename)
    #     assert extension == '.pkl'

    #     basename_parts = '_'.split(basename)
    #     assert(len(basename_parts) == 2)
    #     assert basename_parts[1] in ('train', 'test')
    #     preprocessor_name = basename_parts[0] + "_preprocessor.pkl"
    #     return os.path.join(directory, preprocessor_name)

    #     # assert any(basename.endswith(x) for x in ('train', 'test'))

    #     # if base_path.endswith('train'):
    #     #     base_path = base_path[:-5]
    #     # elif base_path.endswith('test'):
    #     #     base_path = base_path[:-4]

    #     # return base_path + 'preprocessor.pkl'

    # preprocessor = serial.load(get_preprocessor_path(dataset_path))
    # # c01b_axes = ['c', 0, 1, 'b']

    if axes is not None:
        dataset.set_view_converter_axes(axes)

    if crop_shape is not None:
        crop_block = CropBlock(dataset, crop_shape)
        # topo_shape = tuple(crop_shape) + (dataset.view_converter.shape[-1], )
        # dataset.view_conveter = DefaultViewConverter(shape=topo_shape,
        #                                              axes=axes)
        dataset = TransformerDataset(raw=dataset,
                                     transformer=crop_block,
                                     space_preserving=False)

    # if crop_shape is not None:
    #     # assert not return_zca_dataset

    #     cropper = CentralWindow(crop_shape)
    #     print "cropping to %s" % str(crop_shape)
    #     crop_time = time.time()
    #     dataset.apply_preprocessor(cropper, can_fit=False)
    #     crop_time = time.time() - crop_time
    #     print "...finished cropping in %g secs" % crop_time
    #     # dataset.set_view_converter_axes(c01b_axes)
    #     assert tuple(dataset.view_converter.shape[:2]) == tuple(crop_shape)
    #     # if label_format == "obj_onehot":
    #     #     assert len(dataset.y.shape) == 1
    #     #     dataset.convert_to_one_hot()

    return dataset


# class PreprocessingViewConverter(dense_design_matrix.DefaultViewConverter):

#     def __init__(self, preprocessor, shape, axes=('b', 0, 1, 'c')):
#         self._super = super(PreprocessingViewConverter, self)
#         self._preprocessor = preprocessor
#         self._dummy_design_matrix = None

#     def get_formatted_batch(self, batch, dspace):
#         batch = self._super.get_formatted_batch(batch, dspace)
#         if self.dummy_design_matrix is None:
#             self._dummy_design_matrix = DenseDesignMatrix(X=batch)
