"""
Defines the pylearn2.scripts.papers.maxout.norb module.
"""

import os, pickle, sys, time
import numpy
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


def load_norb_instance_dataset(dataset_path,
                               label_format="obj_id",
                               crop_shape=None):
    """Loads a NORB (big or small) instance dataset.

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

    if crop_shape is not None:
        # assert not return_zca_dataset

        cropper = CentralWindow(crop_shape)
        print "cropping to %s" % str(crop_shape)
        crop_time = time.time()
        dataset.apply_preprocessor(cropper, can_fit=False)
        crop_time = time.time() - crop_time
        print "...finished cropping in %g secs" % crop_time
        # dataset.set_view_converter_axes(c01b_axes)
        assert tuple(dataset.view_converter.shape[:2]) == tuple(crop_shape)
        # if label_format == "obj_onehot":
        #     assert len(dataset.y.shape) == 1
        #     dataset.convert_to_one_hot()

        return dataset

        # preprocessor = Pipeline(items=(preprocessor, cropper))
    # else:
    #     assert return_zca_dataset
    #     return ZCA_Dataset(preprocessed_dataset=dataset,
    #                        preprocessor=preprocessor,
    #                        convert_to_one_hot=False)
    #     # return ZCA_Dataset(preprocessed_dataset=dataset,
    #     #                    preprocessor=preprocessor,
    #     #                    convert_to_one_hot=convert_to_one_hot)
    #     # return ZCA_Dataset(preprocessed_dataset=dataset,
    #     #                    preprocessor=preprocessor,
    #     #                    convert_to_one_hot=convert_to_one_hot,
    #     #                    axes=c01b_axes)


# def object_id_to_SmallNORB_label_pair(object_ids):
#     """
#     Given an Nx1 matrix of object IDs, this returns a Nx2 matrix of
#     corresponding class and instance SmallNORB labels.
#     """
#     assert len(object_ids.shape) == 2
#     assert object_ids.shape[1] == 1

#     instance_index = SmallNORB.label_type_to_index['instance']
#     instances_per_class = SmallNORB.num_labels_by_type[instance_index]

#     result = numpy.zeros((object_ids.shape[0], 2), int)
#     result[:, 0] = int(object_ids / instances_per_class)
#     result[:, 1] = numpy.mod(object_ids, instances_per_class)
#     return result


# def SmallNORB_labels_to_object_ids(label_vectors):
#     """
#     Given a NxM matrix of SmallNORB labels, returns a Nx1 matrix of unique
#     IDs for each object.
#     """

#     def contains_equal_numbers_of_all_objects(object_ids, num_objects):
#         """
#         Returns True iff object_ids contains all numbers from 0 to
#         num_objects-1
#         """
#         assert len(object_ids.shape) == 1
#         assert object_ids.shape[0] > 0

#         object_counts = numpy.array([numpy.count_nonzero(object_ids == i)
#                                      for i in xrange(num_objects)], int)

#         return numpy.all(object_counts[1:] == object_counts[0])

#     category_index, instance_index = (SmallNORB.label_type_to_index[n]
#                                       for n in ('category', 'instance'))

#     num_categories, num_instances = (SmallNORB.num_labels_by_type[i]
#                                      for i in (category_index,
#                                                instance_index))

#     categories, instances = (label_vectors[:, i]
#                              for i in (category_index, instance_index))
#     result = categories * num_instances + instances

#     return result
