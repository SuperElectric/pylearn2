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
      A Nx1 matrix of ints, where each row contains a single int, identifying
      the object identity. Ranges from 0 to 49 for small NORB, 0 to 50 for
      big NORB (50 is for "blank" images).
    """
    assert norb_labels.shape[1] in (5, 11)

    categories = norb_labels[:, label_name_to_index['category']]
    instances = norb_labels[:, label_name_to_index['instance']]

    # This gives wrong object_ids for 'blank'k images, which have negative
    # instance labels.
    object_ids = categories * 10 + instances

    # Correct the 'blank' images' object_ids to be 50.
    object_ids[categories == 5] = 50

    unique_ids = frozenset(object_ids)
    expected_num_ids = 50 if norb_labels.shape[1] == 5 else 51
    assert len(unique_ids) == expected_num_ids, \
        ("Only found %d unique objects in "
         "dataset; expected all %d objects."
         % (len(unique_ids), expected_num_ids))

    return object_ids[:, numpy.newaxis]


def load_norb_instance_dataset(dataset_path,
                               convert_to_one_hot=False,
                               return_zca_dataset=True,
                               crop_shape=None):
    """
    Loads a NORB (big or small) instance dataset and its preprocessor,
    returns both as a ZCA_Dataset (contains its own preprocessor).

    The dataset's y (labels) will be converted to object_id integers.

    Small NORB maps to 50 classes (1 per object).
    Big NORB maps to 51 classes (50 objects + "blank" images).

    returns: dataset

    dataset_path: string
      path to instance dataset's .pkl file

    convert_to_one_hot: bool
      If True, convert instance label to one-hot representation.



    """

    assert not (return_zca_dataset and (crop_shape is not None))

    def load_instance_dataset(dataset_path):

        result = serial.load(dataset_path)
        result.y = norb_labels_to_object_ids(result.y,
                                             result.label_name_to_index)

        # No need to update result.view_converter; it only deals with images,
        # not labels.

        return result

    dataset = load_instance_dataset(dataset_path)
    assert dataset.y.shape[1] == 1, ("Dataset labels must have size 1, "
                                     "but dataset.y.shape = %s" %
                                     str(dataset.y.shape))

    def get_preprocessor_path(dataset_path):
        base_path, extension = os.path.splitext(dataset_path)
        assert extension == '.pkl'
        assert any(base_path.endswith(x) for x in ('train', 'test'))

        if base_path.endswith('train'):
            base_path = base_path[:-5]
        elif base_path.endswith('test'):
            base_path = base_path[:-4]

        return base_path + 'preprocessor.pkl'

    preprocessor = serial.load(get_preprocessor_path(dataset_path))
    c01b_axes = ['c', 0, 1, 'b']

    if crop_shape is not None:
        assert not return_zca_dataset

        cropper = CentralWindow(crop_shape)
        print "cropping to %s" % str(crop_shape)
        crop_time = time.time()
        dataset.apply_preprocessor(cropper, can_fit=False)
        crop_time = time.time() - crop_time
        print "...finished cropping in %g secs" % crop_time
        dataset.set_view_converter_axes(c01b_axes)
        assert tuple(dataset.view_converter.shape[:2]) == tuple(crop_shape)
        if convert_to_one_hot:
            assert dataset.y.shape[1] == 1
            # Flatten y; otherwise convert_to_one_hot throws an exception. 
            # That may be a bug?
            dataset.y = dataset.y.flatten()
            dataset.convert_to_one_hot()

        return dataset

        # preprocessor = Pipeline(items=(preprocessor, cropper))
    else:
        assert return_zca_dataset
        return ZCA_Dataset(preprocessed_dataset=dataset,
                           preprocessor=preprocessor,
                           convert_to_one_hot=convert_to_one_hot,
                           axes=c01b_axes)


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
