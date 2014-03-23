"""
Defines the pylearn2.scripts.papers.maxout.norb module.
"""

import os, pickle, sys
import numpy
from pylearn2.utils import serial
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


def load_small_norb_instance_dataset(dataset_path,
                                     convert_to_one_hot=False,
                                     use_norb_labels=True):
    """
    Loads a NORB instance dataset and its preprocessor.

    returns: dataset, original_labels
    dataset_path: ZCA_Dataset
        The labels are the original NORB label vectors.
        Use SmallNORB_Labels_to_object_ids() to convert to object IDs.
    """

    def get_preprocessor_path(dataset_path):
        base_path, extension = os.path.splitext(dataset_path)
        assert extension == '.pkl'
        assert any(base_path.endswith(x) for x in ('train', 'test'))

        if base_path.endswith('train'):
            base_path = base_path[:-5]
        elif base_path.endswith('test'):
            base_path = base_path[:-4]

        return base_path + 'preprocessor.pkl'

    dataset = serial.load(dataset_path)
    preprocessor = serial.load(get_preprocessor_path(dataset_path))

    if not use_norb_labels:

        def num_possible_objects():
            category_index = SmallNORB.label_type_to_index['category']
            instance_index = SmallNORB.label_type_to_index['instance']
            return (SmallNORB.num_labels_by_type[category_index] *
                    SmallNORB.num_labels_by_type[instance_index])

        object_ids = SmallNORB_labels_to_object_ids(dataset.y)
        dataset = DenseDesignMatrix(X=dataset.X,
                                    y=object_ids,
                                    view_converter=dataset.view_converter,
                                    max_labels=num_possible_objects())

    return ZCA_Dataset(preprocessed_dataset=dataset,
                       preprocessor=preprocessor,
                       convert_to_one_hot=convert_to_one_hot,
                       axes=['c', 0, 1, 'b'])


def object_id_to_SmallNORB_label_pair(object_ids):
    """
    Given an Nx1 matrix of object IDs, this returns a Nx2 matrix of
    corresponding class and instance SmallNORB labels.
    """
    assert len(object_ids.shape) == 2
    assert object_ids.shape[1] == 1

    instance_index = SmallNORB.label_type_to_index['instance']
    instances_per_class = SmallNORB.num_labels_by_type[instance_index]

    result = numpy.zeros((object_ids.shape[0], 2), int)
    result[:, 0] = int(object_ids / instances_per_class)
    result[:, 1] = numpy.mod(object_ids, instances_per_class)
    return result


def SmallNORB_labels_to_object_ids(label_vectors):
    """
    Given a NxM matrix of SmallNORB labels, returns a Nx1 matrix of unique
    IDs for each object.
    """

    def contains_equal_numbers_of_all_objects(object_ids, num_objects):
        """
        Returns True iff object_ids contains all numbers from 0 to
        num_objects-1
        """
        assert len(object_ids.shape) == 1
        assert object_ids.shape[0] > 0

        object_counts = numpy.array([numpy.count_nonzero(object_ids == i)
                                     for i in xrange(num_objects)], int)

        return numpy.all(object_counts[1:] == object_counts[0])

    category_index, instance_index = (SmallNORB.label_type_to_index[n]
                                      for n in ('category', 'instance'))

    num_categories, num_instances = (SmallNORB.num_labels_by_type[i]
                                     for i in (category_index,
                                               instance_index))

    categories, instances = (label_vectors[:, i]
                             for i in (category_index, instance_index))
    result = categories * num_instances + instances

    # We expect all object IDs to be represented, and represented equally.
    #
    # This need not necessarily be true, but we expect it to be true under
    # current usage, so we include this as a sanity check.
    num_objects = num_categories * num_instances

    if not contains_equal_numbers_of_all_objects(result, num_objects):
        for object_id in xrange(num_objects):
            print "contains %d of object %d" % \
                  (numpy.count_nonzero(result==object_id), object_id)
        sys.exit(1)

    return result
    #return result[:, numpy.newaxis]  # size N vector -> Nx1 matrix


# def get_instance_dataset(pickle_filepath):
#     """
#     Reads a (preprocessed) SmallNORB dataset from a .pkl file, and replaces its
#     SmallNORB labels with object ID integers.
#     """

#     if os.path.splitext(pickle_filepath)[1] != '.pkl':
#         raise ValueError("Expected a pickle file with suffix '.pkl', but got "
#                          "filepath '%s'" % pickle_filepath)

#     design_matrix = pickle.load(open(pickle_filepath, 'rb'))

#     if design_matrix.y.shape[1] != len(SmallNORB.num_labels_by_type):
#         raise ValueError("Expected SmallNORB labels to have length %d, but "
#                          "got %d" % (len(SmallNORB.num_labels_by_type),
#                                      design_matrix.y.shape[1]))

#     design_matrix.y = SmallNORB_labels_to_object_ids(design_matrix.y)
#     return design_matrix


