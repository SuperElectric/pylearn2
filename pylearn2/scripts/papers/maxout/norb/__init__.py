"""
Defines the pylearn2.scripts.papers.maxout.norb module.
"""

import os, pickle, sys
import numpy
from pylearn2.datasets.norb import SmallNORB


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

    return result[:, numpy.newaxis]  # size N vector -> Nx1 matrix


def get_instance_dataset(pickle_filepath):
    """
    Reads a (preprocessed) SmallNORB dataset from a .pkl file, and replaces its
    SmallNORB labels with object ID integers.
    """

    if os.path.splitext(pickle_filepath)[1] != '.pkl':
        raise ValueError("Expected a pickle file with suffix '.pkl', but got "
                         "filepath '%s'" % pickle_filepath)

    design_matrix = pickle.load(open(pickle_filepath, 'rb'))

    if design_matrix.y.shape[1] != len(SmallNORB.num_labels_by_type):
        raise ValueError("Expected SmallNORB labels to have length %d, but "
                         "got %d" % (len(SmallNORB.num_labels_by_type),
                                     design_matrix.y.shape[1]))

    design_matrix.y = SmallNORB_labels_to_object_ids(design_matrix.y)
    return design_matrix
