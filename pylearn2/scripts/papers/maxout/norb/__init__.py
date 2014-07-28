"""
Defines the pylearn2.scripts.papers.maxout.norb module.
"""

import os, pickle, sys
import numpy
from pylearn2.utils import serial
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def load_norb_instance_dataset(dataset_path,
                               # use_norb_labels=False,
                               convert_to_one_hot=False):
    """
    Loads a NORB (big or small) instance dataset and its preprocessor,
    returns both as a ZCA_Dataset (contains its own preprocessor).

    The dataset's y (labels) will be converted to object_id integers.

    returns: dataset
    dataset_path: string
      path to instance dataset's .pkl file
    """

    def load_instance_dataset(dataset_path):

        result = serial.load(dataset_path)
        object_ids = result.y[:, 0] * 10 + result.y[:, 1]
        unique_ids = frozenset(object_ids)
        assert len(unique_ids) == 50, ("Only found %d unique objects in "
                                       "dataset; expected all 50 objects."
                                       % len(unique_ids))
        result.y = object_ids[:, numpy.newaxis]


        # if not use_norb_labels:
        #     norb_labels = result.y
        #     assert norb_labels.shape[1] >= 5

        #     unique_instances = frozenset(norb_labels[:, 1])
        #     assert len(unique_instances) == 10, ("Expected 10 instance "
        #                                          "labels, found %d" %
        #                                          len(unique_instances))

        #     object_ids = result.y[:, 0] * 10 + result.y[:, 1]
        #     unique_ids = frozenset(object_ids)
        #     assert len(unique_ids) == 50, ("Only found %d unique objects in "
        #                                    "dataset; expected all 50 objects."
        #                                    % len(unique_ids))
        #     result.y = object_ids[:, numpy.newaxis]

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

    # def num_possible_objects():
    #     category_index = SmallNORB.label_type_to_index['category']
    #     instance_index = SmallNORB.label_type_to_index['instance']
    #     return (SmallNORB.num_labels_by_type[category_index] *
    #             SmallNORB.num_labels_by_type[instance_index])

    # object_ids = SmallNORB_labels_to_object_ids(dataset.y)

    # dataset = DenseDesignMatrix(X=dataset.X,
    #                             y=object_ids,
    #                             view_converter=dataset.view_converter,
    #                             max_labels=num_possible_objects())

    return ZCA_Dataset(preprocessed_dataset=dataset,
                       preprocessor=preprocessor,
                       convert_to_one_hot=convert_to_one_hot,
                       axes=['c', 0, 1, 'b'])


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
