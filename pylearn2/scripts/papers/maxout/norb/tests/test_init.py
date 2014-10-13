"""Tests for ./__init__.py"""

import numpy
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import safe_zip


def test_evenly_sampling_iterator():

    # non-contiguous, negagtive and postive label values
    unique_label_values = numpy.arange(10) * 2 - 5

    label_distribution = numpy.linspace(.1, 1, len(unique_label_values))
    label_distribution /= sum(label_distribution)

    even_count_per_label = 1000
    num_examples = len(unique_label_values) * even_count_per_label

    rng = numpy.random.RandomState(2212)
    labels = rng.choice(unique_label_values,
                        size=num_examples,
                        replace=True,
                        p=label_distribution)

    old_label_counts = [numpy.count_nonzero(labels==u)
                        for u in unique_label_values]
    print "old_label_counts: ", old_label_counts

    # values is just a 2D wrapper around labels
    values = labels[:, numpy.newaxis]

    dataset = DenseDesignMatrix(X=values, y=labels)
    batch_size = 10
    iterator = dataset.iterator(mode='evenly_sampling', batch_size=batch_size)

    label_to_count = dict(safe_zip(unique_label_values,
                                   numpy.zeros((len(unique_label_values)))))

    for value_batch in iterator:  # values are equal to labels
        assert value_batch.shape == (batch_size, dataset.X.shape[1])
        for value in value_batch:
            label_to_count[value[0]] += 1

    label_counts = numpy.asarray(label_to_count.values())
    print "new label counts: ", label_counts
    assert label_counts.sum() == num_examples, \
        ("label_counts.sum(): %d, num_examples: %d" %
         (label_counts.sum(), num_examples))

    label_counts_range = label_counts.max() - label_counts.min()

    threshold = 0.15 * even_count_per_label
    assert label_counts_range < threshold, label_counts_range
