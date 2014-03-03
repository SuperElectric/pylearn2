#! /usr/bin/env python
"""
A script for viewing the confusion matrix of softmax labels computed with
./compute_softmax_instance_labels.py
"""

import argparse, numpy
import matplotlib
from matplotlib import pyplot
from pylearn2.utils import safe_zip


def main():

    def parse_args():
        labeler_name = './compute_softmax_instance_labels.py'
        parser = argparse.ArgumentParser(
            description=("Visualizes the confusion matrix of softmax labels "
                         "computed using the %s script." % labeler_name))

        parser.add_argument('--input',
                            '-i',
                            required=True,
                            help=("The .npz file computed by the %s script." %
                                  labeler_name))

        return parser.parse_args()

    args = parse_args()
    input_dict = numpy.load(args.input)
    softmax_labels, ground_truth = tuple(input_dict[key]
                                         for key in ('labels', 'ground_truth'))
    assert softmax_labels.shape[0] == ground_truth.shape[0]

    hard_labels = softmax_labels.argmax(axis=1)

    num_labels = softmax_labels.shape[0]
    num_instances = 50
    hard_confusion_matrix = numpy.zeros([num_instances, num_instances],
                                        dtype=float)
    soft_confusion_matrix = hard_confusion_matrix.copy()

    for ground_truth_label, hard_label, soft_label in safe_zip(ground_truth,
                                                               hard_labels,
                                                               softmax_labels):
        soft_confusion_matrix[ground_truth_label, :] += soft_label
        hard_confusion_matrix[ground_truth_label, hard_label] += 1.0

    # Normalize each row of the confusion matrix by dividing the row by the
    # number of times that row's object actually occurred in the dataset.
    for instance in range(num_instances):
        num_occurences = numpy.count_nonzero(ground_truth == instance)
        for confusion_matrix in (soft_confusion_matrix, hard_confusion_matrix):
            confusion_matrix[instance, :] /= float(num_occurances)

    figure, axes = pyplot.subplots(1, 2, squeeze=True)
    for (plot_index,
         plot_axes,
         confusion_matrix,
         title) in safe_zip(range(2),
                            axes,
                            (soft_confusion_matrix, hard_confusion_matrix),
                            ('softmax', 'binary')):

        plot_axes.imshow(confusion_matrix,
                         norm=matplotlib.colors.no_norm(),
                         interpolation='nearest')
        plot_axes.set_title(title)

    pyplot.show()


if __name__ == '__main__':
    main()
