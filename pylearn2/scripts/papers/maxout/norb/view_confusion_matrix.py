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

    def plot_worst_softmax(softmax_labels, ground_truth, one_or_more_axes):
        def wrongness(softmax_labels, ground_truth):
            """
            Returns a scalar measure of wrongness of a softmax label.
            Wrongness is 0 if argmax(softmax) == ground_truth.
            If argmax(softmax) != ground_truth, returns the squared distance
            between softmax and onehot(ground_truth).

            One drawback to this metric is that it doesn't penalize correct
            softmaxes that are very close to wrong, i.e. softmaxes whose
            second-largest element is very close to the largest element.

            Maybe we can also plot a different wrongness that is a pure
            distance metric from the onehot ground truth, and doesn't care if
            the argmax(softmax) is correct.
            """

            def get_onehot(int_label, num_labels):
                assert len(int_label.shape) == 1
                result = numpy.zeros((int_label.shape[0], num_labels),
                                     dtype=float)
                result[:, int_label] = 1.0
                return result

            result = numpy.zeros(ground_truth.shape[0], dtype=float)

            wrong_rowmask = (numpy.argmax(softmax_labels, axis=1) !=
                             ground_truth)

            softmax_labels = softmax_labels[wrong_rowmask, :]
            ground_truth = ground_truth[wrong_rowmask]
            differences = softmax_labels - get_onehot(ground_truth,
                                                      softmax_labels.shape[1])

            result[wrong_rowmask, :] = (differences**2).sum(axis=1)

            return result

        if isinstance(one_or_more_axes, matplotlib.axes.Axes):
            one_or_more_axes = (one_or_more_axes, )
        else:
            assert isinstance(one_or_more_axes, (tuple, list, numpy.ndarray)), "type: %s" % type(one_or_more_axes)

        wrongnesses = wrongness(softmax_labels, ground_truth)


        # Row indices, in descending order of wrongness
        sorted_row_indices = numpy.argsort(wrongnesses)[::-1]
        softmax_labels = softmax_labels[sorted_row_indices, :]
        ground_truth = ground_truth[sorted_row_indices]

        # num_wrong = numpy.count_nonzero(wrongnesses)
        num_wrong = 500 
        one_or_more_axes[0].imshow(softmax_labels[:num_wrong, :],
                                   interpolation='nearest')
        return

        num_bars = 10  # plot only the top <num_bars> softmax scores

        # use zip instead of safe_zip to iterate only over the first N
        # softmaxes, where N = # of axes
        for softmax, ground_truth, axes in zip(softmax_labels,
                                               ground_truth,
                                               one_or_more_axes):
            # column indices (object IDs) in descending order of softmax score
            sorted_ids = numpy.argsort(softmax)[::-1][:num_bars]
            softmax = softmax[sorted_ids]
            print "sorted_ids.shape: ", sorted_ids.shape
            print "softmax.shape: ", softmax.shape
            axes.bar(left=numpy.arange(num_bars),
                     height=softmax)
            axes.set_xticklabels(tuple(str(i) for i in sorted_ids))
            axes.set_ylabel('softmax')
            axes.set_xlabel('object ID')
            axes.set_title("object with id %d" % ground_truth)



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
            confusion_matrix[instance, :] /= float(num_occurences)

    figure, axes = pyplot.subplots(1, 3, squeeze=True)

    # plots the confusion matrices
    for (plot_index,
         plot_axes,
         confusion_matrix,
         title) in safe_zip(range(2),
                            axes[:2],
                            (soft_confusion_matrix, hard_confusion_matrix),
                            ('softmax', 'binary')):

        plot_axes.imshow(confusion_matrix,
                         norm=matplotlib.colors.no_norm(),
                         interpolation='nearest')
        plot_axes.set_title(title)

    # plots the worst softmax(es)
    # axes[-1].imshow(softmax_labels[:100, :], interpolation='nearest')
    plot_worst_softmax(softmax_labels, ground_truth, axes[2:])

    pyplot.show()


if __name__ == '__main__':
    main()
