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

    axes_to_heatmap = {}

    def plot_heatmap(heatmap, axes, row_ids=None):
        axes.imshow(heatmap,
                    norm=matplotlib.colors.no_norm(),
                    interpolation='nearest')

        if row_ids is None:
            row_ids = numpy.arange(heatmap.shape[0], dtype=int)

        axes_to_heatmap[axes] = {'heatmap': heatmap, 'row_ids': row_ids}

    def on_mouse_motion(event):
        if event.inaxes in axes_to_heatmap.keys():
            heatmap = axes_to_heatmap[event.inaxes]['heatmap']
            row_ids = axes_to_heatmap[event.inaxes]['row_ids']

            # xdata, ydata actually start at -.5, -.5 in the upper-left corner
            row = int(event.ydata + .5)
            col = int(event.xdata + .5)
            row_obj = row_ids[int(event.ydata)]
            print "row obj: %g, col obj: %g, val: %g" % (row_obj,
                                                         col,
                                                         heatmap[row, col])

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
            assert isinstance(one_or_more_axes, (tuple,
                                                 list,
                                                 numpy.ndarray)), \
                              "type: %s" % type(one_or_more_axes)

        wrongnesses = wrongness(softmax_labels, ground_truth)

        # Row indices, in descending order of wrongness
        sorted_row_indices = numpy.argsort(wrongnesses)[::-1]
        softmax_labels = softmax_labels[sorted_row_indices, :]
        ground_truth = ground_truth[sorted_row_indices]

        # num_wrong = numpy.count_nonzero(wrongnesses)
        num_wrong = 500
        plot_heatmap(softmax_labels[:num_wrong, :], one_or_more_axes[0])
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

    def get_most_confused_objects(confusion_matrix):
        """
        Given a confusion matrix, returns a vector containing the object IDs of
        the most frequently misidentified objects.
        """

        confusion_matrix = confusion_matrix.copy()
        numpy.fill_diagonal(confusion_matrix, 0.0)
        misclassification_rates = confusion_matrix.sum(axis=1)
        most_confused_objects = numpy.argsort(misclassification_rates)[::-1]
        sorted_misclassification_rates = \
            misclassification_rates[most_confused_objects]

        nonzero_rowmask = sorted_misclassification_rates > 0.0
        # print "shapes: ", most_confused_objects.shape, nonzero_rowmask.shape
        return most_confused_objects[nonzero_rowmask]


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

    figure, all_axes = pyplot.subplots(2, 4, squeeze=False)

    # plots the confusion matrices
    for (plot_index,
         plot_axes,
         confusion_matrix,
         title) in safe_zip(range(2),
                            all_axes[0, :2],
                            (soft_confusion_matrix, hard_confusion_matrix),
                            ('softmax', 'binary')):

        plot_heatmap(confusion_matrix, plot_axes)
        plot_axes.set_title(title)

    plot_heatmap(numpy.abs(hard_confusion_matrix - soft_confusion_matrix),
                 all_axes[0, 2])
    all_axes[0, 2].set_title('difference')

    def plot_confusion_spread(confusion_matrix, most_confused_objects, axis):
        # confusion_matrix = confusion_matrix.copy()
        # numpy.fill_diagonal(confusion_matrix, 0.0)
        sorted_confusions = confusion_matrix[most_confused_objects, :]
        sorted_confusions.sort(axis=1)  # sorts columns in ascending order
        sorted_confusions = sorted_confusions[:, ::-1]  # descending order

        assert (sorted_confusions >= 0.0).all()
        nonzero_columns = sorted_confusions.sum(axis=0) > 0.0

        plot_heatmap(sorted_confusions[:, nonzero_columns],
                     axis,
                     row_ids=most_confused_objects)
        axis.set_title('Confusion spread')
        axis.set_xticklabels(())
        axis.set_yticklabels(())


    most_confused_objects = get_most_confused_objects(hard_confusion_matrix)
    plot_confusion_spread(hard_confusion_matrix,
                          most_confused_objects,
                          all_axes[0, -1])

    # use of zip rather than safe_zip intentional here
    for (axes, object_id) in zip(all_axes[1, :], most_confused_objects):
        row_mask = ground_truth == object_id
        actual_ids = ground_truth[row_mask]
        softmaxes = softmax_labels[row_mask, :]
        correctness_probability = softmaxes[:, object_id]
        sorted_row_indices = numpy.argsort(correctness_probability)

        # Only include the worst rows
        row_mask = correctness_probability[sorted_row_indices] < 0.1
        sorted_row_indices = sorted_row_indices[row_mask]
        softmaxes = softmaxes[sorted_row_indices, :]
        actual_ids = actual_ids[sorted_row_indices]

        plot_heatmap(softmaxes, axes, row_ids=actual_ids)
        axes.set_title("Softmaxes of\nobject %d" % object_id)
        axes.set_yticklabels(())
        axes.set_xticklabels(())


    # axes[-1].imshow(confusion_matrix[most_confused_objects, :],
    #                 interpolation='nearest')
    # plots the worst softmax(es)
    # axes[-1].imshow(softmax_labels[:100, :], interpolation='nearest')
    # plot_worst_softmax(softmax_labels, ground_truth, axes[2:])

    figure.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

    pyplot.show()


if __name__ == '__main__':
    main()
