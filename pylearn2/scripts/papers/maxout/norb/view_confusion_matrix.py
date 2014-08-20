#! /usr/bin/env python
"""
A script for viewing the confusion matrix of softmax labels computed with
./compute_softmax_instance_labels.py
"""

import sys, argparse, os.path
import numpy, matplotlib
from matplotlib import pyplot
from pylearn2.utils import safe_zip, serial
from pylearn2.scripts.papers.maxout.norb import \
    (load_norb_instance_dataset,
     norb_labels_to_object_ids,
     object_id_to_category_and_instance)
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.norb import SmallNORB
from pylearn2.datasets.zca_dataset import ZCA_Dataset


class DatasetPair(object):
    """
    Encapsulates the raw and preprocessed versions of a training or testing
    set. Provides methods for retrieving example images of a particular object,
    retrieving images with a given label, or retrieving images with the
    closest label to the given label.

    Parameters
    ----------
    dataset_path : str
      The dataset path. If this points to a raw dataset, then self.dataset and
      self.raw_dataset will point to the same object.
    """

    def __init__(self, dataset_path):
        assert os.path.splitext(dataset_path)[1] == '.pkl'

        def load_dataset(dataset_path):
            result = serial.load(dataset_path)
            assert len(result.y.shape) == 2
            assert result.y.shape[1] in (5, 11)
            return result

        self.dataset = load_dataset(dataset_path)

        def get_raw_dataset_path(dataset_path):
            """
            Returns the path to the raw dataset corresponding to the given
            dataset.

            If the given dataset is itself a raw dataset, this just returns
            dataset_path unchanged.
            """
            directory, filename = os.path.split(dataset_path)
            basename, extension = os.path.splitext(filename)
            assert extension == '.pkl'

            basename_parts = basename.split('_')
            assert len(basename_parts) == 2, 'filename: "%s"' % filename
            assert basename_parts[1] in ('test', 'train')

            raw_filename = 'raw_%s.pkl' % basename_parts[1]
            result = os.path.join(directory, raw_filename)

            if basename_parts[0] == 'raw':
                assert result == dataset_path

            return result

        raw_dataset_path = get_raw_dataset_path(dataset_path)

        if raw_dataset_path == dataset_path:
            self.raw_dataset = self.dataset
        else:
            self.raw_dataset = load_dataset(raw_dataset_path)
            assert (self.raw_dataset.y == self.dataset.y).all()

        def get_object_indices():
            """
            Returns a list x where x[i] is a tuple of row indices that
            point to instances of object i.
            """

            object_ids = norb_labels_to_object_ids(self.dataset.y,
                                                   self.dataset.label_name_to_index)
            print "len(object_ids): %d" % len(object_ids)

            unique_ids = frozenset(object_ids)
            result = [(), ] * (max(unique_ids) + 1)

            for unique_id in unique_ids:
                result[unique_id] = numpy.nonzero(object_ids == unique_id)[0]

            return result

        self.object_indices = get_object_indices()

        self.label_to_index = dict(safe_zip(self.dataset.y,
                                            range(self.dataset.y.shape[0])))

    def get_object_example(self, object_id, reference_label=None):
        """
        Returns an example of object_id. If reference_label is provided, try to
        find an example whose label is closest to the reference_label's
        azimuth, elevation, and lighting.
        """
        label = numpy.zeros(self.dataset.y.shape[1], dtype=int)

        if reference_label is not None:
            label[2:] = reference_label[2:]

        category, instance = object_id_to_category_and_instance(object_id)

        label[0] = category
        label[1] = instance
        return self.get_closest(label)

    def get_closest(self, label):
        label = numpy.array(label, dtype='int')
        object_id = norb_labels_to_object_ids(label,
                                              self.dataset.label_name_to_index)
        print "label: %s" % str(label)
        print "object_id: %d" % object_id
        print "len(object_indices): %d" % len(self.object_indices)
        print "self.dataset.y.shape: %s" % str(self.dataset.y.shape)
        # object_id = label[0] * self.instances_per_category + label[1]
        labels = self.dataset.y[self.object_indices[object_id], :]
        images = self.dataset.X[self.object_indices[object_id], :]
        raw_images = self.raw_dataset.X[self.object_indices[object_id], :]

        # Measures the angle difference (SSD of pitch & yaw) from <label>
        print ("labels.shape: %s, label.shape: %s" %
               (str(labels.shape), str(label.shape)))
        angle_differences = (labels[:, 2:4] - label[2:4])
        angle_differences *= numpy.array([5.0, 10.0])  # convert to degrees
        angle_differences = numpy.sum(angle_differences ** 2.0, axis=1)
        print "angle_differences.shape: %s" % str(angle_differences.shape)
        print ("numpy.min(angle_differences).shape: %s" %
               str(numpy.min(angle_differences).shape))
        # Discards all but the closest viewing angles.
        closest_indices = numpy.nonzero(angle_differences ==
                                        numpy.min(angle_differences))[0]
        labels = labels[closest_indices, :]
        images = images[closest_indices, :]
        raw_images = raw_images[closest_indices, :]

        # Look for a label with the same illumination
        row_indices = numpy.nonzero(labels[:, 4] == label[4])[0]
        if len(row_indices) == 0:
            row_indices = [0, ]
        else:
            assert len(row_indices == 1)

        label_row = labels[row_indices, :]
        image_row = images[row_indices, :]
        raw_image_row = raw_images[row_indices, :]

        def get_topo_image(dataset, image_row):
            result = dataset.get_topological_view(image_row)
            # removes batch and channel singleton axes
            return result[0, :, :, 0]

        return tuple(get_topo_image(d, i)
                     for d, i in safe_zip((self.dataset, self.raw_dataset),
                                          (image_row, raw_image_row)))

    def get_images_with_label(self, label):
        assert isinstance(label, (numpy.ndarray, numpy.memmap))

        row_index = self.label_to_index[label]
        assert numpy.all(label == self.labels[row_index, :])

        return tuple(dataset.X[row_index, :]
                     for dataset in (self.dataset, self.raw_dataset))


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

        parser.add_argument('--training_set',
                            '-t',
                            required=True,
                            help=("The .pkl file used as the training set."))

        return parser.parse_args()

    axes_to_heatmap = {}

    def get_objects_pointed_at(mouse_event):
        """
        If the mouse is pointing at a pixel in a plotted image, this returns
        a dictionary with the following keys:

          'row_obj': int. Object ID of that row.
          'col_obj': int. Object ID of that column.
          'value': float. The value of the pixel (0..1).
          'label': None, or 1D ndarray.
            If the image is one of the softmax plots, this returns the NORB
            label of the image corresponding to softmax vector (row) being
            pointed at.
            If the image is one of the confusion matrices, this is None.
        """

        if not mouse_event.inaxes in axes_to_heatmap.keys():
            # Mouse isn't pointing at a heatmap.
            return None
        else:
            heatmap = axes_to_heatmap[mouse_event.inaxes]['heatmap']
            row_ids = axes_to_heatmap[mouse_event.inaxes]['row_ids']
            col_ids = axes_to_heatmap[mouse_event.inaxes]['col_ids']
            labels = axes_to_heatmap[mouse_event.inaxes]['labels']

            # xdata, ydata actually start at -.5, -.5 in the upper-left corner
            row = int(mouse_event.ydata + .5)
            col = int(mouse_event.xdata + .5)
            row_obj = row_ids[row]

            if col_ids is None:
                col_obj = col
            else:
                col_obj = col_ids[row, col]

            label = None if labels is None else labels[row]

            return {'row_obj': row_obj,
                    'col_obj': col_obj,
                    'value': heatmap[row, col],
                    'label': label}

    def plot_heatmap(heatmap, axes, row_ids=None, col_ids=None, labels=None):
        axes.imshow(heatmap,
                    norm=matplotlib.colors.NoNorm(),
                    #norm=matplotlib.colors.no_norm(),
                    interpolation='nearest')

        if row_ids is None:
            row_ids = numpy.arange(heatmap.shape[0], dtype=int)

        axes_to_heatmap[axes] = {'heatmap': heatmap,
                                 'row_ids': row_ids,
                                 'col_ids': col_ids,
                                 'labels': labels}

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

            result[wrong_rowmask, :] = (differences ** 2).sum(axis=1)

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
        return most_confused_objects[nonzero_rowmask]

    args = parse_args()
    input_dict = numpy.load(args.input)
    softmax_labels, norb_labels = tuple(input_dict[key] for key
                                        in ('softmaxes', 'norb_labels'))
    assert softmax_labels.shape[0] == norb_labels.shape[0], \
        ("softmax_labels.shape: %s, norb_labels.shape: %s" %
         (str(softmax_labels.shape), str(norb_labels.shape)))

    dataset_path = str(input_dict['dataset_path'])
    data_dir = os.environ['PYLEARN2_DATA_PATH']
    if not dataset_path.startswith(data_dir):
        dataset_path = os.path.join(data_dir, dataset_path)

    testing_set = DatasetPair(dataset_path)

    # raw_dataset_path = get_raw_dataset_path(dataset_path)
    # raw_dataset = load_norb_instance_dataset(raw_dataset_path)

    # # performs a mapback just to induce that function to compile.
    # print "compiling un-ZCA'ing function (used for visualization)..."
    # dataset.mapback_for_viewer(dataset.X[:1, :])
    # print "...done"

    # label_to_index = {}
    # for index, label in enumerate(testing_set.y):
    #     key = tuple(label)
    #     # assert key not in label_to_index   #DEBUG: uncomment this
    #     label_to_index[key] = index

    # def get_image_with_label(norb_label):
    #     if len(norb_label) != testing_set.y.shape[1]:
    #         raise ValueError("len(norb_label) was %d, testing_set.y.shape[1] was "
    #                          "%d" % (len(norb_label), testing_set.y.shape[1]))

    #     row_index = label_to_index[tuple(norb_label)]
    #     row = testing_set.X[row_index, :]
    #     single_row_batch = row[numpy.newaxis, :]
    #     single_image_batch = testing_set.get_topological_view(single_row_batch)
    #     assert single_image_batch.shape[0] == 1
    #     assert single_image_batch.shape[-1] == 1
    #     return single_image_batch[0, :, :, 0]

    training_set = DatasetPair(args.training_set)

    # def SmallNORB_labels_to_object_ids(norb_labels):
    #     assert norb_labels.shape[1] == 5
    #     result = norb_labels[:, 0] * 10 + norb_labels[:, 1]
    #     return result

    # ground_truth = SmallNORB_labels_to_object_ids(norb_labels)
    ground_truth = norb_labels_to_object_ids(norb_labels,
                                             training_set.dataset.label_name_to_index)
    hard_labels = numpy.argmax(softmax_labels, axis=1)

    num_instances = 50 if training_set.dataset.y.shape[1] == 5 else 51
    hard_confusion_matrix = numpy.zeros([num_instances, num_instances],
                                        dtype=float)
    soft_confusion_matrix = hard_confusion_matrix.copy()

    for ground_truth_label, hard_label, soft_label in safe_zip(ground_truth,
                                                               hard_labels,
                                                               softmax_labels):
        soft_confusion_matrix[ground_truth_label, :] += soft_label
        hard_confusion_matrix[ground_truth_label, hard_label] += 1.0

    # Normalize each row of the confusion matrix by dividing the row by the
    # number of times that row's object actually occurred in the testing_set.
    for instance in range(num_instances):
        num_occurences = numpy.count_nonzero(ground_truth == instance)
        for confusion_matrix in (soft_confusion_matrix, hard_confusion_matrix):
            confusion_matrix[instance, :] /= float(num_occurences)

    figure, all_axes = pyplot.subplots(3, 4, figsize=(16, 12))
    default_status_text = "mouseover heatmaps for object ids and probabiliies"
    status_text = figure.text(.1, .05, default_status_text)

    def on_mouse_motion(event):
        original_text = status_text.get_text()

        picked = get_objects_pointed_at(event)
        if picked is None:
            status_text.set_text(default_status_text)
        else:
            text = ("row obj: %g, col obj: %g, val: %g" %
                    tuple(picked[key] for key in ('row_obj',
                                                  'col_obj',
                                                  'value')))
            label = picked['label']
            if label is not None:
                text = text + "\nImage label: %d %d %d %d %d" % tuple(label)

            status_text.set_text(text)

        if status_text.get_text() != original_text:
            figure.canvas.draw()

    def on_mousedown(event):
        pointed_at = get_objects_pointed_at(event)
        if pointed_at is None:
            return

        is_confusion_matrix = event.inaxes in all_axes[0, :]
        is_softmax_matrix = event.inaxes in all_axes[1, :]

        def plot_image(image, axes):
            # Neither ZCA_Dataset.mapback() nor .mapback_for_viewer() actually
            # map the image back to its original form, because the dataset's
            # preprocessor is unaware of the global contrast normalization we
            # did.
            #
            # Therefore we rely on matpotlib's pixel normalization that
            # happens by default.
            axes.imshow(image,
                        cmap='gray',
                        # norm=matplotlib.colors.no_norm(),
                        interpolation='nearest')

        if is_confusion_matrix or is_softmax_matrix:
            if is_confusion_matrix:
                correct_object = pointed_at['row_obj']
                correct_image_pair = \
                    training_set.get_object_example(correct_object)

                wrong_object = pointed_at['col_obj']
                wrong_image_pair = \
                    training_set.get_object_example(wrong_object)

                titles = ("correct example", "classifier example")
            else:  # i.e. is_softmax_matrix
                # correct image taken from the testing set.
                # It's the actual image we fed to the classifier.
                correct_image_pair = \
                    training_set.get_images_with_label(pointed_at['label'])

                wrong_object = pointed_at['col_obj']
                correct_label = pointed_at['label']
                wrong_image_pair = \
                    training_set.get_object_example(wrong_object,
                                                    correct_label)
                titles = ("actual image", "classifier example")

            # Show classified (i.e. possibly preprocessed) images
            for image, ax, title in safe_zip((correct_image_pair[0],
                                              wrong_image_pair[0]),
                                             all_axes[2, :2],
                                             titles):
                plot_image(image, ax)
                ax.set_title(title)

            # Show raw images
            for image, ax, title in safe_zip((correct_image_pair[1],
                                              wrong_image_pair[1]),
                                             all_axes[2, 2:],
                                             titles):
                plot_image(image, ax)
                ax.set_title(title)

            # for image_pair, ax, title in safe_zip((correct_image_pair,
            #                                        wrong_image_pair),
            #                                       all_axes[2, :2],
            #                                       titles):
            #     plot_image_pair(image_pair, ax)
            #     ax.set_title(title)

            # # Show corresponding un-ZCA'ed images. These are not the same as
            # # the original NORB images, becasue the ZCA preprocessor is unaware
            # # of and therefore can't undo the GCN preprocessing we did before
            # # ZCA.
            # for image, ax, title in safe_zip((correct_image, wrong_image),
            #                                  all_axes[2, 2:],
            #                                  titles):
            #     shape = image.shape
            #     flattened = image.reshape((1, numpy.prod(shape)))

            #     unpreprocessed = testing_set.mapback_for_viewer(flattened)
            #     plot_image(unpreprocessed.reshape(shape), ax)
            #     ax.set_title(title)

            figure.canvas.draw()

    # plots the confusion matrices
    for (plot_axes,
         confusion_matrix,
         title) in safe_zip(all_axes[0, :2],
                            (soft_confusion_matrix, hard_confusion_matrix),
                            ('softmax', 'binary')):

        plot_heatmap(confusion_matrix, plot_axes)
        plot_axes.set_title(title)

    difference_of_confusions = numpy.abs(hard_confusion_matrix -
                                         soft_confusion_matrix)
    print "max difference of confusions: %g" % difference_of_confusions.max()
    plot_heatmap(difference_of_confusions, all_axes[0, 2])
    all_axes[0, 2].set_title('difference')

    def plot_confusion_spread(confusion_matrix, most_confused_objects, axis):
        # confusion_matrix = confusion_matrix.copy()
        # numpy.fill_diagonal(confusion_matrix, 0.0)
        sorted_confusions = confusion_matrix[most_confused_objects, :]

        # sorts columns in ascending order
        sorted_column_indices = numpy.argsort(sorted_confusions, axis=1)
        sorted_column_indices = sorted_column_indices[:, ::-1]

        # re-doing sort out of laziness, rather than figuring out how to
        # properly plug in the above indices. (-_-;)
        sorted_confusions.sort(axis=1)  # sorts columns in ascending order
        sorted_confusions = sorted_confusions[:, ::-1]  # descending order

        assert (sorted_confusions >= 0.0).all()
        nonzero_columns = sorted_confusions.sum(axis=0) > 0.0

        sorted_confusions = sorted_confusions[:, nonzero_columns]
        sorted_column_indices = sorted_column_indices[:, nonzero_columns]

        plot_heatmap(sorted_confusions,
                     axis,
                     row_ids=most_confused_objects,
                     col_ids=sorted_column_indices)
        axis.set_title('Confusion spread')
        axis.set_xticklabels(())
        axis.set_yticklabels(())

    most_confused_objects = get_most_confused_objects(hard_confusion_matrix)
    plot_confusion_spread(hard_confusion_matrix,
                          most_confused_objects,
                          all_axes[0, -1])

    #
    # Plot the worst image-specific softmaxes of the most confused objects
    #

    # use of zip rather than safe_zip intentional here
    for axes, object_id in zip(all_axes[1, :], most_confused_objects):
        # rows of images showing object <object_id>
        row_mask = ground_truth == object_id
        assert len(row_mask.shape) == 1
        # assert row_mask.shape[1] == 1
        # row_mask = row_mask[:, 0]

        # Find the current object's NORB labels, object ids, and softmaxes
        print "row_mask.shape: %s, norb_labels.shape: %s" % (str(row_mask.shape),
                                                             str(norb_labels.shape))
        actual_norb_labels = norb_labels[row_mask, :]
        actual_ids = ground_truth[row_mask]
        assert (actual_ids == object_id).all()
        softmaxes = softmax_labels[row_mask, :]

        # Sort these rows by softmax's correctness (worst first).
        correctness = softmaxes[:, object_id]
        sorted_row_indices = numpy.argsort(correctness)

        # Only include the worst rows
        row_mask = correctness[sorted_row_indices] < 0.1
        sorted_row_indices = sorted_row_indices[row_mask]
        softmaxes = softmaxes[sorted_row_indices, :]
        actual_ids = actual_ids[sorted_row_indices]
        actual_norb_labels = actual_norb_labels[sorted_row_indices]

        plot_heatmap(softmaxes,
                     axes,
                     row_ids=actual_ids,
                     labels=actual_norb_labels)
        axes.set_title("Softmaxes of object %d" % object_id)
        axes.set_yticklabels(())
        axes.set_xticklabels(())

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('motion_notify_event', on_mouse_motion)
    figure.canvas.mpl_connect('button_press_event', on_mousedown)
    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()


if __name__ == '__main__':
    main()
