#! /usr/bin/env python

import argparse, sys
import numpy
from matplotlib import pyplot
from pylearn2.datasets.norb import SmallNORB


def main():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Samples the view hemisphere of an object, dumping "
                        "some images into a given directory.")
        parser.add_argument('--category',
                            '-c',
                            type=int,
                            help="The class label of the object")
        parser.add_argument('--instance',
                            '-i',
                            type=int,
                            help="The instance label of the object")
        parser.add_argument('--azimuth_downsample',
                            '-a',
                            type=int,
                            default=2,
                            metavar='M',
                            help="Output every M'th azimuth.")
        parser.add_argument('--elevation_downsample',
                            '-e',
                            type=int,
                            default=3,
                            metavar='N',
                            help="Output every N'th elevation.")
        parser.add_argument('--lighting',
                            type=int,
                            default=0,
                            help="The lighting type to use.")

        result = parser.parse_args()
        if result.category < 0 or result.category > 4:
            print ("--instance must be in the range [0..4], not %d"
                   % result.category)
            sys.exit(1)

        if result.instance < 0 or result.instance > 9:
            print ("--instance must be in the range [0..9], not %d"
                   % result.instance)
            sys.exit(1)

        return parser.parse_args()

    def get_data(dataset):
        # Gets the left images
        values = dataset.get_topological_view(single_tensor=False)[0]
        labels = dataset.y
        return values, numpy.array(labels, 'int')

    instance_index = SmallNORB.label_type_to_index['instance']

    def remap_instances(which_set, labels):
        if which_set == 'train':
            new_to_old_instance = [4, 6, 7, 8, 9]
        elif which_set == 'test':
            new_to_old_instance = [0, 1, 2, 3, 5]

        num_instances = len(new_to_old_instance)
        old_to_new_instance = numpy.ndarray(10, 'int')
        old_to_new_instance.fill(-1)
        old_to_new_instance[new_to_old_instance] = numpy.arange(num_instances)

        instance_slice = numpy.index_exp[:, instance_index]
        old_instances = labels[instance_slice]

        new_instances = old_to_new_instance[old_instances]
        labels[instance_slice] = new_instances

        azimuth_index = SmallNORB.label_type_to_index['azimuth']
        azimuth_slice = numpy.index_exp[:, azimuth_index]
        labels[azimuth_slice] = labels[azimuth_slice] / 2

        return new_to_old_instance, old_to_new_instance

    def get_label_to_index_map(num_instances):

        # Maps a label vector to the corresponding index in <values>
        num_labels_by_type = numpy.array(SmallNORB.num_labels_by_type,
                                         'int')
        num_labels_by_type[instance_index] = num_instances

        label_to_index = numpy.ndarray(num_labels_by_type, 'int')
        label_to_index.fill(-1)

        for i, label in enumerate(labels):
            label_to_index[tuple(label)] = i

        # all elements have been set
        assert not numpy.any(label_to_index == -1)

        return label_to_index

    args = parse_args()
    which_set = ('test' if args.instance in (0, 1, 2, 3, 5)
                 else 'train')
    norb = SmallNORB(which_set, True)
    left_images, labels = get_data(norb)
    new_to_old_instance, old_to_new_instance = remap_instances(which_set,
                                                               labels)
    label_to_index = get_label_to_index_map(len(new_to_old_instance))

    new_instance = old_to_new_instance[args.instance]
    assert new_instance >= 0
    label_subset = label_to_index[args.category, new_instance, :, :, args.lighting]

    elevations = range(0, label_subset.shape[0], args.elevation_downsample)
    azimuths = range(0, label_subset.shape[1], args.azimuth_downsample)
    # matplotlib.figure(figsize=(8,6))
    fig_size_inches = (len(azimuths) * 2,
                       len(elevations) * 2)
    figure, axes = pyplot.subplots(len(elevations),
                                   len(azimuths),
                                   squeeze=False,
                                   figsize=fig_size_inches)

    for ie, elevation in enumerate(elevations):
        for ia, azimuth in enumerate(azimuths):
            left_image = left_images[label_subset[elevation, azimuth], :, :, 0]
            print "image.shape: ", left_image.shape
            axis = axes[ie][ia]
            axis.imshow(left_image, cmap='gray')

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()


if __name__ == '__main__':
    main()
