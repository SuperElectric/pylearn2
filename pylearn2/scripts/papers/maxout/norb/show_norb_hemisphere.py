#! /usr/bin/env python

import argparse, sys
import numpy
from matplotlib import pyplot
from pylearn2.datasets.new_norb import NORB
from pylearn2.utils import safe_zip


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

    args = parse_args()
    norb = NORB(which_norb="small", which_set="both")

    def check_label_ordering(norb):
        column_order = tuple(norb.label_name_to_index[label_name]
                             for label_name in ('category',
                                                'instance',
                                                'elevation',
                                                'azimuth',
                                                'lighting condition'))
        if column_order != (0, 1, 2, 3, 4):
            raise ValueError("Unexpected label column order: %s" %
                             str(column_order))

    check_label_ordering(norb)

    def get_rowmask(labels, category, instance, lighting):
        column_indices = (0, 1, 4)  # category, instance, lighting
        label_values = (category, instance, lighting)
        rowmasks = (labels[:, c] == v
                    for c, v in safe_zip(column_indices, label_values))
        return reduce(numpy.logical_and, rowmasks)

    left_images = norb.get_topological_view(single_tensor=False)[0]

    rowmask = get_rowmask(norb.y,
                          args.category,
                          args.instance,
                          args.lighting)

    left_images = left_images[rowmask, :]
    labels = norb.y[rowmask, :]

    elevation_index = 2
    azimuth_index = 3

    def get_sorted_unique_elements(values):
        return numpy.sort(numpy.asarray(tuple(frozenset(values))))

    elevations, azimuths = (get_sorted_unique_elements(labels[:, c])
                            for c in (elevation_index, azimuth_index))

    azimuths = azimuths[::args.azimuth_downsample]
    elevations = elevations[::args.elevation_downsample]

    def get_angles_to_index_map(labels):
        elevations = labels[:, elevation_index]
        azimuths = labels[:, azimuth_index]
        angles = tuple((e, a) for e, a in safe_zip(elevations, azimuths))
        return dict(safe_zip(angles, range(len(elevations))))

    angles_to_index = get_angles_to_index_map(labels)

    fig_size_inches = (len(azimuths) * 2,
                       len(elevations) * 2)
    figure, all_axes = pyplot.subplots(len(elevations),
                                       len(azimuths),
                                       squeeze=False,
                                       figsize=fig_size_inches)

    for axes_row, elevation in safe_zip(all_axes, elevations):
        for axes, azimuth in safe_zip(axes_row, azimuths):
            # hides axes' tickmarks
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)

            row_index = angles_to_index[(elevation, azimuth)]
            left_image = left_images[row_index, :, :, 0]
            axes.imshow(left_image, cmap='gray')

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()


if __name__ == '__main__':
    main()
