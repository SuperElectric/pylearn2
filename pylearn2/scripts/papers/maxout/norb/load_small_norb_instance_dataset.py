#! /usr/bin/env python
"""
This script loads the instance recognition datasset and creates a ZCA_Dataset
with it, and visualizes both the ZCA'ed and un-ZCA'ed images.
"""

from __future__ import print_function
import sys, numpy, argparse, matplotlib
from matplotlib import pyplot
from pylearn2.space import Conv2DSpace, VectorSpace, CompositeSpace
from pylearn2.scripts.papers.maxout.norb \
    import load_small_norb_instance_dataset


def parse_args():

    parser = argparse.ArgumentParser(description="A simple viewer that steps "
                                     "through the preprocesed NORB instance "
                                     "dataset")

    parser.add_argument('-i',
                        '--input',
                        type=str,
                        required=True,
                        help="The .pkl file of the NORB instance dataset.")

    return parser.parse_args()


def main():
    args = parse_args()

    print("loading zca_dataset")
    zca_dataset = load_small_norb_instance_dataset(args.input)
    print("loaded zca_dataset")

    image_shape = (int(numpy.sqrt(zca_dataset.X.shape[1])), ) * 2
    assert numpy.prod(image_shape) == zca_dataset.X.shape[1]

    space = CompositeSpace((Conv2DSpace(shape=image_shape, num_channels=1),
                            VectorSpace(dim=zca_dataset.y.shape[1])))
    sources = ("features", "targets")

    figure, all_axes = pyplot.subplots(1, 2)

    iterator = zca_dataset.iterator(mode='sequential',
                                    batch_size=1,
                                    data_specs=(space, sources))
    zca_image, label = iterator.next()

    def show_next_image():
        zca_image, label = iterator.next()
        zca_image = zca_image[0, :, :, 0]

        row_image = zca_dataset.X[:1, :].reshape(image_shape)
        all_axes[1].imshow(row_image, cmap='gray', interpolation='nearest')

        # Neither ZCA_Dataset.mapback() nor .mapback_for_viewer() actually
        # map the image back to its original form, because the dataset's
        # preprocessor is unaware of the global contrast normalization we
        # did.
        #
        # Therefore we rely on matpotlib's pixel normalization that
        # happens by default.
        all_axes[0].imshow(zca_image,
                           cmap='gray',
                           #norm=matplotlib.colors.no_norm(),
                           interpolation='nearest')
        figure.canvas.draw()

    show_next_image()

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)
        elif event.key == ' ':
            show_next_image()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()


if __name__ == '__main__':
    main()
