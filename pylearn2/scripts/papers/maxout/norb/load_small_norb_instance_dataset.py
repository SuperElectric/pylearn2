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
        print("zca_image.shape: %s" % str(zca_image.shape))
        print("elem[1].shape: %s" % str(label.shape))


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
    # P_ = zca_dataset.preprocessor.P_
    # print("P_.shape: %s" % str(P_.shape))
    # print("P's diagonals are 1.0: %s" % str((numpy.diag(P_) == 1.0).all()))
    # print("P is diagonal: %s" % str((numpy.abs(numpy.diag(numpy.ones(P_.shape[0])) - P_) < 0.00001).all()))
    # print("# of nonzeros: %d" % numpy.count_nonzero(numpy.abs(P_) < 0.00001))
    # all_axes[0].imshow(P_, cmap='gray')
    # all_axes[1].plot(range(len(zca_dataset.preprocessor.eigs)),
    #                  zca_dataset.preprocessor.eigs)
    # all_axes[1].hist(P_.flatten())

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)
        elif event.key == ' ':
            show_next_image()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

    pyplot.show()
    # for elem in zca_dataset.iterator(mode='sequential',
    #                                  batch_size=1,  # from cifar-10.yaml
    #                                  data_specs=(space, sources)):

    #     # Show zca'ed image

    #     # Show un-zca'ed image

    #     # Display labels

    #     print("len(elem): %d" % len(elem))
    #     print("elem[0].shape: %s" % str(elem[0].shape))
    #     print("elem[1].shape: %s" % str(elem[1].shape))
    #     # print("in elem loop")
    #     # for thing in elem:
    #     #     print(thing.shape)
    #     # print(elem)
    #     sys.exit(0)

if __name__ == '__main__':
    main()
