#! /usr/bin/env python

import sys, argparse
import os.path
from pylearn2.utils import serial


def resize_mlp_input(mlp, new_image_shape):
    assert isinstance(mlp, MLP)
    assert len(new_image_shape) == 2

    old_image_shape = mlp.input_space.shape[:2]

    for old_dim, new_dim in safe_zip(old_image_shape, new_image_shape):
        if new_dim < old_dim:
            raise ValueError("image_shape %s must be at least as big as the "
                             "original image shape %s." %
                             (str(new_image_shape), str(old_image_shape)))

    layers = [resize_layer(layer, image_shape


def main():
    """Entry point of this script."""

    def parse_args():
        """Parses command-line arguments, returns them as a dict."""

        parser = argparse.ArgumentParser(
            description=("Converts a model (given as a .pkl file) to an "
                         "equivalent model with a different input image size. "
                         "The output is a .pkl file that stores a dictionary "
                         "containing the following keys:\n"
                         "\n"
                         "  'model':  An MLP model M.\n"
                         "  'offset': A tuple of ints (U, V).\n"
                         "  'scale':  An int S.\n"
                         "\n"
                         "M maps an image to a 2-D classification map F. The "
                         "classification at F[i, j] corresponds to the image "
                         "region between the corners:\n"
                         "  min_corner = (U, V) + (i, j) * S\n"
                         "  max_corner = (U, V) + (i+1, j+1) * S"))

        parser.add_argument('--input',
                            '-i',
                            required=True,
                            help=("The .pkl file of a model."))

        parser.add_argument('--output',
                            '-o',
                            required=True,
                            help=("The file path to save the dict to."))

        parser.add_argument('--image_shape',
                            '-s',
                            type=int,
                            nargs=2,
                            required=True,
                            help=("The new shape of the input image. Must be "
                                  "at least as big as the original image, in "
                                  "both dimensions."))

        args = parser.parse_args()
        if not os.path.isfile(args.input):
            print "Couldn't find file %s. Exiting." % args.input
            sys.exit(1)

        output_abspath = os.path.abspath(args.output)
        if not os.path.isdir(os.path.split(output_abspath)[0]):
            print("Couldn't find parent directory of output file %s. Exiting."
                  % output_abspath)
            sys.exit(1)

    args = parse_args()
    mlp = serial.load(args.input)
    try:
        result = resize_mlp_input(mlp, args.image_shape)
    except ValueError, value_error:
        if value_error.what().find(" must be at least as big as the original "
                                   "image shape ") >= 0:
            print value_error.what() + " Exiting."
            sys.exit(1)
        else:
            raise

    serial.save(result, args.output)


if __name__ == '__main__':
    main()
