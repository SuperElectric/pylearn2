#! /usr/bin/env python

import sys, argparse
import os.path
from pylearn2.utils import serial
from pylearn2.space import Conv2DSpace
from pylearn2.models.mlp import Softmax
from pylearn2.models.maxout import Maxout, MaxoutLocalC01B


def make_purely_convolutional(mlp):
    assert isinstance(mlp, MLP)
    model.layers = [get_convolutional_equivalent(x) for x in model.layers]


def get_convolutional_equivalent(layer):
    """
    Returns an equivalent convolutional layer.

    If <layer> is convolutional, it is returned as-is (no-op).

    If <layer> is a M-input, N-output linear layer, this returns a
    convolutional layer with N channels and 1x1xM-shaped convolutional kernels.

    The layer mappings currently implemented are:

      Maxout  -> MaxoutConvC01B
      Softmax -> SoftmaxConvC01B
    """

    assert isinstance(layer, Layer)

    elif isinstance(layer, (MaxoutConvC01B, SoftmaxConvC01B)):
        return layer
    elif isinstance(layer, Maxout):
        assert layer.mask_weights is None
        assert layer.pool_stride == layer.num_pieces
        assert not layer.randomize_pools
        result = MaxoutConvC01B(num_channels=layer.num_units,
                                num_pieces=layer.num_pieces,
                                kernel_shape=(1, 1),
                                pool_shape=(1, 1),
                                pool_stride=(1, 1),
                                layer_name=layer.name,
                                # irange
                                # init_bias,
                                W_lr_scale=layer.W_lr_scale,
                                b_lr_scale=layer.b_lr_scale,
                                pad=0,
                                fix_pool_shape=False,
                                fix_pool_stride=False,
                                fix_kernel_shape=False,
                                # partial_sum=1,
                                tied_b=True,
                                # see pylearn2.linear.matrixmul.lmul. Matrix
                                # multiplication is applied as T.dot(x,
                                # self._W). This makes sense in retrospect;
                                # since the batch vectors are stored in rows of
                                # <x>, the corresponding weights are stored in
                                # columns of W. So max_kernel_norm is set to
                                # max_col_norm here.
                                max_kernel_norm=layer.max_col_norm,
                                # input_normalization,
                                # detector_normalization
                                # output_normalization
                                min_zero=layer.min_zero,
                                kernel_stride=(1, 1))
        weights, bias = result.get_params()

    elif isinstance(layer, Softmax):
        pass

    raise TypeError("Conversion of layer type %s not implemented." %
                    type(layer))


def resize_mlp_input(mlp, new_image_shape):
    """
    Calls make_purely_convolutional on mlp, then resizes its input space
    to take a bigger image.
    """

    assert isinstance(mlp, MLP)
    assert len(new_image_shape) == 2
    assert isinstance(mlp.input_space, Conv2DSpace)
    assert len(mlp.input_space.shape) == 3

    for old_dim, new_dim in safe_zip(mlp.input_space.shape, new_image_shape):
        if new_dim < old_dim:
            raise ValueError("image_shape %s must be at least as big as the "
                             "original image shape %s." %
                             (str(new_image_shape), str(old_image_shape)))

    mlp = make_purely_convolutional(mlp)

    # Retain copies of the layer parameters, before set_input_space nukes them.
    filters = [layer.transformer.get_params() for layer in mlp.layers]
    biases = [layer.b for layer in mlp.layers]

    bigger_input_space = Conv2DSpace(shape=new_image_shape,
                                     num_channels=mlp.input_space.num_channels,
                                     axes=mlp.input_space.axes,
                                     mlp.input_space.dtype)
    mlp.set_input_space(bigger_input_space)

    # Restore layer parameters
    for layer, filter, bias in safe_zip(mlp.layers, filters, biases):
        layer.transformer._filters = filters
        layer.bias = bias


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
