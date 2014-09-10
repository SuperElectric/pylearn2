#! /usr/bin/env python

import sys, argparse, numpy
import os.path
from pylearn2.utils import serial, safe_zip
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.models.mlp import MLP, Layer, Softmax
from pylearn2.models.maxout import Maxout, MaxoutConvC01B


# class ConvolutionalFCLayer(Layer):
#     """
#     A fully-connected layer, to be applied convolutionally.

#     Given a fully-connected layer with M input dimensions and N output
#     dimensions, this applies the operation convolutionally as N 1x1xM
#     convolutions.

#     Parameters
#     ----------
#     fc_layer: a fully-connected layer.
#       fc_layer.set_input_space() must have already been called.
#     """
#     def __init__(self, fc_layer):
#         assert isinstance(fc_layer.get_input_space(), Conv2DSpace)

#         self.fc_input_space = fc_layer.get_input_space()
#         self.fc_output_space = fc_layer.get_output_space()

#         self.conv_fc_layer = instantiate_convolutional_equivalent(fc_layer)

#         # self.conv_input_num_channels = fc_layer.get_input_space.num_channels
#         # self.fc_input_dim = fc_layer.input_dim
#         # assert fc_layer.requires_reformat
#         # assert hasattr(fc_layer, 'desired_space')

#         # self.old_input_space = fc_layer.get_input_space()

#         # self.output_vector_space = fc_layer.desired_space
#         # self.output_conv_space = Conv2DSpace(shape=(1, 1),
#         #                                      num_channels=desired_space.dim,
#         #                                      dtype=desired_space.dtype)

#     def get_params(self):
#         return self.fc_layer.get_params()

#     def set_input_space(self, input_space):
#         """

#         """
#         assert isinstance(input_space, Conv2DSpace)
#         assert input_space.num_channels == self.fc_input_space.num_channels
#         for old_dim, new_dim in safe_zip(self.fc_input_space.shape,
#                                          input_space.shape):
#             assert old_dim <= new_dim

#         output_shape = (numpy.array(input_space.shape) -
#                         numpy.array(self.fc_input_space.shape))
#         self.output_space = Conv2DSpace(shape=output_shape,
#                                         num_channels=self.fc_output_space.ndim,
#                                         dtype=self.fc_output_space.dtype)
#         self.input_space = input_space

#     def get_input_space(self):
#         return self.input_space

#     def get_output_space(self):
#         if not hasattr(self, 'output_space'):
#             raise RuntimeError('Must call set_input_space() first.')

#         return self.output_space

#     def fprop(self, batch):
#         self.input_space.validate(batch)


#         return batch


def instantiate_MaxoutConvC01B_from_Maxout(maxout):
    """
    Takes a Maxout layer and returns a compatibly-shaped MaxoutConvC01B.

    maxout: Maxout
    returns: MaxoutConvC01B
      Weights are uninitialized.
    """
    assert isinstance(maxout, Maxout)
    assert maxout.mask_weights is None
    assert maxout.pool_stride == maxout.num_pieces
    assert not maxout.randomize_pools
    assert isinstance(maxout.get_input_space(), Conv2DSpace)

    old_input_space = maxout.get_input_space()

    return MaxoutConvC01B(num_channels=maxout.num_units,
                          num_pieces=maxout.num_pieces,
                          kernel_shape=old_input_space.shape,
                          pool_shape=(1, 1),
                          pool_stride=(1, 1),
                          layer_name=maxout.layer_name + "_convolutionalized",
                          irange=1.0,  # needed for set_input_space
                          # init_bias,
                          W_lr_scale=maxout.W_lr_scale,
                          b_lr_scale=maxout.b_lr_scale,
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
                          max_kernel_norm=maxout.max_col_norm,
                          # input_normalization,
                          # detector_normalization
                          # output_normalization
                          min_zero=maxout.min_zero,
                          kernel_stride=(1, 1))


def copy_params_from_Maxout_to_MaxoutConv2D(maxout, maxout_conv):
    assert isinstance(maxout, Maxout)
    assert isinstance(maxout_conv, MaxoutConvC01B)

    weights, biases = [x.get_value() for x in maxout.get_params()]
    print("original weights.shape: %s" % str(weights.shape))
    print("original biases.shape: %s" % str(biases.shape))
    print("maxout layer's num_units: %d" % maxout.num_units)

    conv_params = tuple(p.get_value() for p in maxout_conv.get_params())

    print("conv layer's num_channels: %d" % maxout_conv.num_channels)
    print("conv layer's num dummy channels: %d" % maxout_conv.dummy_channels)
    print("conv weights' shape: %s" % str(conv_params[0].shape))
    print("conv biases' shape: %s" % str(conv_params[1].shape))

    def get_kernel_input_space(maxout):
        maxout_input_space = maxout.get_input_space()
        if isinstance(maxout_input_space, VectorSpace):
            shape = (1, 1)
            num_channels = maxout_input_space.dim
        elif isinstance(maxout_input_space, Conv2DSpace):
            shape = maxout_input_space.shape
            num_channels = maxout_input_space.num_channels
        else:
            raise TypeError()

        print("kernel_space's num_channels: %d" % num_channels)
        return Conv2DSpace(shape=shape,
                           num_channels=num_channels,
                           axes=('c', 0, 1, 'b'),
                           dtype=maxout_input_space.dtype)

    kernel_space = get_kernel_input_space(maxout)
    actual_kernel_input_channels = (kernel_space.num_channels +
                                    maxout_conv.dummy_channels)
    actual_kernel_output_channels = (maxout.get_output_space().dim *
                                     maxout.num_pieces)
    conv_weights = numpy.zeros((actual_kernel_input_channels,
                                kernel_space.shape[0],
                                kernel_space.shape[1],
                                actual_kernel_output_channels),
                               dtype=weights.dtype)
    assert conv_weights.shape == conv_params[0].shape
    conv_weights[:kernel_space.num_channels, ...] = \
        maxout.desired_space.np_format_as(weights.transpose(), kernel_space)

    print("conv_weights.shape: %s" % str(conv_weights.shape))

    theano_conv_weights, theano_conv_biases = maxout_conv.get_params()

    theano_conv_weights.set_value(conv_weights)
    theano_conv_biases.set_value(biases)  # no need to reshape biases.

    # print("maxout weights: \n%s" % weights)
    # print("\nconv weights:\n%s" % theano_conv_weights.get_value())


def copy_params(old_layer, conv_layer):
    if isinstance(old_layer, Maxout):
        copy_params_from_Maxout_to_MaxoutConv2D(old_layer, conv_layer)
    else:
        raise NotImplementedError("copy_params not yet implemented for "
                                  "copying from %s to %s." %
                                  (type(old_layer), type(conv_layer)))


class SoftmaxConvC01B(Layer):
    pass


def instantiate_convolutional_equivalent(layer):
    """
    Returns an equivalent convolutional layer, with uninitialized weights.

    If <layer> is convolutional, returns a copy with uninitialized weights.

    If <layer> is a M-input, N-output linear layer, this returns a
    convolutional layer with N channels and 1x1xM-shaped convolutional kernels.

    The layer mappings currently implemented are:

      Maxout  -> MaxoutConvC01B
      Softmax -> SoftmaxConvC01B
    """

    assert isinstance(layer, Layer)

    if isinstance(layer, MaxoutConvC01B):
        pass
    if isinstance(layer, SoftmaxConvC01B):
        pass
    elif isinstance(layer, Maxout):
        return instantiate_MaxoutConvC01B_from_Maxout(layer)
    elif isinstance(layer, Softmax):
        pass

    raise NotImplementedError("Conversion of layer type %s not implemented." %
                              type(layer))


def resize_mlp_input(old_mlp, new_image_shape):
    """
    Returns a copy of old_mlp that operates on a new, larger image shape.

    Parameters:
    -----------

    old_mlp: MLP
      First layer must be convolutional.

    new_image_shape: tuple or list
      2-D image shape (rows, columns). Each dimension must be at least as big
      as the equivalent in old_mlp.input_shape.

    returns: MLP
      Equivalent to old_mlp, but with input space resized to new_image_shape.
      Contains the same parameters (weights, biases) as old_mlp.
      Non-convolutional layers are replaced with 1x1 convolutional layers.
    """
    assert isinstance(old_mlp, MLP)
    assert len(new_image_shape) == 2

    old_input_space = old_mlp.get_input_space()
    assert isinstance(old_input_space, Conv2DSpace)
    assert len(old_input_space.shape) == 2
    for old_dim, new_dim in safe_zip(old_input_space.shape, new_image_shape):
        if new_dim < old_dim:
            raise ValueError("image_shape %s must be at least as big as the "
                             "original image shape %s." %
                             (str(new_image_shape),
                              str(old_mlp.input_space.shape)))

    dtype = old_input_space.dtype
    if dtype is None:
        dtype = old_mlp.layers[0].get_params[0].get_value().dtype

    new_input_space = Conv2DSpace(shape=new_image_shape,
                                  num_channels=old_input_space.num_channels,
                                  axes=old_input_space.axes,
                                  dtype=dtype)

    new_mlp = MLP(layers=[instantiate_convolutional_equivalent(x)
                          for x in old_mlp.layers],
                  batch_size=old_mlp.batch_size,
                  input_space=new_input_space,
                  seed=1234)  # used to init weights. We'll overwrite them.

    for old_layer, new_layer in safe_zip(old_mlp.layers, new_mlp.layers):
        copy_params(old_layer, new_layer)

    return new_mlp

    # make_purely_convolutional(mlp)

    # # Retain copies of the layer parameters, before set_input_space nukes them.
    # filters = [layer.transformer.get_params() for layer in mlp.layers]
    # biases = [layer.b for layer in mlp.layers]

    # bigger_input_space = Conv2DSpace(shape=new_image_shape,
    #                                  num_channels=mlp.input_space.num_channels,
    #                                  axes=mlp.input_space.axes,
    #                                  dtype=mlp.input_space.dtype)
    # mlp.set_input_space(bigger_input_space)

    # # Restore layer parameters
    # for layer, filter, bias in safe_zip(mlp.layers, filters, biases):
    #     layer.transformer._filters = filters
    #     layer.bias = bias


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

        return args

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
