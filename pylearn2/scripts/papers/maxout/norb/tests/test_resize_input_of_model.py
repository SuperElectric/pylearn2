"""
Tests ../resize_input_of_model.py
"""

import numpy
import theano
from pylearn2.space import Conv2DSpace
from pylearn2.utils import safe_zip
from pylearn2.models.mlp import Layer, MLP, Softmax
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.scripts.papers.maxout.norb.resize_input_of_model import (
    resize_mlp_input, SoftmaxConvC01B, SoftmaxConv)


def _convert_axes(from_axes, from_batch, to_axes):
    return from_batch.transpose(tuple(from_axes.index(a) for a in to_axes))


def _test_equivalence(original_layer, conv_layer, batch_size, rng):
    """
    Test that original_layer and conv_layer perform equivalent calculations.
    """
    assert isinstance(original_layer, Layer)
    assert isinstance(conv_layer, Layer)
    assert batch_size > 0
    assert isinstance(rng, numpy.random.RandomState)

    assert isinstance(original_layer.get_input_space(), Conv2DSpace)
    assert isinstance(conv_layer.get_input_space(), Conv2DSpace)
    # assert original_layer.get_input_space().axes == ('c', 0, 1, 'b')
    # assert conv_layer.get_input_space().axes == ('c', 0, 1, 'b')

    def get_func(input_name, layer):
        """
        Returns the numerical function implemented by layer.fprop().
        """
        input_batch_symbol = \
            layer.get_input_space().make_theano_batch(name=input_name,
                                                      batch_size=batch_size)
        output_batch_symbol = layer.fprop(input_batch_symbol)
        return theano.function((input_batch_symbol, ), output_batch_symbol)

    original_func = get_func('small_images', original_layer)
    conv_func = get_func('big_images', conv_layer)

    small_images = \
        original_layer.get_input_space().get_origin_batch(batch_size)

    big_images = \
        conv_layer.get_input_space().get_origin_batch(batch_size)

    print("small_images.shape: %s" % str(small_images.shape))
    print("big_images.shape: %s" % str(big_images.shape))
    print("conv_layer.weights.shape %s" %
          str(conv_layer.layers[0].get_params()[0].get_value().shape))
    small_images.flat[:] = numpy.arange(small_images.size)
    big_images.flat[:] = rng.uniform(low=-4.0,
                                     high=4.0,
                                     size=big_images.size)

    # temporarily transpose to a predetermined axis order
    b01c = ('b', 0, 1, 'c')
    print("original_layer.get_input_space().axes: %s" %
          str(original_layer.get_input_space().axes))
    print("small_images.shape: %s" % str(small_images.shape))

    small_images = _convert_axes(original_layer.get_input_space().axes,
                                 small_images,
                                 b01c)
    print("after transpose to b01c, small_images.shape: %s" % str(small_images.shape))
    big_images = _convert_axes(conv_layer.get_input_space().axes,
                               big_images,
                               b01c)
    # small_images = small_images.transpose(tuple(b01c.index(a)
    #                                             for a in original_layer.get_input_space().axes))
    # big_images = big_images.transpose(tuple(b01c.index(a)
    #                                         for a in conv_layer.get_input_space().axes))

    # Copy small image to upper-left corner of big image
    big_images[:,
               :small_images.shape[1],
               :small_images.shape[2],
               :] = small_images

    # restore original axis order
    small_images = _convert_axes(b01c,
                                 small_images,
                                 original_layer.get_input_space().axes)
    big_images = _convert_axes(b01c,
                               big_images,
                               conv_layer.get_input_space().axes)

    # small_images = small_images.transpose(tuple(original_layer.get_input_space().axes.index(a)
    #                                             for a in b01c))
    # big_images = big_images.transpose(tuple(conv_layer.get_input_space().axes.index(a)
    #                                         for a in b01c))

    print("original input space axes: %s" %
          str(original_layer.get_input_space().axes))
    print("conv input space axes: %s" % str(conv_layer.get_input_space().axes))

    print("small_images.shape: %s" % str(small_images.shape))
    original_output = original_func(small_images)
    print("big_images.shape: %s" % str(big_images.shape))
    conv_output = conv_func(big_images)

    print("original output shape: %s" % str(original_output.shape))

    def get_original_c01b_output_space(conv_layer):
        output_space = conv_layer.get_output_space()
        # assert output_space.axes == ('c', 0, 1, 'b')

        return Conv2DSpace(shape=(1, 1),
                           num_channels=output_space.num_channels,
                           # this need not be the same as output_space.axes
                           axes=('c', 0, 1, 'b'),
                           #axes=output_space.axes,
                           dtype=output_space.dtype)

    # reshape original output to conv output
    original_output = original_layer.get_output_space().np_format_as(
        original_output,
        get_original_c01b_output_space(conv_layer))

    # def get_conv_c01b_output_space(conv_layer):
    #     output_space = conv_layer.get_output_space()
    #     return Conv2DSpace(shape=output_space.shape,
    #                        num_channels=output_space.num_channels,
    #                        axes=('c', 0, 1, 'b'),
    #                        dtype=output_space.dtype)

    # transposes conv output to C01B order, if it isn't already
    conv_output = _convert_axes(conv_layer.get_output_space().axes,
                                conv_output,
                                ('c', 0, 1, 'b'))
    # conv_output = conv_output.transpose(tuple(conv_layer.get_output_space().axes.index(a)
    #                                           for a
    #                                           in ('c', 0, 1, 'b')))
    # conv_c01b_output_space = get_conv_c01b_output_space(conv_layer)
    # conv_output = conv_layer.get_output_space().np_format_as(
    #     conv_output,
    #     conv_c01b_output_space)

    print("np_format_as'ed original output shape: %s" %
          str(original_output.shape))
    print("conv output shape: %s" % str(conv_output.shape))

    # compare original_output vector with the 0, 0'th pixel of the conv output
    abs_difference = numpy.abs(original_output - conv_output[:, :1, :1, :])
    assert numpy.all(abs_difference < 1e-06), ("max abs difference: %g" %
                                               abs_difference.max())


def _test_convert_Maxout_to_MaxoutConvC01B(rng=None):
    input_shape = (2, 2, 12)
    assert input_shape[0] == input_shape[1], ("Bug in test setup: Image not "
                                              "square. MaxoutConvC01B requires"
                                              " square images.")

    input_space = Conv2DSpace(shape=input_shape[:2],
                              num_channels=input_shape[2],
                              axes=('c', 0, 1, 'b'))
    batch_size = 5
    seed = 1234

    maxout = Maxout(layer_name='test_maxout_layer_name',
                    irange=.05,
                    num_units=4,
                    num_pieces=4,
                    min_zero=True,
                    max_col_norm=1.9)

    assert (maxout.num_units * maxout.num_pieces) % 16 == 0, \
        ("Bug in test setup: cuda_convnet requires that num_channels * "
         "num_pieces be divisible by 16. num_channels: %d, num_pieces: "
         "%d, product %% 16: %d" %
         (maxout.num_units,
          maxout.num_pieces,
          (maxout.num_units * maxout.num_pieces) % 16))

    # Sadly, we need to wrap the layers in MLPs, because their
    # set_input_space() methods call self.mlp.rng, and it's the MLP constructor
    # that sets its layers' self.mlp fields.
    maxout_mlp = MLP(layers=[maxout],
                     batch_size=batch_size,
                     input_space=input_space,
                     seed=seed)

    size_increase = 3
    conv_mlp = resize_mlp_input(maxout_mlp,
                                (input_shape[0] + size_increase,
                                 input_shape[1] + size_increase))

    if rng is None:
        rng = numpy.random.RandomState(1234)

    # test_equivalence(maxout, maxout_conv, batch_size, rng)
    _test_equivalence(maxout_mlp, conv_mlp, batch_size, rng)


def test_convert_Softmax_to_SoftmaxConvC01B(rng=None):
    input_shape = (2, 2, 8)
    assert input_shape[0] == input_shape[1], ("Bug in test setup: Image not "
                                              "square. MaxoutConvC01B requires"
                                              " square images.")

    input_space = Conv2DSpace(shape=input_shape[:2],
                              num_channels=input_shape[2],
                              axes=('b', 0, 1, 'c'))
                              #axes=('c', 0, 1, 'b'))
    softmax = Softmax(n_classes=16,  # must be divisible by 16
                      layer_name='test_softmax_layer_name',
                      irange=.05,
                      max_col_norm=1.9)

    # Sadly, we need to wrap the layers in MLPs, because their
    # set_input_space() methods call self.mlp.rng, and it's the MLP constructor
    # that sets its layers' self.mlp fields.

    batch_size = 5
    seed = 1234

    softmax_mlp = MLP(layers=[softmax],
                      batch_size=batch_size,
                      input_space=input_space,
                      seed=seed)

    print("mlp input space: %s" % softmax_mlp.get_input_space())
    print("mlp.layers[0] input space: %s" % softmax_mlp.layers[0].get_input_space())

    size_increase = 1
    conv_mlp = resize_mlp_input(softmax_mlp,
                                (input_shape[0] + size_increase,
                                 input_shape[1] + size_increase))

    assert isinstance(conv_mlp.layers[0], SoftmaxConvC01B)

    if rng is None:
        rng = numpy.random.RandomState(1234)

    # test_equivalence(maxout, maxout_conv, batch_size, rng)
    _test_equivalence(softmax_mlp, conv_mlp, batch_size, rng)


def _test_convert_Softmax_to_SoftmaxConv(rng=None):
    input_shape = (1, 2, 3)
    #input_shape = (1, 1, 8)
    # assert input_shape[0] == input_shape[1], ("Bug in test setup: Image not "
    #                                           "square. MaxoutConvC01B requires"
    #                                           " square images.")

    input_space = Conv2DSpace(shape=input_shape[:2],
                              num_channels=input_shape[2],
                              axes=('b', 'c', 0, 1)) #axes=('c', 0, 1, 'b'))  # change
    softmax = Softmax(n_classes=4,  # not divisible by 16
                      layer_name='test_softmax_layer_name',
                      irange=.05,
                      max_col_norm=1.9)

    # Sadly, we need to wrap the layers in MLPs, because their
    # set_input_space() methods call self.mlp.rng, and it's the MLP constructor
    # that sets its layers' self.mlp fields.

    batch_size = 1  # change
    seed = 1234

    softmax_mlp = MLP(layers=[softmax],
                      batch_size=batch_size,
                      input_space=input_space,
                      seed=seed)

    size_increase = 0
    conv_mlp = resize_mlp_input(softmax_mlp,
                                (input_shape[0] + size_increase,
                                 input_shape[1] + size_increase))

    assert isinstance(conv_mlp.layers[0], SoftmaxConv)

    if rng is None:
        rng = numpy.random.RandomState(1234)

    # test_equivalence(maxout, maxout_conv, batch_size, rng)
    _test_equivalence(softmax_mlp, conv_mlp, batch_size, rng)


def test_convert_stack():
    layers = [MaxoutConvC01B(num_channels=4,
                             num_pieces=4,
                             kernel_shape=(2, 2),
                             pool_shape=(1, 1),  # change
                             pool_stride=(1, 1),  # change
                             layer_name='h1',
                             irange=.05,
                             W_lr_scale=.05,
                             b_lr_scale=.05,
                             pad=0,  # change
                             fix_pool_shape=False,
                             fix_pool_stride=False,
                             fix_kernel_shape=False,
                             tied_b=True,
                             max_kernel_norm=1.9,
                             min_zero=True,
                             kernel_stride=(1, 1)),  # change
              Maxout(layer_name='h2, maxout',
                     irange=0.5,
                     num_units=8,
                     num_pieces=2,
                     min_zero=False,
                     max_col_norm=1.9),
              Softmax(n_classes=16,
                      layer_name='h3, softmax',
                      irange=0.5,
                      max_col_norm=1.9)]

    input_space = Conv2DSpace(shape=(4, 4),
                              num_channels=3,
                              dtype=theano.config.floatX,  # change to 'int32'
                              axes=('c', 0, 1, 'b'))  # change to b01c

    batch_size = 3
    mlp = MLP(layers=layers,
              batch_size=batch_size,
              input_space=input_space,
              seed=1234)

    input_shape = input_space.shape
    size_increase = 2
    resized_mlp = resize_mlp_input(mlp,
                                   (input_shape[0] + size_increase,
                                    input_shape[1] + size_increase))

    rng = numpy.random.RandomState(4321)

    _test_equivalence(mlp, resized_mlp, batch_size, rng=rng)
