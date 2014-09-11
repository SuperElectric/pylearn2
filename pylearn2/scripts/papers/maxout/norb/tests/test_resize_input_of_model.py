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
    resize_mlp_input)


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
    assert original_layer.get_input_space().axes == ('c', 0, 1, 'b')
    assert conv_layer.get_input_space().axes == ('c', 0, 1, 'b')

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
    small_images[...] = rng.uniform(low=-4.0,
                                    high=4.0,
                                    size=small_images.shape)
    big_images[...] = rng.uniform(low=-4.0,
                                  high=4.0,
                                  size=big_images.shape)
    big_images[:,
               :small_images.shape[1],
               :small_images.shape[2],
               :] = small_images

    print("original input space axes: %s" %
          str(original_layer.get_input_space().axes))
    print("conv input space axes: %s" % str(conv_layer.get_input_space().axes))

    original_output = original_func(small_images)
    conv_output = conv_func(big_images)

    print("original output shape: %s" % str(original_output.shape))

    # reshape original output to conv output
    def get_original_conv_output_space(conv_layer):
        output_space = conv_layer.get_output_space()
        assert output_space.axes == ('c', 0, 1, 'b')

        return Conv2DSpace(shape=(1, 1),
                           num_channels=output_space.num_channels,
                           axes=output_space.axes,
                           dtype=output_space.dtype)

    original_output = original_layer.get_output_space().np_format_as(
        original_output,
        get_original_conv_output_space(conv_layer))

    print("np_format_as'ed original output shape: %s" %
          str(original_output.shape))
    print("conv output shape: %s" % str(conv_output.shape))

    # compare original_output vector with the 0, 0'th pixel of the conv output
    abs_difference = numpy.abs(original_output - conv_output[:, :1, :1, :])
    assert numpy.all(abs_difference < 1e-06), ("max abs difference: %g" %
                                               abs_difference.max())


def test_convert_Maxout_to_MaxoutConvC01B(rng=None):
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
                              axes=('c', 0, 1, 'b'))
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

    size_increase = 1
    conv_mlp = resize_mlp_input(softmax_mlp,
                                (input_shape[0] + size_increase,
                                 input_shape[1] + size_increase))

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
