"""
Tests ../resize_input_of_model.py
"""

import numpy
import theano
from pylearn2.space import Conv2DSpace
from pylearn2.utils import safe_zip
from pylearn2.models.mlp import Layer, MLP, Softmax
from pylearn2.models.maxout import Maxout
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

    # conv_input = original_layer.get_input_space().np_format_as(
    #     original_input,
    #     conv_layer.get_input_space())
    print("original input space axes: %s" %
          str(original_layer.get_input_space().axes))
    print("conv input space axes: %s" % str(conv_layer.get_input_space().axes))

    # DEBUG (the following are temporary)
    # assert small_images.shape == big_images.shape
    # assert original_layer.get_input_space() == conv_layer.get_input_space()

    original_output = original_func(small_images)
    conv_output = conv_func(big_images)

    print("original output shape: %s" % str(original_output.shape))

    # original_output, conv_output = tuple(f(input_batch) for f in funcs)

    # reshape original output to conv output
    def get_original_conv_output_space(conv_layer):
        output_space = conv_layer.get_output_space()
        assert output_space.axes == ('c', 0, 1, 'b')

        conv_weights = conv_layer.get_params()[0].get_value()

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

    # print("np_reformatted original output:\n%s" % original_output)
    # print("conv output:\n%s" % conv_output)

    # DEBUG (see if any output pixel matches the original)
    for r in range(conv_output.shape[1]):
        for c in range(conv_output.shape[2]):
            max_abs_diff = numpy.abs(original_output -
                                     conv_output[:, r:r+1, c:c+1, :]).max()
            print("abs diff for %d %d: %g" % (r, c, max_abs_diff))


    # compare original_output vector with the 0, 0'th pixel of the conv output
    abs_difference = numpy.abs(original_output - conv_output[:, :1, :1, :])
    assert numpy.all(abs_difference < 1e-06), ("max abs difference: %g" %
                                               abs_difference.max())
    # assert (original_output == conv_output[:, :1, :1, :]).all()


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

    # assert (maxout.num_units * maxout.num_pieces) % 16 == 0, \
    #     ("Bug in test setup: cuda_convnet requires that num_channels * "
    #      "num_pieces be divisible by 16. num_channels: %d, num_pieces: "
    #      "%d, product %% 16: %d" %
    #      (maxout.num_units,
    #       maxout.num_pieces,
    #       (maxout.num_units * maxout.num_pieces) % 16))

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


# def test_get_convolutional_equivalent():
#     test_dir = os.path.split(os.path.realpath(__file__))[0]
#     model = yaml_parse.load_path(os.path.join(test_dir, 'test_model.yaml'))

#     def test_impl(layer, rng):
#         def get_func(layer):
#             input_batch_symbol = layer.get_input_space().make_theano_batch()
#             output_batch_symbol = layer.fprop(input_batch_symbol)
#             return theano.function(input_batch_symbol, [output_batch_symbol])

#         conv_layer = get_convolutional_equivalent(layer)

#         original_func = get_func(layer)
#         conv_func = get_func(conv_layer)

#         for iteration in range(5):
#             input_batch = layer.get_input_space().get_origin_batch()
#             input_batch[...] = rng.uniform(input_batch.shape())

#             original_output = layer_func(input_batch)

#             original_outspace = layer.get_output_space()
#             conv_outspace = conv_layer.get_output_space()
#             expected_conv_output = original_outpspace.np_convert(output_batch,
#                                                                  conv_outspace)

#             original_inspace = layer.get_input_space()
#             conv_inspace = conv_layer.get_input_space()
#             conv_input_batch = original_inspace.np_convert(input_batch,
#                                                            conv_inspace)
#             conv_output = conv_func(conv_input_batch)

#             assert (expected_conv_output == conv_output).all()
