import numpy
import theano
from pylearn2.space import (Conv2DSpace)
from pylearn2.utils import safe_zip
from pylearn2.models.mlp import MLP
from pylearn2.models.maxout import Maxout, MaxoutConvC01B
from pylearn2.scripts.papers.maxout.norb.resize_input_of_model import (
    convert_Maxout_to_MaxoutConvC01B,
    get_convolutional_equivalent)


def _test_equivalence(original_layer, conv_layer, batch_size, rng):
    """
    Test that original_layer and conv_layer perform equivalent calculations.
    """
    def get_func(layer):
        """
        Returns the numerical function implemented by layer.fprop().
        """
        input_batch_symbol = layer.get_input_space().make_theano_batch()
        output_batch_symbol = layer.fprop(input_batch_symbol)
        return theano.function(input_batch_symbol, [output_batch_symbol])

    funcs = [get_func(layer) for layer in (original_layer, conv_layer)]

    original_input = original_layer.get_origin_batch(batch_size=batch_size)
    original_input[...] = rng.uniform(low=-4.0,
                                      high=4.0,
                                      size=original_input.shape)

    conv_input = original_layer.get_input_space().np_format_as(
        original_input,
        conv_layer.get_input_space())

    inputs = (original_input, conv_input)

    original_output, conv_output = tuple(f(x)
                                         for f, x in safe_zip(funcs, inputs))

    # reshape original output to conv output
    original_output = original_layer.get_output_space().np_format_as(
        original_output,
        conv_layer.get_output_space())

    assert numpy.all(original_output == conv_output)


def test_convert_Maxout_to_MaxoutConvC01B(rng=None):
    input_shape = (2, 2, 96)
    input_space = Conv2DSpace(shape=input_shape[:2],
                              num_channels=input_shape[2])
    batch_size = 10
    seed = 1234

    maxout = Maxout(layer_name='test_maxout_layer_name',
                    irange=.05,
                    num_units=50,
                    num_pieces=3,
                    min_zero=True,
                    max_col_norm=1.9)

    # Sadly, we need to wrap the layers in MLPs, because their
    # set_input_space() methods call self.mlp.rng, and it's the MLP constructor
    # that sets its layers' self.mlp fields.
    maxout_mlp = MLP(layers=[maxout],
                     batch_size=batch_size,
                     input_space=input_space,
                     seed=seed)

    maxout_conv = convert_Maxout_to_MaxoutConvC01B(maxout)
    conv_mlp = MLP(layers=[maxout],
                   batch_size=batch_size,
                   input_space=input_space,
                   seed=seed)

    if rng is None:
        rng = numpy.random.RandomState(1234)

    test_equivalence(maxout, maxout_conv, batch_size, rng)
    test_equivalence(maxout_mlp, conv_mlp, batch_size, rng)


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
