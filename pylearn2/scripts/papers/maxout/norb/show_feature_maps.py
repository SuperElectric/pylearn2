#! /usr/bin/env python

"""
Steps through a dataset's images, showing the feature maps of selected levels.
"""


import argparse, sys, pdb
import theano, numpy
from matplotlib import pyplot
from pylearn2.utils import serial, sharedX, safe_zip
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.models.mlp import Softmax


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Steps through a dataset's images, showing the feature "
                     "maps of a given model."))

    parser.add_argument("-m",
                        "--model",
                        required=True,
                        help=".pkl file of the trained model")

    parser.add_argument("-d",
                        "--dataset",
                        required=True,
                        help=".pkl file of the (preprocessed) dataset.")

    # parser.add_argument("-l",
    #                     "--layers",
    #                     default=None,
    #                     nargs='+',
    #                     help=("Show outputs of these layer numbers "
    #                           "(0-indexed, counting from bottom."))

    return parser.parse_args()


def shuffle_axes(batch, batch_axes, output_axes):
    return numpy.transpose(batch, [batch_axes.index(a) for a in output_axes])


def main():
    args = parse_args()

    model = serial.load(args.model)

    def get_funcs(model):
        # def get_var_from_space(space):
        #     batch = space.get_origin_batch(batch_size=1)
        #     return sharedX(batch)

        # input_var = get_var_from_space(model.get_input_space())
        input_var = model.get_input_space().make_theano_batch()
        output_vars = model.fprop(input_var, return_all=True)

        return [theano.function([input_var], y) for y in output_vars]

    funcs = get_funcs(model)

    # output_spaces = [x.get_output_space() for x in model.layers]
    # if args.layers is not None:
    #     output_spaces = output_spaces[args.layers]
    #     funcs = funcs[args.layers]  # select just the requested layers

    dataset = serial.load(args.dataset)

    # figure, all_axes = pyplot.subplots(2, len(funcs) + 1, figsize=(16, 12))
    figure = pyplot.gcf()
    grid_shape = (2, len(funcs) + 1)
    all_axes = numpy.zeros(grid_shape, dtype=object)
    all_axes.flat[:] = None
    # all_axes = [[None, ] * grid_shape[1], ] * grid_shape[0]
    for c_1, layer in enumerate(model.layers):
        c = c_1 + 1
        if isinstance(layer.get_output_space(), Conv2DSpace):
            all_axes[0, c] = pyplot.subplot2grid(grid_shape, (0, c))

        all_axes[1, c] = pyplot.subplot2grid(grid_shape, (1, c))

    # for i in range(1, len(funcs) + 1):
    #     all_axes[0][i] = pyplot.subplot2grid(grid_shape, (0, i))
    #     all_axes[1][i] = pyplot.subplot2grid(grid_shape, (1, i))

    all_axes[1][0] = pyplot.subplot2grid(grid_shape, (1, 0), rowspan=1)
    all_axes[1][-1] = pyplot.subplot2grid(grid_shape,
                                          (0, grid_shape[1] - 1),
                                          rowspan=2)

    # Hide tickmarks
    for axes_row in all_axes:
        for axes in axes_row:
            if axes is not None:
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)

    row_index = numpy.array(0)
    layer_index = numpy.array(0)
    feature_indices = numpy.zeros(len(funcs), dtype=int)

    def make_b01c_feature_map(features, layer):
        """
        Returns features as a feature map in B01C axis order.

        If features are already a feature map, this just reorders the axes.

        If features are a feature vector, this copies them into a square
        feature map.

        Softmax layers are a special case: output as a B01C "vector" (i.e. all
        dimensions except 1 are of size 1).
        """

        assert len(features.shape) in (2, 4)

        feature_space = layer.get_output_space()

        if isinstance(layer, Softmax):
            features = features.transpose()
            return features[numpy.newaxis, :, :, numpy.newaxis]
        if isinstance(feature_space, Conv2DSpace):
            return shuffle_axes(features, feature_space.axes, ('b', 0, 1, 'c'))
            # b01c = Conv2DSpace(shape=feature_space.shape,
            #                    num_channels=feature_space.num_channels,
            #                    axes=('b', 0, 1, 'c'),
            #                    dtype=feature_space.dtype)
            # return feature_space.np_format_as(features, b01c)
        elif isinstance(feature_space, VectorSpace):
            dim = int(numpy.sqrt(feature_space.dim))
            if dim ** 2 < feature_space.dim:
                dim += 1

            b01c = Conv2DSpace(shape=(dim, dim),
                               num_channels=1,
                               axes=('b', 0, 1, 'c'),
                               dtype=feature_space.dtype)
            padded_features = numpy.zeros((1, dim ** 2), dtype=features.dtype)
            padded_features[0, :features.size] = features
            padded_feature_space = VectorSpace(dim=padded_features.size,
                                               dtype=feature_space.dtype)
            return padded_feature_space.np_format_as(padded_features, b01c)
        else:
            raise TypeError("Don't know what to do with feature space of type "
                            "%s" % type(feature_space))

    def get_kernel_weights(layer, output_channel):
        '''
        Returns kernel weights corresponding to the first input channel.
        Axis 'c' corresponds to the output channel here.
        '''
        output_space = layer.get_output_space()
        if not isinstance(output_space, Conv2DSpace):
            return None

        weights = layer.get_params()[0].get_value()
        weights = shuffle_axes(weights,
                               output_space.axes,
                               ('c', 0, 1, 'b'))  # 'b' corresponds to output 'c'

        assert weights.shape[0] == layer.get_input_space().num_channels
        assert weights.shape[-1] == layer.get_output_space().num_channels

        return weights[0, :, :, output_channel]

    def show_images():
        assert dataset.view_converter.axes == ('b', 0, 1, 'c')

        batch = dataset.X[row_index:row_index + 1, :]
        topo_batch = dataset.get_topological_view(batch)
        # pdb.set_trace()
        topo_space = Conv2DSpace(shape=topo_batch.shape[1:3],
                                 num_channels=topo_batch.shape[3],
                                 axes=dataset.view_converter.axes,
                                 dtype=dataset.X.dtype)
        input_batch = topo_space.np_format_as(topo_batch,
                                              model.get_input_space())

        feature_maps = [f(input_batch) for f in funcs]

        # convert all feature maps to B01C axis order, convert feature
        # vectors to a B01C "feature map" just by copying them into
        # a 4-D tensor.
        # pdb.set_trace()
        feature_maps = [make_b01c_feature_map(feature_map, layer)
                        for feature_map, layer
                        in safe_zip(feature_maps, model.layers)]

        # Select just the features selected for drawing
        feature_maps = [feature_map[..., c:c + 1]
                        for feature_map, c
                        in safe_zip(feature_maps, feature_indices)]

        kernel_weights = [get_kernel_weights(layer, c)
                          for layer, c
                          in safe_zip(model.layers, feature_indices)]

        # feature_colormaps = ['gray', ] + [None, ] * len(funcs)

        # feature_maps = [topo_batch, ] + feature_maps
        # kernel_weights = [None, ] + kernel_weights
        # layers = [None, ] + list(model.layers)

        for (feature_map,
             feature_colormap,
             kernel_weight,
             layer,
             kernel_axes,
             feature_axes) in safe_zip([topo_batch, ] + feature_maps,
                                       ['gray', ] + [None, ] * len(funcs),
                                       [None, ] + kernel_weights,
                                       [None, ] + list(model.layers),
                                       all_axes[0, :],
                                       all_axes[1, :]):
            # show the kernel
            # if kernel_weight is None:
            #     kernel_axes.set_visible(False)
            # else:
            if kernel_weight is not None:
                kernel_axes.imshow(kernel_weight,
                                   cmap='gray',
                                   interpolation='nearest')

            # pdb.set_trace()
            feature_axes.imshow(feature_map[0, :, :, 0],
                                cmap=feature_colormap,
                                interpolation='nearest')

        figure.canvas.draw()

    def on_key_press(event):
        '''
        Pgup/Pgdn: change image
        left/right: change layer
        up/down: change feature within current layer
        '''

        def increment_index(index, size, increment):
            assert size > 0
            assert index.size == 1
            assert numpy.all(index >= 0)
            assert numpy.all(index <= size)

            index[...] += increment
            index[...] = index[...] % size  # this actually works for negative index


        print("key string: '%s'" % event.key)

        if event.key == 'q':
            sys.exit(0)
        elif event.key == 'pageup':
            increment_index(row_index, dataset.X.shape[0], -1)
        elif event.key == 'pagedown':
            increment_index(row_index, dataset.X.shape[0], 1)
        elif event.key == 'right':
            increment_index(layer_index, len(model.layers), 1)
        elif event.key == 'left':
            increment_index(layer_index, len(model.layers), -1)
        elif event.key == 'up' or event.key == 'down':
            output_space = model.layers[layer_index].get_output_space()
            if isinstance(output_space, Conv2DSpace):
                num_features = output_space.num_channels
            else:
                num_features = output_space.dim

            increment_index(feature_indices[layer_index:layer_index + 1],
                            num_features,
                            -1 if event.key == 'down' else 1)
        else:
            return  # do nothing. Don't even redraw.

        show_images()

    figure.canvas.mpl_connect('key_press_event', on_key_press)
    show_images()
    pyplot.show()

if __name__ == '__main__':
    main()
