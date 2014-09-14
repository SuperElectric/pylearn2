#! /usr/bin/env python

"""
Steps through a dataset's images, showing the feature maps of selected levels.
"""


import argparse, sys
from matplotlib import pyplot
from pylearn2.utils import serial


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

    parser.add_argument("-l",
                        "--layers",
                        default=None,
                        nargs='+',
                        help=("Show outputs of these layer numbers "
                              "(0-indexed, counting from bottom."))

    result = parser.parse_args()


def main():
    args = parse_args()

    model = serial.load(args.model)

    def get_funcs(model):
        def get_var_from_space(space):
            batch = space.get_origin_batch(batch_size=1)
            return utils.sharedX(batch)

        input_var = get_var_from_space(model.get_input_space())
        output_vars = model.fprop(input_var, return_all=True)

        result = [theano.function([input_var], y for y in output_vars)]

    output_spaces = [x.get_output_space() for x in model.layers]
    funcs = get_funcs(model)
    if args.layers is not None:
        output_spaces = output_spaces[args.layers]
        funcs = funcs[args.layers]  # select just the requested layers

    dataset = serial.load(args.dataset)

    figure, all_axes = pyplot.subplots(2, len(funcs), figsize=(16, 12))

    row_index = 0

    def increment_index(increment):
        row_index += increment
        row_index = row_index % dataset.X.shape[0]

    def make_feature_map(features):
        """
        Does nothing to feature maps. Copies feature vectors into square-shaped
        matrices.
        """
        assert len(features.shape) in (2, 3)

        if len(features.shape) == 3:
            return features

        assert features.shape[0] == 1

        num_dims = features.shape[1]
        side_length = numpy.sqrt(num_dims)
        if side_length - int(side_length) > 0.:
            side_length = int(side_length) + 1
        else:
            side_length = int(side_length)

        result = numpy.zeros((side_length, side_length), dtype=features.dtype)
        result.flat[:features.shape[0]] = features.flat

    def show_images():
        batch = dataset.X[row_index:row_index + 1, :]
        topo_batch = dataset.view_converter.get_topo_view(batch)
        feature_maps = [f(topo_batch) for f in funcs]

        feature_maps = [make_feature_map(x) for x in feature_maps]

        all_axes[0][0].imshow(topo_batch[:, :, 0])

        for feature_map, axes in safe_zip(feature_maps, all_axes[1]):
            axes.imshow(feature_map[:, :, 0])

        figure.canvas.draw()

    def on_key_press(event):
        if event.key == 'q':
            sys.exit(0)
        elif event.key == 'left':
            increment_index(-1)
            show_images()
        elif event.key == 'right':
            increment_index(1)
            show_images()

    figure.canvas.mpl_connect('key_press_event', on_key_press)

if __name__ == '__main__':
    main()
