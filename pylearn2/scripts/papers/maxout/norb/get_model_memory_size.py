#! /usr/env python

import numpy
from pylearn2.utils import safe_zip
from pylearn2.models import MLP


def get_conv_output_shape(input_shape,
                          kernel_shape,
                          pool_shape,
                          pool_stride,
                          pad):

    def canonicalize_and_check_args(input_shape,
                                    kernel_shape,
                                    pool_shape,
                                    pool_stride,
                                    pad):
        """
        Take all args and safely turn them into size-2 vectors of int32s.
        """
        if numpy.isscalar(pad):
            pad = [pad, pad]

        # turn all inputs into size-2 vectors of ints
        shapes_2d = (input_shape, kernel_shape, pool_shape, pool_stride, pad)
        names = ("input_shape",
                 "kernel_shape",
                 "pool_shape",
                 "pool_stride",
                 "pad")

        for shape_2d, name in safe_zip(shapes_2d, names):
            if len(shape_2d) != 2:
                raise ValueError("Expected %s to be of length 2, but got %s" %
                                 (name, str(shape_2d)))

            for elem in shape_2d:
                if not numpy.issubdtype(numpy.dtype(type(shape_2d)), 'int32'):
                    raise ValueError("Expected %s's elements to have integral "
                                     "type, but got %s." %
                                     (name, str(shape_2d)))

        return tuple(numpy.asarray(s, dtype='int32') for s in shapes_2d)

    (input_shape,
     kernel_shape,
     pool_shape,
     pool_stride,
     pad) = canonicalize_and_check_args(input_shape,
                                        kernel_shape,
                                        pool_shape,
                                        pool_stride,
                                        pad)
    detector_shape = input_shape + pad*2 - kernel_shape + 1  # aka conv. output
    pooled_shape = (detector_shape - pool_shape) / pool_stride


def main():
    args = parse_args()

    model = serial.load(args.model)

    assert isinstance(model, MLP)

    for layer in model.layers:
        output_shape, memory_shape = get_output_and_memory_shape(layer,
                                                                 model.batch_size)

if __name__ == '__main__':
    main()


# Input shape: (108, 108)
# Detector space: (109, 109)
# Output space: (16, 16)
# Input shape: (16, 16)
# Detector space: (17, 17)
# Output space: (4, 4)
# Input shape: (4, 4)
# Detector space: (6, 6)
# Output space: (2, 2)
