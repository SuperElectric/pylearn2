#! /usr/bin/env python

import argparse
import os.path
import numpy
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)
from pylearn2.datasets.preprocessing import CentralWindow


def crop_rows(old_view_converter, old_rows, cropper):
    tmp_dataset = DenseDesignMatrix(X=old_rows,
                                    view_converter=old_view_converter)
    cropper.apply(tmp_dataset)
    return tmp_dataset.X


def main():

    def parse_args():
        parser = argparse.ArgumentParser(description=("crops big norb instance"
                                                      " datasets"))

        parser.add_argument('--input',
                            '-i',
                            required=True,
                            help=("The .pkl file of a big NORB instance "
                                  "dataset."))

        parser.add_argument('--batch_size',
                            '-b',
                            type=int,
                            default=1000)

        return parser.parse_args()

    args = parse_args()

    def get_paths(input_path, cropped_shape):
        assert cropped_shape[0] == cropped_shape[1]
        assert len(cropped_shape) == 2
        output_dir, input_filename = os.path.split(input_path)
        input_filename_base = os.path.splitext(input_filename)[0]
        preprocessor, test_or_train = input_filename_base.split('_')
        assert test_or_train in ('test', 'train')
        preprocessor += ("-cropped%d" % cropped_shape[0])
        output_filename = '%s_%s.pkl' % (preprocessor,
                                         test_or_train)
        output_images_filename = '%s_%s_images.pkl' % (preprocessor,
                                                       test_or_train)
        return tuple(os.path.join(output_dir, f)
                     for f in (output_filename, output_images_filename))

    dataset = serial.load(args.input)

    cropped_shape = (96, 96)

    dataset_path, image_memmap_path = get_paths(args.input,
                                                cropped_shape)

    new_X = numpy.memmap(filename=image_memmap_path,
                         dtype=dataset.X.dtype,
                         shape=(dataset.X.shape[0], numpy.prod(cropped_shape)),
                         mode='w+')

    cropper = CentralWindow(cropped_shape)

    num_samples = dataset.X.shape[0]
    num_batches = (num_samples // args.batch_size +
                   (0 if num_samples % args.batch_size == 0 else 1))

    # print ("num_samples, batch_size, num_batches: %d %d %d" %
    #        (num_samples, args.batch_size, num_batches))
    for b in range(num_batches):
        first_row = b * args.batch_size
        end_row = min(first_row + args.batch_size, num_samples)
        selector = numpy.s_[first_row:end_row, :]
        
        new_X[selector] = crop_rows(dataset.view_converter,
                                    dataset.X[selector],
                                    cropper)

        print "cropped %2.1f %%" % (100. * float(b) / num_batches)

    dataset.X = new_X
    dataset.view_converter = DefaultViewConverter(cropped_shape + (1, ),
                                                  dataset.view_converter.axes)

    serial.save(dataset_path, dataset)


if __name__ == '__main__':
    main()
