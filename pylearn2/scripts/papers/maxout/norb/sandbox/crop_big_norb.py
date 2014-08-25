#! /usr/bin/env python

import os, time, resource, argparse, numpy
from pylearn2.utils.string_utils import preprocess
from pylearn2.scripts.papers.maxout.norb import (load_norb_instance_dataset,
                                                 human_readable_memory_size)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=("Prints the memory usage and load time for the "
                     "big NORB instance dataset, when using cropping "
                     "vs non-cropping."))

    parser.add_argument('-c',
                        '--crop',
                        choices=('preprocess', 'crop', 'none'),
                        help="Crop via preprocessor, crop 'manually', "
                        "or don't crop")

    return parser.parse_args()


def main():
    args = _parse_args()
    dataset_path = preprocess("${PYLEARN2_DATA_PATH}/norb/"
                              "instance_recognition/"
                              "left_02_01/raw_train.pkl")

    print "loading %s" % os.path.split(dataset_path)[1]
    start_time = time.time()

    # crop_shape = (77, 77)
    crop_shape = (106, 106)

    if args.crop == 'preprocess':
        print "cropping dataset with preprocessor"
        dataset = load_norb_instance_dataset(dataset_path,
                                             label_format="norb",
                                             crop_shape=crop_shape)
    else:
        dataset = load_norb_instance_dataset(dataset_path,
                                             label_format="norb")
        if args.crop == 'crop':
            print "cropping dataset 'manually'"
            print "getting topo view"
            topo_X = dataset.get_topological_view()

            print "cropping topo view"
            start = (numpy.asarray(topo_X.shape[1:-1]) -
                     numpy.asarray(crop_shape)) // 2
            end = start + crop_shape
            topo_X = topo_X[:, start[0]:end[0], start[1]:end[1], :]
            assert topo_X.shape[1:-1] == crop_shape, \
                ("topo_X.shape: %s, crop_shape: %s" %
                 (str(topo_X.shape), str(crop_shape)))

            print "setting topo"
            dataset.set_topological_view(topo_X)
        else:
            print "not cropping dataset"

    duration = time.time() - start_time

    print "Loading dataset took %g secs" % duration
    memory_kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print ("Total memory usage: %s" %
           human_readable_memory_size(memory_kB * 1024))

    dataset_nbytes = dataset.X.nbytes + dataset.y.nbytes
    print ("dataset memory footprint: %s" %
           human_readable_memory_size(dataset_nbytes))

    print("dataset.X's type is %s" % type(dataset.X))

if __name__ == '__main__':
    main()
