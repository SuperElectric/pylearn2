#! /usr/bin/env python
"""
Video demo. Press 'q' to quit.

Code copied over from OpenCV's tutorial at:
http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
"""

import sys, argparse
import numpy, matplotlib, cv2, theano
from matplotlib import pyplot
from pylearn2.utils import serial
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.models.mlp import MLP


def main():

    # turns on interactive mode, which enables the draw() function
    pyplot.ion()

    def parse_args():
        parser = argparse.ArgumentParser(
            description=("Shows video input, and optionally classifies "
                         "frames using a loaded model"))

        parser.add_argument("-m",
                            "--model",
                            default=None,
                            help=".pkl file of the trained model")

        parser.add_argument("--matplotlib",
                            default=False,
                            action='store_true',
                            help=("Additionally display a color image using "
                                  "matplotlib, to sanity-check the pixel "
                                  "layout"))

        result = parser.parse_args()

        if result.model is not None and not result.model.endswith('.pkl'):
            print("Expected --model to end with '.pkl', but got %s." %
                  args.machine)
            print("Exiting.")
            sys.exit(1)

        return result

    args = parse_args()

    def get_model_and_specs(args):
        if args.model is None:
            return None, None
        else:
            model = serial.load(args.model)

            # This is just a sanity check. It's not necessarily true; it's just
            # expected to be true in the current use case.
            assert isinstance(model, MLP)

            specs = (model.input_space, 'features')
            # specs = (CompositeSpace((model.input_space,
            #                          VectorSpace(dim=test_set.y.shape[1]))),
            #          ('features', 'targets'))

            return model, specs

    model, data_specs = get_model_and_specs(args)

    if model is not None:
        norb = NORB(which_norb='small')
        object_to_example = [None, ] * 50

        # Select image examples whose labels end in this
        label_end = numpy.zeros(3, dtype='int32')

        for row_image, label in safe_zip(norb.X, norb.y):
            if (label[-3:] == label_end).all():
                object_id = label[0] * 10 + label[1]
                assert object_to_example[object_id] is None
                object_to_example[object_id] = row_image.reshape([96, 96])

        assert not any(example is None for example in object_to_example)

    def get_model_function(model):
        """
        Returns an evaluatable function that numerically performs a model's
        fprop method on a batch of size 1.
        """
        input_space = model.input_space
        floatX = theano.config.floatX
        input_symbol = input_space.make_theano_batch(name='features',
                                                     dtype=floatX,
                                                     batch_size=1)
        output_symbol = model.fprop(input_symbol)
        return theano.function([input_symbol, ], output_symbol)

    model_function = (None if model is None
                      else get_model_function(model, batch_size))

    cap = cv2.VideoCapture(0)

    keep_going = True

    print("Press 'q' to quit.")

    if args.matplotlib:
        figure = pyplot.figure()
        axes = figure.add_subplot(111)

    while(keep_going):
        # Capture frame-by-frame
        keep_going, frame = cap.read()
        assert len(frame.shape) == 3
        assert frame.shape[-1] == 3

        if not keep_going:
            print "Error in reading a frame. Exiting..."
        else:
            # Our operations on the frame come here
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            assert len(gray_image.shape) == 2

            # Displays the resulting frame
            cv2.imshow('frame', gray_image)

            if args.matplotlib:
                axes.imshow(frame[..., ::-1],
                            interpolation='nearest',
                            norm=matplotlib.colors.no_norm())
                figure.canvas.draw()

            # Adds a singleton batch axis
            gray_image = gray_image[numpy.newaxis, ...]
            if model_function is not None:
                softmax_vector = model_function(gray_image)
                object_id = numpy.argmax(softmax_vector, axis=1)[0]
                example = object_to_example[object_id]
                # category = int(object_id / 10)
                # instance = object_id % 10

                # TODO: display an example of category, instance from small NORB

            # Checks whether user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_going = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
