#! /usr/bin/env python
"""
Video demo. Press 'q' to quit.

Code copied over from OpenCV's tutorial at:
http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video
"""

import sys, argparse
import numpy, matplotlib, cv2, theano
from matplotlib import pyplot
from pylearn2.utils import serial, safe_zip
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.models.mlp import MLP
from pylearn2.datasets import new_norb

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

    parser.add_argument("-s",
                        "--scale",
                        default=2,
                        type=int,
                        help="Detection window scale.")
    result = parser.parse_args()

    if result.model is not None and not result.model.endswith('.pkl'):
        print("Expected --model to end with '.pkl', but got %s." %
              args.machine)
        print("Exiting.")
        sys.exit(1)

    return result


class MatplotlibDisplay(object):
    def __init__(self):
        pyplot.ion()
        self.figure = pyplot.figure()
        self.axes = self.figure.add_subplot(111)

    def update(self, video_frame):
        # reverses channels from BGR to RGB
        self.axes.imshow(video_frame[..., ::-1],  
                        interpolation='nearest',
                        norm=matplotlib.colors.NoNorm())
        self.figure.canvas.draw()
        

class VideoDisplay(object):
    def __init__(self):
        pass

    def update(self, video_frame):
        cv2.imshow('video', video_frame)


class ClassifierDisplay(object):

    def __init__(self, model_path, object_scale):

        def load_model_function(model_path):
            model = serial.load(model_path)

            # This is just a sanity check. It's not necessarily true; it's just
            # expected to be true in the current use case.
            assert isinstance(model, MLP)

            input_space = model.input_space
            floatX = theano.config.floatX
            input_symbol = input_space.make_theano_batch(name='features',
                                                         dtype=floatX,
                                                         batch_size=1)
            output_symbol = model.fprop(input_symbol)
            return theano.function([input_symbol, ], output_symbol)

            # specs = (model.input_space, 'features')
            # specs = (CompositeSpace((model.input_space,
            #                          VectorSpace(dim=test_set.y.shape[1]))),
            #          ('features', 'targets'))

        self.model_function = load_model_function(model_path)

        def get_example_images():
            small_norb = new_norb.NORB(which_norb='small', which_set='both')

            result = [None, ] * 50

            # Select image examples whose labels end in this
            label_end = numpy.zeros(3, dtype='int32')

            labels = small_norb.y
            images = small_norb.get_topological_view(single_tensor=False)[0]

            for row_image, label in safe_zip(images, labels):
                if (label[-3:] == label_end).all():
                    object_id = label[0] * 10 + label[1]
                    assert result[object_id] is None
                    result[object_id] = row_image.reshape([96, 96])

            assert not any(example is None for example in result)

            return result

        self.example_images = get_example_images()

        self.margin = 10  # space between images in self.status_pixels
        self.object_scale = object_scale # scale object detection region size by this much

        # TODO: replace this by reading the input space from the model.
        self.norb_image_shape = numpy.asarray((96, 96) 
                                              if model_path.find('small')
                                              else (106, 106))

        self.all_pixels = None  # set by _init_pixels
        
    def _init_pixels(self, video_frame):
        self.all_pixels = numpy.zeros(shape=(video_frame.shape[0], 
                                             video_frame.shape[1]*2,
                                             3),
                                      dtype=video_frame.dtype)
 
        self.video_pixels = self.all_pixels[:, :video_frame.shape[1], :]
        self.status_pixels = self.all_pixels[:, video_frame.shape[1]:, :]


        object_shape = self.norb_image_shape * self.object_scale
        self.object_min_corner = (numpy.asarray(self.video_pixels.shape[:2]) -
                                  object_shape) / 2
        self.object_max_corner = self.object_min_corner + object_shape

        def get_object_pixels(all_pixels, min_corner, max_corner):
            return all_pixels[min_corner[0]:max_corner[0],
                              min_corner[1]:max_corner[1],
                              :]

        self.object_pixels = get_object_pixels(self.all_pixels, 
                                               self.object_min_corner,
                                               self.object_max_corner)
        
        def get_model_input_pixels(status_pixels, shape, margin):
            return status_pixels[margin:margin+shape[0],
                                 margin:margin+shape[1],
                                 :]

        self.model_input_pixels = get_model_input_pixels(self.status_pixels,
                                                         self.norb_image_shape,
                                                         self.margin)




    def update(self, video_frame):
        if self.all_pixels is None:
            self._init_pixels(video_frame)

        self.video_pixels[...] = video_frame

        # greyify object detection region.
        gray_object = cv2.cvtColor(self.object_pixels, cv2.COLOR_BGR2GRAY)
        self.object_pixels[...] = gray_object[..., numpy.newaxis]

        # Draw black border around object detection region.
        cv2.rectangle(self.video_pixels, 
                      tuple(reversed(self.object_min_corner)), 
                      tuple(reversed(self.object_max_corner)),
                      (0, 0, 0),  # color
                      2)  # thickness

        model_input = cv2.resize(gray_object, tuple(self.norb_image_shape))

        # TODO: preprocess model_input!
        self.model_input_pixels[...] = model_input[..., numpy.newaxis]

        model_input = model_input[numpy.newaxis, :, :, numpy.newaxis]
        assert model_input.shape == (1, 96, 96, 1), str(model_input)
        softmax_vector = self.model_function(model_input).flatten()

        # Print 5 most likely categories, probabilities
        # TODO: display their examples, probabilities
        sorted_ids = numpy.argsort(softmax_vector)[-1:-5:-1]
        message = "Category/probability: "
        for sorted_id in sorted_ids:
            category = new_norb.get_category_value(sorted_id / 10)
            instance = sorted_id % 10
            probability = softmax_vector[sorted_id]
            message = message + "%s%d/%0.2f " % (category, 
                                                 instance,
                                                 probability)

        print message
        cv2.imshow('classifier', self.all_pixels)


def main():

    # turns on interactive mode, which enables the draw() function
    pyplot.ion()

    args = parse_args()


    cap = cv2.VideoCapture(0)

    keep_going = True

    print("Press 'q' to quit.")

    displays = []
    displays.append(VideoDisplay() if args.model is None
                    else ClassifierDisplay(args.model, args.scale))

    if args.matplotlib:
        displays.append(MatplotlibDisplay())

    while(keep_going):
        # Capture frame-by-frame
        keep_going, video_frame = cap.read()

        if not keep_going:
            print "Error in reading a frame. Exiting..."
        else:

            assert len(video_frame.shape) == 3
            assert video_frame.shape[-1] == 3

            for display in displays:
                display.update(video_frame)

            # Checks whether user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_going = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
