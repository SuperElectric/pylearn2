#! /usr/bin/env python
"""
Video demo. Press 'q' to quit.

Code copied over from OpenCV's tutorial at:
http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video  # nopep8
"""

from __future__ import print_function
import sys, argparse, os, shutil, time
import numpy, matplotlib, cv2, theano
from matplotlib import pyplot
from pylearn2.utils import serial, safe_zip
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace
from pylearn2.models.mlp import MLP
from pylearn2.datasets import new_norb
from pylearn2.datasets.preprocessing import ZCA, LeCunLCN
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.datasets.dense_design_matrix import (DenseDesignMatrix,
                                                   DefaultViewConverter)
from pylearn2.scripts.papers.maxout.norb.resize_input_of_model import \
    resize_mlp_input


def parse_args():
    """
    Parses command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=("Shows video input, and optionally classifies "
                     "frames using a loaded model"))

    parser.add_argument("-m",
                        "--model",
                        default=None,
                        help=".pkl file of the trained model")

    parser.add_argument("-p",
                        "--preprocessor",
                        default=None,
                        help=("The .pkl file of the data preprocessor. "
                              "If omitted, this will assume that the --model "
                              "argument's preprocessor prefix (e.g. gcn-lcn7 "
                              "is sufficient to specify the preprocessor."))

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

    parser.add_argument("--localize",
                        action='store_true',
                        default=False,
                        help="Perform localization.")

    result = parser.parse_args()

    if (result.model is None) != (result.preprocessor is None):
        print("Must provide --model and --preprocessor, or neither.")
        sys.exit(1)

    # if result.model is not None and not result.model.endswith('.pkl'):
    #     print("Expected --model to end with '.pkl', but got %s." %
    #           args.model)
    #     print("Exiting.")
    #     sys.exit(1)

    # if result.preprocessor is not None \
    #    and not result.preprocessor.endswith('.pkl'):
    #     print("Expected --preprocessor to end with '.pkl', but got %s." %
    #           result.preprocessor)
    #     print("Exiting.")
    #     sys.exit(1)

    return result


class MatplotlibDisplay(object):
    def __init__(self):
        pyplot.ion()
        self.figure = pyplot.figure()
        self.axes = self.figure.add_subplot(111)

    def update(self, video_frame):
        assert len(video_frame.shape) == 3
        assert video_frame.shape[-1] == 3

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

    def _load_model_function(self, model_path):
        model = serial.load(model_path)

        # This is just a sanity check. It's not necessarily true; it's just
        # expected to be true in the current use case.
        assert isinstance(model, MLP)

        input_space = model.get_input_space()
        floatX = theano.config.floatX
        input_symbol = input_space.make_theano_batch(name='features',
                                                     dtype=floatX,
                                                     batch_size=1)
        output_symbol = model.fprop(input_symbol)
        return theano.function([input_symbol, ], output_symbol), input_space

        # specs = (model.input_space, 'features')
        # specs = (CompositeSpace((model.input_space,
        #                          VectorSpace(dim=test_set.y.shape[1]))),
        #          ('features', 'targets'))


    def __init__(self, model_path, preprocessor, object_scale):

        # # debug stuff
        # pyplot.ion()
        # self.figure = pyplot.figure()
        # self.axes = self.figure.add_subplot(111)

        self.output_dir = '/tmp/video_demo/'

        if os.path.isdir(self.output_dir):
            shutil.rmtree(self.output_dir)

        os.mkdir(self.output_dir)

        self.frame_number = 0

        self.model_function, input_space = self._load_model_function(model_path)

        def get_preprocessor(preprocessor, image_shape=None):
            if preprocessor.endswith('.pkl'):
                return serial.load(preprocessor)
            elif preprocessor.startswith('lcn7'):
                assert image_shape is not None
                return LeCunLCN(img_shape=image_shape, kernel_size=7)
            else:
                raise ValueError('unrecognized value "%s" for --preprocessor' %
                                 preprocessor)

        self.preprocessor = get_preprocessor(preprocessor, input_space.shape)

        def get_example_images():
            small_norb = new_norb.NORB(which_norb='small', which_set='both')

            result = [None, ] * 51

            # Select image examples whose labels end in [0, 0, 0]
            label_end = numpy.zeros(3, dtype='int32')

            labels = small_norb.y
            images = small_norb.get_topological_view(single_tensor=False)[0]

            for row_image, label in safe_zip(images, labels):
                if (label[-3:] == label_end).all():
                    object_id = label[0] * 10 + label[1]
                    assert result[object_id] is None
                    result[object_id] = row_image.reshape([96, 96])

            assert not any(example is None for example in result[:-1])
            assert result[-1] is None

            # Add an example of 'background'
            result[-1] = result[-2].copy()
            result[-1].flat[...] = 1.0

            return result

        self.example_images = get_example_images()

        self.margin = 10  # space between images in self.status_pixels
        self.object_scale = object_scale  # scales object detection region size

        # self.norb_image_shape = input_space.shape
        # assert len(self.norb_image_shape) == 2
        # TODO: replace this by reading the input space from the model
        self.norb_image_shape = numpy.asarray((96, 96)
                                              if model_path.find('small')
                                              else (106, 106))

        self.all_pixels = None  # set by _init_pixels
        self.model_input_dataset = None

    def _get_object_shape(self):
        return self.norb_image_shape * self.object_scale

    def _init_pixels(self, video_frame):
        self.all_pixels = numpy.zeros(shape=(video_frame.shape[0],
                                             video_frame.shape[1] * 2,
                                             3),
                                      dtype=video_frame.dtype)

        self.video_pixels = self.all_pixels[:, :video_frame.shape[1], :]
        self.status_pixels = self.all_pixels[:, video_frame.shape[1]:, :]

        object_shape = self._get_object_shape()
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

        def get_grid_window_pixels(row, column):
            """
            Returns a window of self.status_pixels.

            The window is chosen using grid coordinates of a grid layout.

            Assuming that the status_pixels are to be broken up into grid
            squares, each big enough for a norb image, and separated from
            each other by self.margin, this returns the image pixels for
            the grid window corresponding to the given grid coordinates.
            """
            image_shape = numpy.asarray(self.norb_image_shape)
            grid_square_shape = image_shape + self.margin
            min_corner = (self.margin +
                          grid_square_shape * numpy.asarray((row, column)))
            max_corner = min_corner + image_shape
            return self.status_pixels[min_corner[0]:max_corner[0],
                                      min_corner[1]:max_corner[1],
                                      :]

        def get_model_input_pixels(status_pixels, shape, margin):
            return status_pixels[margin:(margin + shape[0]),
                                 margin:(margin + shape[1]),
                                 :]

        self.model_input_pixels = get_grid_window_pixels(0, 0)

        num_candidates = 4

        self.candidate_pixels_list = tuple(get_grid_window_pixels(1, c)
                                           for c in range(num_candidates))

        self.probability_pixels_list = tuple(get_grid_window_pixels(2, c)
                                             for c in range(num_candidates))

    def _preprocess(self, model_input):
        image_shape = model_input.shape

        assert str(model_input.dtype) == 'uint8'

        # flatten into a design matrix
        model_input = numpy.asarray(model_input.reshape(1, -1),
                                    dtype=theano.config.floatX)
        model_input /= 255.0

        # if isinstance(self.preprocessor, ZCA):
        # TODO: in create_instance_dataset.py, use the GCN preprocessor
        # class rather than the GCN funtion as below, and save it along
        # with the ZCA preprocessor. This will require you to add an
        # in-place option to the GCN preprocesssor class.
        global_contrast_normalize(model_input, scale=55.0, in_place=True)

        if self.model_input_dataset is None:
            view_converter = DefaultViewConverter(image_shape + (1,),
                                                  axes=('b', 0, 1, 'c'))
            self.model_input_dataset = \
                DenseDesignMatrix(X=model_input,
                                  view_converter=view_converter)
        else:
            self.model_input_dataset.X = model_input

        self.preprocessor.apply(self.model_input_dataset, can_fit=False)
        return self.model_input_dataset.X.reshape(image_shape)

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
        model_input = self._preprocess(model_input)
        model_input = model_input[numpy.newaxis, :, :, numpy.newaxis]
        assert model_input.shape == (1, 96, 96, 1), str(model_input)

        softmax_vector = self.model_function(model_input).flatten()

        # Print 5 most likely categories, probabilities
        # TODO: display their examples, probabilities
        num_candidates = len(self.candidate_pixels_list)
        sorted_ids = numpy.argsort(softmax_vector)[-1:-(num_candidates + 1):-1]
        message = "Category/probability: "
        for (sorted_id,
             candidate_pixels,
             probability_pixels) in safe_zip(sorted_ids,
                                             self.candidate_pixels_list,
                                             self.probability_pixels_list):
            category = new_norb.get_category_value(sorted_id / 10)
            instance = sorted_id % 10
            probability = softmax_vector[sorted_id]

            example_image = self.example_images[sorted_id]
            candidate_pixels[...] = example_image[:, :, numpy.newaxis]

            # Draws probabilities
            # Reference: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
            probability_pixels.fill(0)
            cv2.putText(probability_pixels,
                        "%.02f" % probability,
                        (15, 40),  # position x, y (+x = right, +y = down)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 255, 255), # color
                        2,  # thickness
                        cv2.CV_AA)  # line type

        def get_visible_model_input(model_input):
            assert str(model_input.dtype) == theano.config.floatX
            assert len(model_input.shape) == 4
            assert model_input.shape[0] == 1
            assert model_input.shape[-1] == 1
            min_pixel = model_input.min()
            max_pixel = model_input.max()
            result = model_input - min_pixel
            result *= (255 / (max_pixel - min_pixel))

            assert result.max() <= 255.0001, "result.max() = %d" % result.max()
            assert result.min() >= 0.0, "result.min() = %d" % result.min()
            # print "min, max: %g %g \trescaled: %g %g" % (min_pixel,
            #                                              max_pixel,
            #                                              result.min(),
            #                                              result.max())

            result = numpy.asarray(result, dtype='uint8')

            return result

        self.model_input_pixels[...] = get_visible_model_input(model_input)

        # self.axes.imshow(model_input[0, :, :, 0],
        #                  interpolation='nearest',
        #                  cmap='gray')
        # self.figure.canvas.draw()

        cv2.imshow('classifier', self.all_pixels)

        image_path = os.path.join(self.output_dir,
                                  "frame_%05d.png" % self.frame_number)
        cv2.imwrite(image_path, self.all_pixels)
        self.frame_number += 1


class LocalizerDisplay(ClassifierDisplay):

    def __init__(self,
                 model_path,
                 preprocessor,
                 object_scale,
                 video_frame_shape):
        self.video_frame_shape = video_frame_shape
        super(LocalizerDisplay, self).__init__(model_path,
                                               preprocessor,
                                               object_scale)

    def _load_model_function(self, model_path):
        model = serial.load(model_path)
        assert isinstance(model, MLP)

        input_dim = min(self.video_frame_shape[:2])  # TODO: switch to max?
        model = resize_mlp_input(model, (input_dim, input_dim))

        # DEBUG
        for i, layer in enumerate(model.layers):
            print("layer %d type: %s # output channels: %d" %
                  (i,
                   type(layer),
                   layer.get_output_space().num_channels))

        input_space = model.get_input_space()
        floatX = theano.config.floatX
        input_symbol = input_space.make_theano_batch(name='features',
                                                     dtype=floatX,
                                                     batch_size=1)
        output_symbol = model.fprop(input_symbol)
        function = theano.function([input_symbol, ], output_symbol)
        return function, input_space

    def _get_object_shape(self):
        object_width = min(self.video_pixels.shape[:2])
        return numpy.asarray((object_width, object_width), dtype='int')

    def _init_pixels(self, video_frame):
        super(LocalizerDisplay, self)._init_pixels(video_frame)
        #del self.model_input_pixels

    def get_model_scale(self, model):
        convolutional_layers = [x for x in model.layers
                                if isinstance(x.get_output_space(),
                                              Conv2DSpace)]
        scales = numpy.asarray([get_scale(x) for x in model.layers])
        scale = numpy.prod(scales, axis=0)
        assert scale[0] == scale[1]
        return scale

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

        scaled_object_shape = self._get_object_shape // self.object_scale
        scaled_object_shape.append
        assert len(scaled_object_shape) == 3
        assert scaled_object_shape.dtype == numpy.dtype('int')
        assert (scaled_object_shape > 0).all()

        model_input = cv2.resize(gray_object, tuple(scaled_object_shape))
        model_input = self._preprocess(model_input)
        model_input = model_input[numpy.newaxis, :, :, numpy.newaxis]

        softmax_map = self.model_function(model_input)
        no_bkg_softmax_map = softmax_map[..., :50]
        no_bkg_max_location = numpy.unravel_index(no_bkg_softmax_map.argmax(),
                                                  no_bkg_softmax_map.shape)

        # only bother showing classification if it beats out the background.
        if no_bkg_max_location[-1] != 50:
            softmax_map_shape = numpy.asarray(softmax_map.shape[:2],
                                              dtype=float)

            centered_max_location = (no_bkg_max_location[:2] -
                                     (softmax_map_shape // 2))

            softmax_map_scale = get_model_scale(model)
            centered_max_location *= softmax_map_scale
            input_center = (self.object_min_corner +
                            self.object_max_corner) // 2

            # absolute coordinates of object center
            max_location = numpy.asarray(input_center + centered_max_location,
                                         dtype=int)

            # TODO: replace this with self.norb_image_shape, once you start
            # setting that using model.input_space
            norb_image_shape = numpy.asarray((96, 96), dtype=int) #self.norb_image_shape
            max_min_corner = max_location - norb_image_shape // 2
            max_max_corner = max_min_corner + norb_image_shape
            self.model_input_pixels[...] = \
                self.video_pixels[max_min_corner[0]:max_max_corner[0],
                                  max_min_corner[1]:max_max_corner[1]]

            # Draw a red square around the object bounding box.
            cv2.rectangle(self.video_pixels,
                          tuple(reversed(self.object_min_corner)),
                          tuple(reversed(self.object_max_corner)),
                          (255, 0, 0),  # color
                          2)  # thickness

            softmax_vector = softmax_map[no_bkg_max_location[0],
                                         no_bkg_max_location[1],
                                         :]

            # TODO: This whole chunk was copied from ClassifierWindow.
            #       Refactor to avoid this code duplication.
            num_candidates = len(self.candidate_pixels_list)
            sorted_ids = \
                numpy.argsort(softmax_vector)[-1:-(num_candidates + 1):-1]
            message = "Category/probability: "
            for (sorted_id,
                 candidate_pixels,
                 probability_pixels) in safe_zip(sorted_ids,
                                                 self.candidate_pixels_list,
                                                 self.probability_pixels_list):
                category = new_norb.get_category_value(sorted_id / 10)
                instance = sorted_id % 10
                probability = softmax_vector[sorted_id]

                example_image = self.example_images[sorted_id]
                candidate_pixels[...] = example_image[:, :, numpy.newaxis]

                # Draws probabilities
                # Reference: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
                probability_pixels.fill(0)
                cv2.putText(probability_pixels,
                            "%.02f" % probability,
                            (15, 40),  # position x, y (+x = right, +y = down)
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 255, 255), # color
                            2,  # thickness
                            cv2.CV_AA)  # line type

        cv2.imshow('classifier', self.all_pixels)

        image_path = os.path.join(self.output_dir,
                                  "frame_%05d.png" % self.frame_number)
        cv2.imwrite(image_path, self.all_pixels)
        self.frame_number += 1


def main():

    # turns on interactive mode, which enables the draw() function
    pyplot.ion()

    args = parse_args()

    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)

    # capture a video frame just to know its shape
    def get_video_shape(cap):
        keep_going, video_frame = cap.read()

        if not keep_going:
            print("Error in reading a frame. Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

        return video_frame.shape

    displays = []
    if args.model is None:
        displays.append(VideoDisplay())
    elif args.localize:
        displays.append(LocalizerDisplay(args.model,
                                         args.preprocessor,
                                         args.scale,
                                         get_video_shape(cap)))
    else:
        displays.append(ClassifierDisplay(args.model,
                                          args.preprocessor,
                                          args.scale))

    if args.matplotlib:
        displays.append(MatplotlibDisplay())

    while keep_going:
        # Capture frame-by-frame
        keep_going, video_frame = cap.read()

        if not keep_going:
            print("Error in reading a frame. Exiting...")
        else:

            assert len(video_frame.shape) == 3
            assert video_frame.shape[-1] == 3

            start_time = time.time()
            for display in displays:
                display.update(video_frame)

            duration = time.time() - start_time
            print("%g fps" % (1.0 / duration))

            # Checks whether user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_going = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
