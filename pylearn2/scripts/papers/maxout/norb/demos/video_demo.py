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

    args = parse_args()

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


    if args.model is not None:
        model_function = load_model_function(args.model)
        example_images = get_example_images()

    cap = cv2.VideoCapture(0)

    keep_going = True

    print("Press 'q' to quit.")

    if args.matplotlib:
        figure = pyplot.figure()
        axes = figure.add_subplot(111)

    all_pixels = None

    while(keep_going):
        # Capture frame-by-frame
        keep_going, video_frame = cap.read()

        if not keep_going:
            print "Error in reading a frame. Exiting..."
        else:

            assert len(video_frame.shape) == 3
            assert video_frame.shape[-1] == 3


            if all_pixels is None:
                all_pixels = numpy.zeros(shape=(video_frame.shape[0], 
                                                video_frame.shape[1]*2,
                                                3),
                                         dtype=video_frame.dtype)
            

            video_pixels = all_pixels[:, :video_frame.shape[1], :]
            video_pixels[...] = video_frame

            status_pixels = all_pixels[:, video_frame.shape[1]:, :]
            # all_pixels[:, :video_pixels.shape[1], :] = video_pixels


            # classifies gray_image
            if args.model is not None:
                norb_image_shape = numpy.asarray((96, 96) 
                                                 if args.model.find('small')
                                                 else (106, 106))
                object_shape = norb_image_shape * args.scale
                object_min_corner = (numpy.asarray(video_pixels.shape[:2]) - 
                                     object_shape) / 2
                object_max_corner = object_min_corner + object_shape
                object_pixels = all_pixels[object_min_corner[0]:object_max_corner[0],
                                           object_min_corner[1]:object_max_corner[1],
                                           :]

                gray_object = cv2.cvtColor(object_pixels, cv2.COLOR_BGR2GRAY)
                object_pixels[...] = gray_object[..., numpy.newaxis]

                cv2.rectangle(video_pixels, 
                              tuple(reversed(object_min_corner)), 
                              tuple(reversed(object_max_corner)),
                              (0, 0, 0),  # color
                              2)  # thickness
                # cv2.rectangle(object_pixels, (0, 0), object_pixels.shape[:2], (0, 0, 0), 1)

                # adds channel and batch axes (C01B order).
                model_input = cv2.resize(gray_object, tuple(norb_image_shape))
                # model_input = cv2.resize(gray_object,
                #                          numpy.zeros(norb_image_shape,
                #                                      dtype=gray_object.dtype),
                #                          0,  # compute dest size for me
                #                          1.0/args.scale,
                #                          1.0/args.scale,
                #                          cv2.INTER_AREA)

                margin = 10
                status_model_input = status_pixels[margin:margin+model_input.shape[0],
                                                   margin:margin+model_input.shape[1],
                                                   :]
                status_model_input[...] = model_input[..., numpy.newaxis]

                model_input = model_input[numpy.newaxis, :, :, numpy.newaxis]
                assert model_input.shape == (1, 96, 96, 1), str(model_input)
                softmax_vector = model_function(model_input).flatten()
                sorted_ids = numpy.argsort(softmax_vector)[-1:-5:-1]
                message = "Category/probability: "
                for sorted_id in sorted_ids:
                    category = new_norb.get_category_value(sorted_id / 10)
                    instance = sorted_id % 10
                    probability = softmax_vector[sorted_id]
                    message = message + "%s%d/%0.2f " % (category, instance, probability)

                print message
                # object_id = numpy.argmax(softmax_vector)
                # category = int(object_id / 10)
                # category_name = new_norb.get_category_value(category)
                # print "Category: %d (%s)" % (category, category_name)
                

            # Displays the resulting frame
            cv2.imshow('frame', all_pixels)

            if args.matplotlib:
                # reverses channels from BGR to RGB
                axes.imshow(video_pixels[..., ::-1],  
                            interpolation='nearest',
                            norm=matplotlib.colors.no_norm())
                figure.canvas.draw()

            # # Adds a singleton batch axis
            # gray_image = gray_image[numpy.newaxis, ...]
            # if args.model is not None:
            #     softmax_vector = model_function(gray_image)
            #     object_id = numpy.argmax(softmax_vector, axis=1)[0]
            #     example = object_to_example[object_id]
            #     # category = int(object_id / 10)
            #     # instance = object_id % 10

            #     # TODO: display an example of category, instance from small NOR
                B

            # Checks whether user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                keep_going = False

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
