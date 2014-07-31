#! /usr/bin/env python

import argparse, subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=
                                     "Converts a set of images into a video.")

    parser.add_argument("-f",
                        "--frame_template",
                        required=True,
                        help=("A C-style string template (e.g. "
                              "'frame_%%05d.png') that fits the "
                              "filenames of the input frame images."))

    parser.add_argument("-o",
                        "--output",
                        default="output.mkv",
                        help=("Output video file. Format inferred from file "
                              "extension. MKV seems to do better than mpg."))

    parser.add_argument("-r",
                        "--framerate",
                        type=int,
                        required=True,
                        help="The frame rate.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Example command:
    # avconv -r 7 -i frame_%05d.png output.mkv

    command = ['avconv',
               '-r',
               '%d' % args.framerate,
               '-i',
               args.frame_template,
               args.output]

    subprocess.call(command)


if __name__ == '__main__':
    main()
