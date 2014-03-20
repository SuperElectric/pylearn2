"""
An interface to the small NORB dataset. Unlike ./norb_small.py, this reads the
original NORB file format, not the LISA lab's .npy version.

Currently only supports the Small NORB Dataset.

Download the dataset from:
http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/

NORB dataset(s) by Fu Jie Huang and Yann LeCun.
"""

__authors__ = "Guillaume Desjardins and Matthew Koichi Grimes"
__copyright__ = "Copyright 2010-2014, Universite de Montreal"
__credits__ = __authors__.split(" and ")
__license__ = "3-clause BSD"
__maintainer__ = "Matthew Koichi Grimes"
__email__ = "mkg alum mit edu (@..)"


import os, gzip, bz2, warnings, functools
import numpy, theano
from pylearn2.utils import safe_zip
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace, Conv2DSpace, CompositeSpace


class SmallNORB(DenseDesignMatrix):
    """
    An interface to the small NORB dataset.

    If instantiated with default arguments, target labels are integers
    representing categories, which can be looked up using

      category_name = SmallNORB.get_category(label).

    If instantiated with multi_target=True, labels are vectors of indices
    representing:

      [ category, instance, elevation, azimuth, lighting ]

    Like with category, there are class methods that map these ints to their
    actual values, e.g:

      category = SmallNORB.get_category(label[0])
      elevation = SmallNORB.get_elevation_degrees(label[2])
    """

    # Actual image shape may change, e.g. after being preprocessed by
    # datasets.preprocessing.Downsample
    original_image_shape = (96, 96)

    _categories = ['animal',  # four-legged animal
                   'human',  # human figure
                   'airplane',
                   'truck',
                   'car']

    @classmethod
    def get_category(cls, scalar_label):
        """
        Returns the category string corresponding to an integer category label.
        """
        return cls._categories[int(scalar_label)]

    @classmethod
    def get_elevation_degrees(cls, scalar_label):
        """
        Returns the elevation, in degrees, corresponding to an integer
        elevation label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label < 9
        return 30 + 5 * scalar_label

    @classmethod
    def get_azimuth_degrees(cls, scalar_label):
        """
        Returns the azimuth, in degrees, corresponding to an integer
        label.
        """
        scalar_label = int(scalar_label)
        assert scalar_label >= 0
        assert scalar_label <= 34
        assert (scalar_label % 2) == 0
        return scalar_label * 10

    # Maps azimuth labels (ints) to their actual values, in degrees.
    azimuth_degrees = numpy.arange(0, 341, 20)

    # Maps a label type to its index within a label vector.
    label_type_to_index = {'category':  0,
                           'instance':  1,
                           'elevation': 2,
                           'azimuth':   3,
                           'lighting':  4}

    # Number of labels, for each label type.
    num_labels_by_type = (len(_categories),
                          10,  # instances
                          9,   # elevations
                          18,  # azimuths
                          6)   # lighting

    # [mkg] Dropped support for the 'center' argument for now. In Pylearn 1, it
    # shifted the pixel values from [0:255] by subtracting 127.5. Seems like a
    # form of preprocessing, which might be better implemented separately using
    # the Preprocess class.
    def __init__(self, which_set, multi_target=False):
        """
        parameters
        ----------

        which_set : str
            Must be 'train' or 'test'.

        multi_target : bool
            If False, each label is an integer labeling the image catergory. If
            True, each label is a vector: [category, instance, lighting,
            elevation, azimuth]. All labels are given as integers. Use the
            categories, elevation_degrees, and azimuth_degrees arrays to map
            from these integers to actual values.
        """

        assert which_set in ['train', 'test']

        self.which_set = which_set

        X = SmallNORB.load(which_set, 'dat')

        # Casts to the GPU-supported float type, using theano._asarray(), a
        # safer alternative to numpy.asarray().
        #
        # TODO: move the dtype-casting to the view_converter's output space,
        #       once dtypes-for-spaces is merged into master.
        X = theano._asarray(X, theano.config.floatX)

        # Formats data as rows in a matrix, for DenseDesignMatrix
        X = X.reshape(-1, 2 * numpy.prod(self.original_image_shape))

        # This is uint8
        y = SmallNORB.load(which_set, 'cat')
        if multi_target:
            y_extra = SmallNORB.load(which_set, 'info')
            y = numpy.hstack((y[:, numpy.newaxis], y_extra))
            self.label_to_index = {}
            for index, label in enumerate(y):
                self.label_to_index[label] = index
        else:
            self.label_to_index = None

        datum_shape = ((2, ) +  # two stereo images
                       self.original_image_shape +
                       (1, ))  # one color channel

        # 's' is the stereo channel: 0 (left) or 1 (right)
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(datum_shape, axes)

        super(SmallNORB, self).__init__(X=X,
                                        y=y,
                                        view_converter=view_converter)

    @staticmethod
    def _parseNORBFile(file_handle, subtensor=None, debug=False):
        """
        Load all or part of file 'file_handle' into a numpy ndarray

        Parameters
        ----------

        file_handle: file
          A file from which to read. Can be a handle returned by
          open(), gzip.open() or bz2.BZ2File().

        subtensor: slice, or None
          If subtensor is not None, it should be like the
          argument to numpy.ndarray.__getitem__.  The following two
          expressions should return equivalent ndarray objects, but the one
          on the left may be faster and more memory efficient if the
          underlying file f is big.

          read(file_handle, subtensor) <===> read(file_handle)[*subtensor]

          Support for subtensors is currently spotty, so check the code to
          see if your particular type of subtensor is supported.
        """

        def readNums(file_handle, num_type, count):
            """
            Reads 4 bytes from file, returns it as a 32-bit integer.
            """
            num_bytes = count * numpy.dtype(num_type).itemsize
            string = file_handle.read(num_bytes)
            return numpy.fromstring(string, dtype=num_type)

        def readHeader(file_handle, debug=False, from_gzip=None):
            """
            parameters
            ----------

            file_handle : file or gzip.GzipFile
            An open file handle.


            from_gzip : bool or None
            If None determine the type of file handle.

            returns : tuple
            (data type, element size, shape)
            """

            if from_gzip is None:
                from_gzip = isinstance(file_handle,
                                      (gzip.GzipFile, bz2.BZ2File))

            key_to_type = {0x1E3D4C51: ('float32', 4),
                           # what is a packed matrix?
                           # 0x1E3D4C52: ('packed matrix', 0),
                           0x1E3D4C53: ('float64', 8),
                           0x1E3D4C54: ('int32', 4),
                           0x1E3D4C55: ('uint8', 1),
                           0x1E3D4C56: ('int16', 2)}

            type_key = readNums(file_handle, 'int32', 1)[0]
            elem_type, elem_size = key_to_type[type_key]
            if debug:
                print "header's type key, type, type size: ", \
                    type_key, elem_type, elem_size
            if elem_type == 'packed matrix':
                raise NotImplementedError('packed matrix not supported')

            num_dims = readNums(file_handle, 'int32', 1)[0]
            if debug:
                print '# of dimensions, according to header: ', num_dims

            if from_gzip:
                shape = readNums(file_handle,
                                 'int32',
                                 max(num_dims, 3))[:num_dims]
            else:
                shape = numpy.fromfile(file_handle,
                                       dtype='int32',
                                       count=max(num_dims, 3))[:num_dims]

            if debug:
                print 'Tensor shape, as listed in header:', shape

            return elem_type, elem_size, shape

        elem_type, elem_size, shape = readHeader(file_handle, debug)
        beginning = file_handle.tell()

        num_elems = numpy.prod(shape)

        result = None
        if isinstance(file_handle, (gzip.GzipFile, bz2.BZ2File)):
            assert subtensor is None, \
                "Subtensors on gzip files are not implemented."
            result = readNums(file_handle,
                              elem_type,
                              num_elems*elem_size).reshape(shape)
        elif subtensor is None:
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        elif isinstance(subtensor, slice):
            if subtensor.step not in (None, 1):
                raise NotImplementedError('slice with step',
                                          subtensor.step)
            if subtensor.start not in (None, 0):
                bytes_per_row = numpy.prod(shape[1:]) * elem_size
                file_handle.seek(beginning+subtensor.start * bytes_per_row)
            shape[0] = min(shape[0], subtensor.stop) - subtensor.start
            result = numpy.fromfile(file_handle,
                                    dtype=elem_type,
                                    count=num_elems).reshape(shape)
        else:
            raise NotImplementedError('subtensor access not written yet:',
                                      subtensor)

        return result


    @classmethod
    def load(cls, which_set, filetype):
        """
        Reads and returns a single file as a numpy array.
        """

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        def get_path(which_set):
            dirname = os.path.join(os.getenv('PYLEARN2_DATA_PATH'),
                                   'norb_small/original')
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'

            filename = 'smallnorb-5x%sx9x18x6x2x96x96-%s-%s.mat' % \
                (instance_list, which_set + 'ing', filetype)

            return os.path.join(dirname, filename)


        file_handle = open(get_path(which_set))
        return cls._parseNORBFile(file_handle)

    @functools.wraps(DenseDesignMatrix.get_topological_view)
    def get_topological_view(self, mat=None, single_tensor=True):
        result = super(SmallNORB, self).get_topological_view(mat)

        if single_tensor:
            warnings.warn("The single_tensor argument is True by default to "
                          "maintain backwards compatibility. This argument "
                          "will be removed, and the behavior will become that "
                          "of single_tensor=False, as of August 2014.")
            axes = list(self.view_converter.axes)
            s_index = axes.index('s')
            assert axes.index('b') == 0
            num_image_pairs = result[0].shape[0]
            shape = (num_image_pairs, ) + self.view_converter.shape

            # inserts a singleton dimension where the 's' dimesion will be
            mono_shape = shape[:s_index] + (1, ) + shape[(s_index+1):]

            for i, res in enumerate(result):
                print "result %d shape: %s" % (i, str(res.shape))

            result = tuple(t.reshape(mono_shape) for t in result)
            result = numpy.concatenate(result, axis=s_index)
        else:
            warnings.warn("The single_tensor argument will be removed on "
                          "August 2014. The behavior will be the same as "
                          "single_tensor=False.")

        return result


class StereoViewConverter(object):
    """
    Converts stereo image data between two formats:
      A) A dense design matrix, one stereo pair per row (VectorSpace)
      B) An image pair (CompositeSpace of two Conv2DSpaces)
    """
    def __init__(self, shape, axes=None):
        """
        The arguments describe how the data is laid out in the design matrix.

        shape : tuple
          A tuple of 4 ints, describing the shape of each datum.
          This is the size of each axis in <axes>, excluding the 'b' axis.

        axes : tuple
          A tuple of the following elements in any order:
            'b'  batch axis)
            's'  stereo axis)
             0   image axis 0 (row)
             1   image axis 1 (column)
            'c'  channel axis
        """
        shape = tuple(shape)

        if not all(isinstance(s, int) for s in shape):
            raise TypeError("Shape must be a tuple/list of ints")

        if len(shape) != 4:
            raise ValueError("Shape array needs to be of length 4, got %s." %
                             shape)

        datum_axes = list(axes)
        datum_axes.remove('b')
        if shape[datum_axes.index('s')] != 2:
            raise ValueError("Expected 's' axis to have size 2, got %d.\n"
                             "  axes:       %s\n"
                             "  shape:      %s" %
                             (shape[datum_axes.index('s')],
                              axes,
                              shape))
        self.shape = shape
        self.set_axes(axes)

        def make_conv2d_space(shape, axes):
            shape_axes = list(axes)
            shape_axes.remove('b')
            image_shape = tuple(shape[shape_axes.index(axis)]
                                for axis in (0, 1))
            conv2d_axes = list(axes)
            conv2d_axes.remove('s')
            return Conv2DSpace(shape=image_shape,
                               num_channels=shape[shape_axes.index('c')],
                               axes=conv2d_axes)

        conv2d_space = make_conv2d_space(shape, axes)
        self.topo_space = CompositeSpace((conv2d_space, conv2d_space))
        self.storage_space = VectorSpace(dim=numpy.prod(shape))

    def get_formatted_batch(self, batch, space):
        return self.storage_space.np_format_as(batch, space)

    def design_mat_to_topo_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_formatted_view(), get_batch_topo()
        """
        return self.storage_space.np_format_as(design_mat, self.topo_space)

    def design_mat_to_weights_view(self, design_mat):
        """
        Called by DenseDesignMatrix.get_weights_view()
        """
        return self.design_mat_to_topo_view(design_mat)

    def topo_view_to_design_mat(self, topo_batch):
        """
        Used by DenseDesignMatrix.set_topological_view(), .get_design_mat()
        """
        return self.topo_space.np_format_as(topo_batch, self.storage_space)

    def view_shape(self):
        return self.shape

    def weights_view_shape(self):
        return self.view_shape()

    def set_axes(self, axes):
        axes = tuple(axes)

        if len(axes) != 5:
            raise ValueError("Axes must have 5 elements; got %s" % str(axes))

        for required_axis in ('b', 's', 0, 1, 'c'):
            if required_axis not in axes:
                raise ValueError("Axes must contain 'b', 's', 0, 1, and 'c'. "
                                 "Got %s." % str(axes))

        if axes.index('b') != 0:
            raise ValueError("The 'b' axis must come first (axes = %s)." %
                             str(axes))

        def get_batchless_axes(axes):
            axes = list(axes)
            axes.remove('b')
            return tuple(axes)

        if hasattr(self, 'axes'):
            # Reorders the shape vector to match the new axis ordering.
            assert hasattr(self, 'shape')
            old_axes = get_batchless_axes(self.axes)
            new_axes = get_batchless_axes(axes)
            new_shape = tuple(self.shape[old_axes.index(a)] for a in new_axes)
            self.shape = new_shape

        self.axes = axes


def _merge_dicts(*args):
    result = {}
    for arg in args:
        result.update(arg)

    return result

class Norb(SmallNORB):
    """
    A stereo dataset for the same 50 objects (5 classes, 10 objects each) as
    SmallNORB, but with natural imagery composited into the background, and
    distractor objects added near the border (one distractor per image).

    Furthermore, the image labels have the following additional attributes:
      horizontal shift (-6 to +6)
      vertical shift (-6 to +6)
      lumination change (-20 to +20)
      contrast (0.8 to 1.3)
      object scale (0.78 to 1.0)
      rotation (-5 to +5 degrees)

    To allow for these shifts, the images are slightly bigger (108 x 108).
    """

    original_image_shape = (108, 108)

    label_type_to_index = _merge_dicts(SmallNORB.label_type_to_index,
                                      {'horizontal shift': 5,
                                       'vertical shift': 6,
                                       'lumination change': 7,
                                       'contrast': 8,
                                       'object scale': 9,
                                       'rotation': 10})

    num_labels_by_type = SmallNORB.num_labels_by_type + (-1,  # h. shift
                                                         -1,  # v. shift
                                                         -1,  # lumination
                                                         -1,  # contrast
                                                         -1,  # scale
                                                         -1)  # rotation

    @classmethod
    def load(cls, which_set, number, filetype):
        """
        Loads the data from a single NORB file, returning it as a 1-D numpy
        array.
        """

        assert which_set in ['train', 'test']
        assert filetype in ['dat', 'cat', 'info']

        if which_set == 'train':
            assert number in range(1, 3)
        else:
            assert number in range(1, 11)

        def get_path(which_set, number, filetype):
            dirname = os.path.join(os.getenv('PYLEARN2_DATA_PATH'), 'norb')
            if which_set == 'train':
                instance_list = '46789'
            elif which_set == 'test':
                instance_list = '01235'
            else:
                raise ValueError("Expected which_set to be 'train' or 'test', "
                                 "but got '%s'" % which_set)

            filename = 'norb-5x%sx9x18x6x2x108x108-%s-%02d-%s.mat' % \
                (instance_list, which_set + 'ing', number, filetype)

            return os.path.join(dirname, filename)

        file_handle = open(get_path(which_set, number, filetype))
        return cls._parseNORBFile(file_handle)


    def __init__(self,
                 which_set,
                 multi_target=False,
                 memmap_dir=None):
        """
        Loads NORB dataset from $PYLEARN2_DATA_PATH/norb/*.mat into
        memory-mapped numpy.ndarrays, stored by default in
        $PYLEARN2_DATA_PATH/norb/memmap_files/

        We use memory-mapped ndarrays stored on disk instead of conventional
        ndarrays stored in memory, because NORB is > 7GB.

        Parameters:
        -----------
        which_set: str
          'test' or 'train'.

        multi_target: bool
          If True, load all labels in a N x 11 matrix.
          If False, load only the category label in a length-N vector.
          All labels are always read and stored in the memmap file. This
          parameter only changes what is visible to the user.

        memmap_dir: str or None
          Directory to store disk buffers in.
          If None, this defaults to $PYLEARN2_DATA_PATH/norb/memmap_files/

          The following memory-mapped files will be created:
            memmap_dir/<which_set>_images.npy
            memmap_dir/<which_set>_labels.npy

          If either of the above files already exist, it will be used instead
          of reading the NORB files.
        """
        if not which_set in ('test', 'train'):
            raise ValueError("Expected which_set to be 'train' or "
                             "'test', but got '%s'" % which_set)

        norb_dir = os.path.join(os.getenv('PYLEARN2_DATA_PATH'), 'norb')

        if memmap_dir is None:
            memmap_dir = os.path.join(norb_dir, 'memmap_files')
            if not os.path.isdir(memmap_dir):
                os.mkdir(memmap_dir)

        def get_memmap_paths():
            return tuple(os.path.join(memmap_dir,
                                      "%s_%s.npy" % (which_set, which_file))
                         for which_file in ('images', 'labels'))

        images_path, labels_path = get_memmap_paths()

        if os.path.isfile(images_path) != os.path.isfile(labels_path):
            raise ValueError("There is %s memmap file for images, but there "
                             "is %s memmap file for labels. This should not "
                             "happen under normal operation (they must either "
                             "both be missing, or both be present). Erase the "
                             "existing memmap file to regenerate both memmap "
                             "files from scratch." %
                             ("a" if os.path.isfile(images_path) else "no",
                              "a" if os.path.isfile(labels_path) else "no"))

        num_norb_files = 2 if which_set == 'test' else 10
        num_rows = 29160 * num_norb_files
        pixels_per_row = 2 * numpy.prod(Norb.original_image_shape)
        labels_per_row = len(Norb.num_labels_by_type)
        images_shape = (num_rows, pixels_per_row)
        labels_shape = (num_rows, labels_per_row)

        # Opens memmap files as read-only if they already exist.
        memmaps_already_existed = os.path.isfile(images_path)

        if not memmaps_already_existed:
            print "allocating memmap files in %s" % memmap_dir

        memmap_mode = 'r' if memmaps_already_existed else 'w+'
        images = numpy.memmap(images_path,
                              dtype='uint8',
                              mode=memmap_mode,
                              shape=images_shape)
        labels= numpy.memmap(labels_path,
                             dtype='int32',
                             mode=memmap_mode,
                             shape=labels_shape)

        assert str(images.dtype) == 'uint8', images.dtype
        assert str(labels.dtype) == 'int32', labels.dtype

        # Load data from NORB data files if memmap files didn't already exist.
        if not memmaps_already_existed:

            def get_norb_filepaths(which_set, filetype):
                if which_set == 'train':
                    instance_list = '46789'
                    numbers = range(1, 11)
                elif which_set == 'test':
                    instance_list = '01235'
                    numbers = range(1, 3)

                template = ('norb-5x%sx9x18x6x2x108x108-%s-%%02d-%s.mat' %
                            (instance_list, which_set + 'ing', filetype))
                return tuple(os.path.join(norb_dir, template % n)
                             for n in numbers)

            # Temporarily folds images, labels into file-sized chunks, to
            # iterate through.
            images = images.reshape((num_norb_files, -1, images.shape[1]))
            labels = labels.reshape((num_norb_files, -1, labels.shape[1]))

            # Reads images from NORB's 'dat' files.
            for images_chunk, norb_filepath in \
                safe_zip(images, get_norb_filepaths(which_set, 'dat')):

                print "copying images from %s" % norb_filepath

                data = Norb._parseNORBFile(open(norb_filepath))
                assert data.dtype == images.dtype, \
                    ("data.dtype: %s, images.dtype: %s" %
                     (data.dtype, images.dtype))

                images_chunk[...] = data.reshape(images_chunk.shape)

            # Reads label data from NORB's 'cat' and 'info' files
            for labels_chunk, cat_filepath, info_filepath in \
                safe_zip(labels,
                         get_norb_filepaths(which_set, 'cat'),
                         get_norb_filepaths(which_set, 'info')):
                categories = Norb._parseNORBFile(open(cat_filepath))

                print ("copying labels from %s and %s" %
                       (os.path.split(cat_filepath)[0],
                        os.path.split(cat_filepath)[1]))
                info = Norb._parseNORBFile(open(info_filepath))
                info = info.reshape((labels_chunk.shape[0],
                                     labels_chunk.shape[1]-1))

                assert categories.dtype == labels.dtype, \
                    ("categories.dtype: %s, labels.dtype: %s" %
                     (categories.dtype, labels.dtype))

                assert info.dtype == labels.dtype, \
                    ("info.dtype: %s, labels.dtype: %s" %
                     (info.dtype, labels.dtype))

                labels_chunk[:, 0] = categories
                labels_chunk[:, 1:] = info

            # Unfolds images, labels back into matrices
            images = images.reshape((num_rows, -1))
            labels = labels.reshape((num_rows, -1))

        if not multi_target:
            labels = labels[:, 0:1]

        stereo_pair_shape = ((2, ) +  # two stereo images
                             Norb.original_image_shape +  # image dimesions
                             (1, ))   # one channel
        axes = ('b', 's', 0, 1, 'c')
        view_converter = StereoViewConverter(stereo_pair_shape, axes)

        # Call DenseDesignMatrix constructor directly, skipping SmallNORB ctor
        super(SmallNORB, self).__init__(X=images,
                                        y=labels,
                                        view_converter=view_converter)

        # code here for debugging, to print out the observed values for each
        # label type to confirm their ranges and density.

        def debug_confirm():
            for category in self.label_type_to_index.keys():
                labels = self.y[:, self.label_type_to_index[category]]
                unique_labels = list(frozenset(labels))
                unique_labels = numpy.sort(unique_labels)
                print "%s values: " % category, unique_labels


        debug_confirm()
