"""
Plugins for the Train object.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import numpy as np
import os.path
from pylearn2.utils import serial
from pylearn2.gui.get_weights_report import get_weights_report

class TrainExtension(object):
    """
    An object called by pylearn2.train.Train at various
    points during learning.
    Useful for adding custom features to the basic learning
    procedure.

    This base class implements all callback methods as no-ops.
    To add a feature to the Train class, implement a subclass of this
    base class that overrides any subset of these no-op methods.
    """

    def on_save(self, model, dataset, algorithm):
        """
        Train calls this immediately before it saves the model.

        Parameters
        ----------
        model : object
            The model object being trained (implementing some subset of the \
            `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

    def on_monitor(self, model, dataset, algorithm):
        """
        Train calls this immediately after each call to the Monitor
        (i.e., when training begins, and at the end of each epoch).

        Parameters
        ----------
        model : object
            The model object being trained (implementing some \
            subset of the `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

    def setup(self, model, dataset, algorithm):
        """
        Train calls this immediately upon instantiation,
        before any monitoring is done.

        Parameters
        ----------
        model : object
            The model object being trained (implementing some \
            subset of the `pylearn2.models` interface).

        dataset : object
            The dataset object being trained (implementing the \
            `pylearn2.datasets` interface).

        algorithm : object
            The object representing the training algorithm being \
            used to train the model (and thus implementing the \
            `pylearn2.training_algorithms` interface).
        """

class SharedSetter(TrainExtension):
    """
    Sets shared variables to take on the specified values after the
    specified amounts of epochs have taken place.

    epoch_updates = [ [i, x, y] ]

    means run x.set_value(cast(y))

    after i epochs have passed.
    """

    def __init__(self, epoch_updates):
        """
        .. todo::

            WRITEME
        """
        self._count = 0
        self._epoch_to_updates = {}
        self._vars = set([])
        for update in epoch_updates:
            epoch, var, val = update
            self._vars.add(var)
            if epoch not in self._epoch_to_updates:
                self._epoch_to_updates[epoch] = []
            assert hasattr(var, 'get_value')
            assert var.name is not None
            self._epoch_to_updates[epoch].append((var,val))

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        if self._count == 0:
            monitor = model.monitor
            # TODO: make Monitor support input-less channels so this hack
            # isn't necessary
            hack = monitor.channels.values()[0]
            for var in self._vars:
                monitor.add_channel(name=var.name, val=var,
                                    ipt=hack.graph_input, dataset=hack.dataset)


        if self._count in self._epoch_to_updates:
            for update in self._epoch_to_updates[self._count]:
                var, val = update
                var.set_value(np.cast[var.dtype](val))
        self._count += 1

class ChannelSmoother(TrainExtension):
    """
    Makes a smoothed version of a monitoring channel by averaging together
    the k most recent values of that channel.
    This is a little bit dangerous because if other TrainExtensions depend
    on the channel being up to date they must appear after this one in the
    extensions list. A better long term solution would be to make the Monitor
    support this kind of channel directly instead of hacking it in.
    Note that the Monitor will print this channel as having a value of -1, and
    then the extension will print the right value.
    """

    def __init__(self, channel_to_smooth, channel_to_publish, k=5):
        """
        .. todo::

            WRITEME
        """
        self.__dict__.update(locals())
        del self.self

    def setup(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        monitor = model.monitor
        channels = monitor.channels
        channel_to_smooth = channels[self.channel_to_smooth]
        ipt = channel_to_smooth.graph_input
        dataset = channel_to_smooth.dataset

        monitor.add_channel(name=self.channel_to_publish,
                            ipt=ipt,
                            val=-1.,
                            dataset=dataset)

        self.in_ch = channel_to_smooth
        self.out_ch = channels[self.channel_to_publish]

    def on_monitor(self, model, dataset, algorithm):
        """
        .. todo::

            WRITEME
        """
        val_record = self.in_ch.val_record

        start = max(0, len(val_record) - self.k + 1)
        values = val_record[start:]
        mean = sum(values) / float(len(values))

        self.out_ch.val_record[-1] = mean
        print '\t' + self.channel_to_publish + ': ' + str(mean)


class EpochLogger(TrainExtension):
    """
    Saves the machine and/or an image of its first-level weights on each epoch.
    """

    def __init__(self, output_dir, save_models=False, save_images=True):

        output_dir = os.path.abspath(output_dir)

        def is_empty(dirname):
            return len(os.listdir(dirname)) == 0

        if os.path.exists(output_dir):
            if not os.path.isdir(output_dir):
                raise IOError("Output directory %s is not a directory." %
                              output_dir)
            elif not is_empty(output_dir):
                raise IOError("Output directory %s is not empty." % output_dir)
        else:
            parent, dirname = os.path.split(output_dir)
            if parent == '':
                raise IOError("Are we really writing to the root directory?")
            os.makedirs(output_dir)


        self.output_dir = output_dir
        self.num_finished_epochs = 0
        self.save_models = save_models
        self.save_images = save_images


    def on_monitor(self, model, dataset, algorithm):
        """
        Overrides TrainExtension's on_monitor, which gets called once before
        training, and after each epoch.
        """
        class SerializationGuard(object):

            def __getstate__(self):
                raise IOError("You tried to serialize something that should not"
                              " be serialized.")


        save_path = os.path.join(self.output_dir,
                                 "model_after_%04d_epochs.pkl" %
                                 self.num_finished_epochs)
        if os.path.exists(save_path):
            raise IOError("The output file already exists. "
                          "This should never happen.")

        if self.save_models:
            try:
                # Prevents the dataset from being saved along with the model.
                dataset._serialization_guard = SerializationGuard()

                serial.save(save_path, model, on_overwrite = 'ignore')
            finally:
                dataset._serialization_guard = None

        if self.save_images:
            # Uses same options as show_weights.py
            patch_viewer = get_weights_report(model = model,
                                              rescale = "individual",
                                              border = False)
            patch_viewer.save(os.path.join(self.output_dir,
                                           "weights_after_%04d_epochs.png" %
                                           self.num_finished_epochs))

        self.num_finished_epochs = 1 + self.num_finished_epochs

