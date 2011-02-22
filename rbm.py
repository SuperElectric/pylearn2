"""
Implementations of Restricted Boltzmann Machines and associated sampling
strategies.
"""
import numpy
import theano
from theano import tensor
from theano.tensor import nnet
from pylearn.gd.sgd import sgd_updates
from pylearn.algorithms.mcRBM import contrastive_grad

from base import Block, Optimizer
from utils import sharedX

theano.config.warn.sum_div_dimshuffle_bug = False
floatX = theano.config.floatX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams

class Sampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.
    """
    def __init__(self, conf, rbm, particles, rng):
        self.__dict__.update(conf=conf, rbm=rbm)
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2**30))
        self.s_rng = RandomStreams(seed)
        self.particles = sharedX(particles, name='particles')

    def updates(self):
        """
        These are update formulas that deal with the Markov chain, not
        model parameters.
        """
        raise NotImplementedError()

class PersistentCDSampler(Sampler):
    def updates(self):
        """
        These are update formulas that deal with the Markov chain, not
        model parameters.
        """
        new_particles, _locals = self.rbm.gibbs_step_for_v(
            self.particles,
            self.s_rng
        )
        if not hasattr(self.rbm, 'h_sample'):
            self.rbm.h_sample = sharedX(numpy.zeros((0, 0), 'h_sample'))
        return {
            self.particles: new_particles,
            self.rbm.h_sample: _locals['h_mean']
        }

class RBM(Block):
    """A base interface for RBMs, implementing the binary-binary case."""
    def __init__(self, conf, rng=None):
        if rng is None:
            rng = numpy.random.RandomState(conf['rbm_seed'])
        self.conf = conf
        self.visbias = sharedX(
            numpy.zeros(conf['n_vis']),
            name='vb',
            borrow=True
        )
        self.hidbias = sharedX(
            numpy.zeros(conf['n_hid']),
            name='hb',
            borrow=True
        )
        self.weights = sharedX(
            .5 * rng.rand(conf['n_vis'], conf['n_hid']),
            name='W',
            borrow=True
        )

    def cd_updates(self, pos_v, neg_v, lr, other_cost=0):
        """
        Get the contrastive gradients given positive and negative phase
        visible units, and do a gradient step on the parameters using
        the learning rates in `lr` (which is a list in the same order
        as self.params()).
        """
        grads = contrastive_grad(
            self.free_energy_given_v,
            pos_v, neg_v,
            wrt=self.params(),
            other_cost=other_cost
        )
        stepsizes = lr
        rval = dict(sgd_updates(self.params(), grads, stepsizes=stepsizes))
        grad_shared_vars = [
            sharedX(0 * p.get_value().copy(), '') for p in self.params()
        ]

    def gibbs_step_for_v(self, v, rng):
        """
        Do a round of block Gibbs sampling given visible configuration
        `v`, which could be training examples or "fantasy" particles.
        """
        # For binary hidden units
        # TODO: factor further to extend to other kinds of hidden units
        #       (e.g. spike-and-slab)
        h_mean = self.mean_h_given_v(v)
        h_mean_shape = self.conf['batchsize'], self.conf['n_hid']
        h_sample = tensor.cast(rng.uniform(size=h_mean_shape) < h_mean, floatX)
        v_mean_shape = self.conf['batchsize'], self.conf['n_vis']
        # v_mean is always based on h_sample, not h_mean, because we don't
        # want h transmitting more than one bit of information per unit.
        v_mean = self.mean_v_given_h(h_sample)
        v_sample = self.sample_visibles([v_mean], v_mean_shape, rng)
        return v_sample, locals()

    def sample_visibles(self, params, shape, rng):
        v_mean = params[0]
        return tensor.cast(rng.uniform(size=shape) < v_mean, floatX)

    def input_to_h_from_v(self, v):
        return self.hidbias + self.dot(v, self.weights)

    def mean_h_given_v(self, v):
        """
        Mean values of the hidden units given a visible configuration.
        Threshold this in order to sample.
        """
        return nnet.sigmoid(self._input_to_h_from_v(v))

    def mean_v_given_h(self, h):
        """
        Mean reconstruction of the visible units given a hidden unit
        configuration.
        """
        return nnet.sigmoid(self.visbias + self.dot(h, self.weights.T))

    def free_energy_given_v(self, v):
        """
        Calculate the free energy of a visible unit configuration by
        marginalizing over the hidden units.
        """
        sigmoid_arg = self.input_to_h_from_v(v)
        return -(tensor.dot(v, self.visbias).sum(axis=1) +
                 nnet.softplus(sigmoid_arg).sum(axis=1))

    def __call__(self, v):
        return self.mean_h_given_v(v)

