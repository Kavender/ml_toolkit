#!/usr/bin/env python
import numpy


def huber_loss(true, pred, delta):
    loss = numpy.where(numpy.abs(true - pred) <  delta, 0.5*numpy.power(true - pred, 2), \
            delta*numpy.abs(true - pred) - 0.5*numpy.power(delta, 2))
    return numpy.sum(loss)

def logcosh_loss(true, pred):
    loss = numpy.log(numpy.cosh(pred - true))
    return numpy.sum(loss)

def quantilted_loss(true, pred, quantile):
    loss = (true - pred)
    return numpy.mean(numpy.max(quantile * loss, (quantile -1) * loss), axis = -1)

