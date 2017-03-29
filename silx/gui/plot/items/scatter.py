


class Scatter(Base, ColormapMixIn, SymbolMixIn):
    """Description of a scatter plot"""

    # TODO add toRgba method

    _DEFAULT_SYMBOL = 'o'
    """Default symbol of the scatter plots"""

    def __init__(self, plot, legend=None):
        Base.__init__(self, plot, legend)
        ColormapMixIn.__init__(self)
        SymbolMixIn.__init__(self)
        self._x = ()
        self._y = ()
        self._value = ()

    def getXData(self, copy=True):
        """Returns the x coordinates of the data points
        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._x, copy=copy)

    def getYData(self, copy=True):
        """Returns the y coordinates of the data points
        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._y, copy=copy)

    def getValueData(self, copy=True):
        """Returns the value of the data points
        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :rtype: numpy.ndarray
        """
        return numpy.array(self._value, copy=copy)

    def getData(self, copy=True):
        """Returns the x, y coordinates and the value of the data points
        :param copy: True (Default) to get a copy,
                     False to use internal representation (do not modify!)
        :returns: (x, y, value)
        :rtype: 3-tuple of numpy.ndarray
        """
        x = self.getXData(copy)
        y = self.getYData(copy)
        value = self.getValueData(copy)
        return x, y, value

    def _setData(self, x, y, value, copy=True):
        x = numpy.array(x, copy=copy)
        y = numpy.array(y, copy=copy)
        value = numpy.array(value, copy=copy)
        assert x.ndim == y.ndim == value.ndim == 1
        assert len(x) == len(y) == len(value)
        self._x, self._y, self._value = x, y, value
