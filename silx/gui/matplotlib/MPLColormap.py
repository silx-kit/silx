# New matplotlib colormaps by Nathaniel J. Smith, Stefan van der Walt,
# and (in the case of viridis) Eric Firing.
#
# This file and the colormaps in it are released under the CC0 license /
# public domain dedication. We would appreciate credit if you use or
# redistribute these colormaps, but do not impose any legal restrictions.
#
# To the extent possible under law, the persons who associated CC0 with
# mpl-colormaps have waived all copyright and related or neighboring rights
# to mpl-colormaps.
#
# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
"""Matplotlib's new colormaps"""


import numpy
from matplotlib.colors import ListedColormap
import silx.resources


__all__ = ['magma', 'inferno', 'plasma', 'viridis']

cmaps = {}
for name in ('magma', 'inferno', 'plasma', 'viridis'):
    filename = silx.resources.resource_filename("gui/colormaps/%s.npy" % name)
    data = numpy.load(filename)
    cmaps[name] = ListedColormap(data, name=name)

magma = cmaps['magma']
inferno = cmaps['inferno']
plasma = cmaps['plasma']
viridis = cmaps['viridis']
