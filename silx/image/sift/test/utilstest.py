#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Test suite for all OpenCL preprocessing kernels.
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2013 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/09/2016"

import os
import imp
import sys
import subprocess
import threading
import distutils.util
import logging
try:  # Python3
    from urllib.request import urlopen, ProxyHandler, build_opener
except ImportError:  # Python2
    from urllib2 import urlopen, ProxyHandler, build_opener
# import urllib2
# import bz2
# import gzip
import numpy
import shutil
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def copy(infile, outfile):
    "link or copy file according to the OS"
    if "symlink" in dir(os):
        os.symlink(infile, outfile)
    else:
        shutil.copy(infile, outfile)


class UtilsTest(object):
    """
    Static class providing useful stuff for preparing tests.
    """
    timeout = 60  # timeout in seconds for downloading images
    url_base = "http://upload.wikimedia.org"
    # Nota https crashes with error 501 under windows.
#    url_base = "https://forge.epn-campus.eu/attachments/download"
    test_home = os.path.dirname(os.path.abspath(__file__))
    sem = threading.Semaphore()
    recompiled = False
    name = "sift_pyocl"
    image_home = os.path.join(test_home, "testimages")
    if not os.path.isdir(image_home):
        os.makedirs(image_home)
    platform = distutils.util.get_platform()
    architecture = "lib.%s-%i.%i" % (platform,
                                    sys.version_info[0], sys.version_info[1])
    sift_home = os.path.join(os.path.dirname(test_home),
                                        "build", architecture)
    logger.info("sift Home is: " + sift_home)
    if name in sys.modules:
        logger.info("sift module was already loaded from  %s" % sys.modules[name])
        sift = None
        sys.modules.pop(name)

    if not os.path.isdir(sift_home):
        with sem:
            if not os.path.isdir(sift_home):
                logger.warning("Building sift to %s" % sift_home)
                p = subprocess.Popen([sys.executable, "setup.py", "build"],
                                 shell=False, cwd=os.path.dirname(test_home))
                logger.info("subprocess ended with rc= %s" % p.wait())
                recompiled = True
    opencl = os.path.join(os.path.dirname(test_home), "openCL")
    for clf in os.listdir(opencl):
        if clf.endswith(".cl") and clf not in os.listdir(os.path.join(sift_home, name)):
            copy(os.path.join(opencl, clf), os.path.join(sift_home, name, clf))
    sift = imp.load_module(*((name,) + imp.find_module(name, [sift_home])))
    sys.modules[name] = sift
    logger.info("sift loaded from %s" % sift.__file__)

    @classmethod
    def deep_reload(cls):
        logger.info("Loading sift")
        cls.sift = None
        sift = None
        sys.path.insert(0, cls.sift_home)
        for key in sys.modules.copy():
            if key.startswith(cls.name):
                sys.modules.pop(key)

        import sift_pyocl as sift
        cls.sift = sift
        logger.info("sift loaded from %s" % sift.__file__)
        sys.modules[cls.name] = sift
        return cls.sift

    @classmethod
    def forceBuild(cls, remove_first=True):
        """
        force the recompilation of sift
        """
        if not cls.recompiled:
            with cls.sem:
                if not cls.recompiled:
                    logger.info("Building sift to %s" % cls.sift_home)
                    if cls.name in sys.modules:
                        logger.info("sift module was already loaded from  %s" % sys.modules[cls.name])
                        cls.sift = None
                        sys.modules.pop(cls.name)
                    if remove_first:
                        recursive_delete(cls.sift_home)
                    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                                     shell=False, cwd=os.path.dirname(cls.test_home))
                    logger.info("subprocess ended with rc= %s" % p.wait())
                    opencl = os.path.join(os.path.dirname(cls.test_home), "openCL")
                    for clf in os.listdir(opencl):
                        if clf.endswith(".cl") and clf not in os.listdir(os.path.join(cls.sift_home, cls.name)):
                            copy(os.path.join(opencl, clf), os.path.join(cls.sift_home, cls.name, clf))
                    cls.sift = cls.deep_reload()
                    cls.recompiled = True



    @classmethod
    def timeoutDuringDownload(cls, imagename=None):
            """
            Function called after a timeout in the download part ...
            just raise an Exception.
            """
            if imagename is None:
                imagename = "2252/testimages.tar.bz2 unzip it "
            raise RuntimeError("Could not automatically \
                download test images!\n \ If you are behind a firewall, \
                please set both environment variable http_proxy and https_proxy.\
                This even works under windows ! \n \
                Otherwise please try to download the images manually from \n %s/%s and put it in in test/testimages." % (cls.url_base, imagename))



    @classmethod
    def getimage(cls, imagename):
        """
        Downloads the requested image from Internet. DEPRECATED
        
        :param imagename: name of the image.
        For the RedMine forge, the filename contains a directory name that is removed
        :return: full path of the locally saved file
        """
        baseimage = os.path.basename(imagename)
        logger.info("UtilsTest.getimage('%s')" % baseimage)
        fullimagename = os.path.abspath(os.path.join(cls.image_home, baseimage))
        if not os.path.isfile(fullimagename):
            logger.info("Trying to download image %s, timeout set to %ss"
                          % (imagename, cls.timeout))
            dictProxies = {}
            if "http_proxy" in os.environ:
                dictProxies['http'] = os.environ["http_proxy"]
                dictProxies['https'] = os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                dictProxies['https'] = os.environ["https_proxy"]
            if dictProxies:
                proxy_handler = ProxyHandler(dictProxies)
                opener = build_opener(proxy_handler).open
            else:
                opener = urlopen

#           Nota: since python2.6 there is a timeout in the urllib2
            timer = threading.Timer(cls.timeout + 1, cls.timeoutDuringDownload, args=[imagename])
            timer.start()
            logger.info("wget %s/%s" % (cls.url_base, imagename))
            data = opener("%s/%s" % (cls.url_base, imagename),
                          data=None, timeout=cls.timeout).read()
            logger.info("Image %s successfully downloaded." % imagename)

            try:
                open(fullimagename, "wb").write(data)
            except IOError:
                raise IOError("unable to write downloaded \
                    data to disk at %s" % cls.image_home)

            if not os.path.isfile(fullimagename):
                raise RuntimeError("Could not automatically \
                download test images %s!\n \ If you are behind a firewall, \
                please set both environment variable http_proxy and https_proxy.\
                This even works under windows ! \n \
                Otherwise please try to download the images manually from \n%s/%s" % (imagename, cls.url_base, imagename))

        return fullimagename


def recursive_delete(strDirname):
    """
    Delete everything reachable from the directory named in "top",
    assuming there are no symbolic links.
    CAUTION:  This is dangerous!  For example, if top == '/', it
    could delete all your disk files.
    :param strDirname: top directory to delete
    :type strDirname: string
    """
    for root, dirs, files in os.walk(strDirname, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(strDirname)


def getLogger(filename=__file__):
    """
    small helper function that initialized the logger and returns it
    """
    basename = os.path.basename(os.path.abspath(filename))
    basename = os.path.splitext(basename)[0]
    mylogger = logging.getLogger(basename)
    mylogger.setLevel(logging.root.level)
    mylogger.debug("tests loaded from file: %s" % basename)
    return mylogger

