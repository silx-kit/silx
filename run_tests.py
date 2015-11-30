#!/usr/bin/env python
# coding: utf-8

__authors__ = ["Jérôme Kieffer", "Thomas Vincent"]
__date__ = "30/11/2015"
__license__ = "MIT"

import distutils
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_tests")


logger.info("Python %s %s" % (sys.version, tuple.__itemsize__ * 8))

try:
    import numpy
except Exception as error:
    logger.warning("Numpy missing: %s" % error)
else:
    logger.info("Numpy %s" % numpy.version.version)


try:
    import h5py
except Exception as error:
    logger.warning("h5py missing: %s" % error)
else:
    logger.info("h5py %s" % h5py.version.version)


def report_rst(cov, package, version="0.0.0", base=""):
    """
    Generate a report of test coverage in RST (for Sphinx inclusion)
    
    :param cov: test coverage instance
    :param str package: Name of the package
    :return: RST string
    """
    import tempfile
    fd, fn = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    cov.xml_report(outfile=fn)

    from lxml import etree
    xml = etree.parse(fn)
    classes = xml.xpath("//class")

    import time
    line0 = "Test coverage report for %s" % package
    res = [line0, "=" * len(line0), ""]
    res.append("Measured on *%s* version %s, %s" % (package, version, time.strftime("%d/%m/%Y")))
    res += ["",
            ".. csv-table:: Test suite coverage",
            '   :header: "Name", "Stmts", "Exec", "Cover"',
            '   :widths: 35, 8, 8, 8',
            '']
    tot_sum_lines = 0
    tot_sum_hits = 0

    for cl in classes:
        name = cl.get("name")
        fname = cl.get("filename")
        if not os.path.abspath(fname).startswith(base):
            continue
        lines = cl.find("lines").getchildren()
        hits = [int(i.get("hits")) for i in lines]

        sum_hits = sum(hits)
        sum_lines = len(lines)

        cover = 100.0 * sum_hits / sum_lines if sum_lines else 0

        res.append('   "%s", "%s", "%s", "%.1f %%"' % (name, sum_lines, sum_hits, cover))
        tot_sum_lines += sum_lines
        tot_sum_hits += sum_hits
    res.append("")
    res.append('   "%s total", "%s", "%s", "%.1f %%"' %
               (package, tot_sum_lines, tot_sum_hits,
                100.0 * tot_sum_hits / tot_sum_lines))
    res.append("")
    return os.linesep.join(res)


def build_project(root_dir):
    """Run python setup.py build for the project.

    Build directory can be modified by environment variables.

    :param str root_dir: Root directory of the project
    :return: The path to the directory were build was performed
    """
    logger.debug("Getting project name in %s" % root_dir)
    p = subprocess.Popen([sys.executable, "setup.py", "--name"],
                         shell=False, cwd=root_dir, stdout=subprocess.PIPE)
    name, stderr_data = p.communicate()
    logger.debug("subprocess ended with rc= %s" % p.returncode)
    logger.info("Project name: %s" % name)

    platform = distutils.util.get_platform()
    architecture = "lib.%s-%i.%i" % (platform,
                                     sys.version_info[0], sys.version_info[1])

    if os.environ.get("PYBUILD_NAME") == name:
        # we are in the debian packaging way
        home = os.environ.get("PYTHONPATH", "").split(os.pathsep)[-1]
    elif os.environ.get("BUILDPYTHONPATH"):
        home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
    else:
        home = os.path.join(root_dir, "build", architecture)

    logger.warning("Building %s to %s" % (name, home))
    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=root_dir)
    logger.debug("subprocess ended with rc= %s" % p.wait())
    return home


from argparse import ArgumentParser

parser = ArgumentParser(description='Run the tests.')

parser.add_argument("-i", "--insource",
                    action="store_true", dest="insource", default=False,
                    help="Use the build source and not the installed version")
parser.add_argument("-c", "--coverage", dest="coverage",
                    action="store_true", default=False,
                    help="report coverage of fabio code (requires 'coverage' module)")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="increase verbosity")
options = parser.parse_args()
sys.argv = [sys.argv[0]]


if options.verbose == 1:
    logging.root.setLevel(logging.INFO)
    logger.info("Set log level: INFO")
elif options.verbose > 1:
    logging.root.setLevel(logging.DEBUG)
    logger.info("Set log level: DEBUG")


if options.coverage:
    logger.info("Running test-coverage")
    import coverage
    try:
        cov = coverage.Coverage(omit=["*test*", "*third_party*"])
    except AttributeError:
        cov = coverage.coverage(omit=["*test*", "*third_party*"])
    cov.start()


# Prevent importing from source directory
if (os.path.dirname(os.path.abspath(__file__)) ==
        os.path.abspath(sys.path[0])):
    removed_from_sys_path = sys.path.pop(0)
    logger.info("Patched sys.path, removed: '%s'" % removed_from_sys_path)


if not options.insource:
    try:
        import silx
    except:
        logger.warning(
            "silx missing, using built (i.e. not installed) version")
        options.insource = True

if options.insource:
    build_dir = build_project(os.path.dirname(os.path.abspath(__file__)))

    sys.path.insert(0, build_dir)
    logger.warning("Patched sys.path, added: '%s'" % build_dir)
    import silx


logger.warning("Test silx %s from %s" % (silx.version, silx.__path__[0]))
import silx.test
if silx.test.run_tests():
    logger.info("Test suite  succeeded")
else:
    logger.warning("Test suite failed")


if options.coverage:
    cov.stop()
    cov.save()
    with open("coverage.rst", "w") as fn:
        fn.write(report_rst(cov, "silx", silx.version, silx.__path__[0]))
    print(cov.report())
