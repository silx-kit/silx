import logging
import os.path
import glob

try:
    from fabio.utils.cli import relax_ulimit as _relax_ulimit
except ImportError:
    try:
        import resource
    except ImportError:
        resource = None

    _logger = logging.getLogger(__name__)

    def _relax_ulimit():
        if resource is None:
            _logger.debug("No resource module available")
        else:
            if hasattr(resource, "RLIMIT_NOFILE"):
                try:
                    hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
                    resource.setrlimit(
                        resource.RLIMIT_NOFILE, (hard_nofile, hard_nofile)
                    )
                except (ValueError, OSError):
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    while 2 * soft < hard:
                        try:
                            resource.setrlimit(resource.RLIMIT_NOFILE, (2 * soft, hard))
                        except (ValueError, OSError):
                            _logger.warning(
                                f"Set the max opened files limit to ({soft}, {hard})"
                            )
                            return
                        else:
                            soft *= 2
                    _logger.warning(
                        "Failed to retrieve and set the max opened files limit"
                    )

                else:
                    _logger.debug("Set max opened files to %d", hard_nofile)


def expand_filenames(filenames):
    """
    Takes a list of paths and expand it into a list of files.

    :param List[str] filenames: list of filenames or path with wildcards
    :rtype: List[str]
    :return: list of existing filenames or non-existing files
        (which was provided as input)
    """
    result = []
    for filename in filenames:
        if os.path.exists(filename):
            result.append(filename)
        elif glob.has_magic(filename):
            expanded_filenames = glob.glob(filename)
            if expanded_filenames:
                result += expanded_filenames
            else:  # Cannot expand, add as is
                result.append(filename)
        else:
            result.append(filename)
    return result


def increase_opened_files_limit():
    """Increases the soft limit on number of opened files. Only works on Unix."""
    _relax_ulimit()
