import logging
import sys

logging.basicConfig()

from silx.app.view.main import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv)
