import logging

logging.basicConfig()

# Import here for static analysis to work
import silx.app.compare.main  # noqa: E402, F401
import silx.app.convert  # noqa: E402, F401
import silx.app.test_  # noqa: E402, F401
import silx.app.view.main  # noqa: E402, F401
from silx.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
