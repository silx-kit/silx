How to contribute to *silx*
---------------------------

This document describes how to contribute to the *silx* project.
The process is similar to many other open-source projects like *numpy*, just lighter as the project is smaller, so you won't be surprised with the pipeline.

*scikit-image* provides a nice tutorial in `their own CONTRIBUTING guide`_.


1. Create your GitHub account `https://help.github.com/categories/setup/`
   and upload your SSH keys.

2. Fork the silx project from `https://github.com/silx-kit/silx/`.
   The button is on the top right of the page.

3. Clone your GitHub version locally on the computer you intend to work on.

.. code-block:: bash

   git clone git@github.com/<your_user_name>/silx

4. Install the dependencies defined in *requirements-dev.txt*.

.. code-block:: bash

   pip install -r requirements-dev.txt

5. Make sure the silx test suite pass on your computer using the ``python3 run_tests.py`` or
   ``python3 run_tests.py silx.gui.test.test_qt`` if you want to test only a subset of it. 
   You can use ``python /path/to/silx/bootstrap.py script.py`` to test your scripts without
   installing silx, but the test suite is required to pass.

6. Open an issue in ``https://github.com/silx-kit/silx/issues`` to inform the
   maintainers of your intentions.

7. Create a local branch to start working on your issue ``git branch my_feature``.

8. Code, enjoy but ensure that the new code is tested and does not break
   the current test suite.

9. Push your local branch to your GitHub account: ``git push origin my_feature``.

10. Create a pull request (PR) from your feature branch on GitHub to trigger
    the review process. Indicate this PR is related to the issue you opened in 6.

11. Discuss with the maintainer who is reviewing your code using the GitHub interface.

If you encounter any problems or have any questions you can always ask on the `Issues page`_.

.. _their own CONTRIBUTING guide: https://github.com/scikit-image/scikit-image/blob/3736339272b9d129f98fc723b508ac5490c171fa/CONTRIBUTING.rst
.. _Issues page: https://github.com/silx-kit/silx/issues
