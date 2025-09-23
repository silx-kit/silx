Development process
===================

This project follows the standard open-source project github workflow,
which is described in other projects like `scikit-image <https://scikit-image.org/docs/stable/development/contribute.html>`_.

1. Create your `GitHub <https://github.com/>`_ account and upload your SSH keys.

2. `Fork the silx project <https://github.com/silx-kit/silx/fork>`_.

3. Clone your GitHub repository on your local computer:

   .. code-block:: bash

      git clone git@github.com:<your_user_name>/silx
      cd silx

4. `Install silx for development`_.

5. `Run the tests`_ to make sure the silx test suite pass on your computer.

6. Open an issue in ``https://github.com/silx-kit/silx/issues`` to inform the
   maintainers of your intentions.

7. Create a local branch to start working on your issue: ``git branch my_feature``.

8. Code, enjoy but ensure that the new code is tested and does not break
   the current test suite.

9. Push your local branch to your GitHub account: ``git push origin my_feature``.

10. Create a pull request (PR) from your feature branch on GitHub to trigger
    the review process. Indicate this PR is related to the issue you opened in 6.
    Make sure to follow the `Pull Request title format`_.

11. Discuss with the maintainer who is reviewing your code using the GitHub interface.

If you encounter any problems or have any questions you can always ask on the `Issues page <https://github.com/silx-kit/silx/issues>`_.


Install silx for development
----------------------------

1. Install `build dependencies <https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html#build-dependencies>`_::

      pip install meson-python ninja cython

2. Install silx in `editable mode <https://peps.python.org/pep-0660/>`_ with the development dependencies::

      pip install --no-build-isolation --editable .[dev]

.. note::

    If the project "entry points" are modified, the project must be re-installed.

.. seealso::

    `Meson editable installs <https://mesonbuild.com/meson-python/how-to-guides/editable-installs.html>`_


Format the code
---------------

To format the code, use `black <https://black.readthedocs.io>`_::

    black .


Build the documentation
-----------------------

- `Install silx for development`_.
- From the silx project root folder, run `Sphinx <http://www.sphinx-doc.org/>`_::

    sphinx-build doc/source/ build/html

.. note::

    To re-generate the example script screenshots, build the documentation with the
    environment variable ``DIRECTIVE_SNAPSHOT_QT`` set to ``True``.


Run the tests
-------------

- `Install silx for development`_.
- From the silx project root folder, use `pytest <https://docs.pytest.org/en/stable/how-to/usage.html>`_:

.. warning::
     
     GUI tests are part of the complete test suite and will make windows appear and disappear very quickly.
     
      **Do not run these if you have a history of epilepsy or motion sickness** 
      
  * To run the complete test suite::

      pytest

  * To run a specfic test::

      pytest <src/silx/path/to/test_file.py::test_function>

To run the tests of an installed version of *silx*, run the following from the Python interpreter:

.. code-block:: python

     import silx.test
     silx.test.run_tests()


Pull Request title format
-------------------------

To ease release notes authoring, when creating a Pull Request (PR), please use the following syntax for the title::

  <Subpackage/Module/Topic>: <Action> <summary of the main change affecting silx's users>


With:

- **Subpackage/Topic**: One of:

  - A subpackage or a module: Use the fully qualified name of the subpackage or module of silx the PR is changing.
    For example: ``silx.gui.qt`` or ``silx.gui.plot.PlotWidget``.
  - A topic: If changes do not affect a particular subpackage or module, provide the topic of the change.
    This can be for example: ``Build``, ``Documentation``, ``CI``,... or the name of a silx application (e.g., ``silx view``).

- **Action**: How the changes affect the project from a silx user point of view.
  Prefer using one of the following actions:

  - **Added**: For new feature or new APIs
  - **Deprecated**
  - **Removed**
  - **Changed**
  - **Improved**
  - **Refactored**
  - **Fixed**

- **Summary**: A short description of the main change that will be included in the release notes.
