How to contribute
=================


Development process
-------------------

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

      pytest <src/silx/path/to/test_file.py>  # or
      pytest --pyargs <silx.subpackage.test.test_module>

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


How-to make a release
---------------------

Use cases
+++++++++

The `release branch` is the ``main`` branch, except for bug fix releases.

First, decide which kind of release is needed:

Release candidates
..................

Use this release process and make sure only bug fix pull requests are merged on the ``main`` branch until the final release is published.

Major/minor releases
....................

Follow this release process.

Bug fix releases
................

- For ``vM.m.1``, first create a ``M.m`` branch from the ``vM.m.0`` tag when not already done.
- Merge relevant pull requests on the ``M.m`` branch.
- Follow this release process, but with release branch ``M.m`` instead of ``main``.

Pre-release testing
+++++++++++++++++++

Automated tests
...............

- Run the `release workflow`_ called "Build and deploy" manually on the release branch (see `manually running a workflow`_).
  This is to verify that the release artifacts are built correctly.
  The wheels generated by the workflow can be downloaded from the release workflow run web page.
  *Note: Running the workflow manually does NOT publish artifacts to pypi.*

- Run the `bob workflow`_ with the following variables:
  - ``REPOSITORY``: ``https://github.com/<user>/silx`` (default: ``https://github.com/silx-kit/silx``)
  - ``TAG``: branch or tag to test (default: ``main`` branch)

  These tests take a long time. You can move to the *Prepare the release* section in the meantime.

Manual testing
..............

Download wheels generated by the github release workflow from the github action release workflow web page and install silx
from those wheels locally for manual testing.

Prepare the release
+++++++++++++++++++

Write the release notes
.......................

- Generate the list of pull requests included in the release with github's automatically generated release notes
  (see `github automatically generated release notes`_) between a new tag and the previous release.
- Copy the generated changelog to ``CHANGELOG.rst`` and close github's release web page.
  **Warning: DO NOT publish the release yet!**
- Sort, curate and fix the list of PRs and match the styling of previous release notes. You can run ``tools/format_GH_release_notes.py``
  first, that will format the GH release notes in `CHANGELOG_new.rst`. 

Steps
.....

- Create a branch from the release branch.
- Update ``CHANGELOG.rst``.
- Bump the version number in ``src/silx/_version.py``.
- Create a pull request to the release branch with those changes, wait for reviews and merge it.

Publish the release
+++++++++++++++++++

Create the release
..................

* Draft a new release from `github new release page`_ using similar conventions as previous releases:
  - Create a new tag which **MUST** be named ``v<release_version>`` and match the version in ``src/silx/_version.py``.
  - Select the release branch as the target.
  - Combine the release notes manually edited from ``CHANGELOG.rst`` with `github automatically generated release notes`_.
* Press the "Publish release" button to push the new tag to the release branch and trigger the release workflow which builds
  the documentation, the source tarball, the wheels and the Windows "fat binaries" of the release. You should see them starting from the `actions page <https://github.com/silx-kit/silx/actions>`_. 

.. note::

  If any step in the release process (such as creating wheels or building documentation) fails, you can cancel the github workflow, delete the github release and the associated tag.
  Then add new PR(s) and repeat the release operation.

Publish Windows "fat binaries"
..............................

Once Windows "fat binaries" are built and tested, the release workflow requests the approval from a reviewer of the "assets" `deployment environment`_.
Upon approval, the following files are added to the github release assets:

- ``silx-<release_version>-windows-application.zip``
- ``silx-<release_version>-windows-installer-x86_64.exe``

Publish to pypi
...............

Once build and tests are completed, the release workflow requests the approval from a reviewer of the "pypi" `deployment environment`_.
Upon approval, the release artifacts are published to `pypi`_.

Deploy the documentation
........................

Skip this step for **release candidates**.

- Download the ``documentation`` artifact from the release workflow run web page.
- Unzip it in the ``doc/silx/M.m.p`` folder on www.silx.org/doc/silx.
- Update the ``doc/silx/latest`` symbolic link.

Publish on conda-forge
......................

Skip this step for **release candidates**.

Shortly after the publication on `pypi`_, conda-forge bot opens a PR on the `silx feedstock`_ to add this version to the conda-forge channel.
Once this PR is merged, the new version is published on conda-forge.

.. _release workflow: https://github.com/silx-kit/silx/actions/workflows/release.yml
.. _manually running a workflow: https://docs.github.com/en/actions/using-workflows/manually-running-a-workflow
.. _github new release page: https://github.com/silx-kit/silx/releases/new
.. _github automatically generated release notes: https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes#creating-automatically-generated-release-notes-for-a-new-release
.. _bob workflow: https://gitlab.esrf.fr/silx/bob/silx/-/pipelines/new
.. _deployment environment: https://github.com/silx-kit/silx/settings/environments
.. _pypi: https://pypi.org/project/silx/
.. _silx feedstock: https://github.com/conda-forge/silx-feedstock
