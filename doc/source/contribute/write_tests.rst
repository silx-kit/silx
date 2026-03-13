Write tests
===========

Test framework
--------------

Tests are written using `pytest <https://docs.pytest.org/en/stable/>`_.

Some historical tests are still using  ``unittest`` but **please use** ``pytest`` **for new tests**.

Test location
-------------

Tests should written in a separate file with a name starting with ``test_``. The name should also mention the tested module. 

This file should be placed a ``test`` subfolder placed in the folder containing the tested code.

.. admonition:: Example

    If testing ``src/silx/io/url.py``, place tests in a file called ``test_url.py`` in the ``src/silx/io/test`` folder.


Test types
----------

To know more about the different test types, see the `testing strategy section <https://gitlab.esrf.fr/scisoft/how-we-code/-/blob/main/code_of_conduct.md?ref_type=heads#5-testing-strategy>`_ of the *How we code* document.