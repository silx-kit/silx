# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Test of validators"""


from silx.gui import qt
from .. import validators


def test_double__move_dot(qapp):
    """The initial number was 1.22 a '.' is typed at the end

    This moves the dot at another place.
    """
    v = validators.DoubleValidator()
    result = v.validate("1.22.", 5)
    assert result == (qt.QValidator.Acceptable, "122.", 4)


def test_double__type_a_group(qapp):
    """The groups can split the value (for example: `1,000,000.05`).

    Make sure a new typed group is removed.
    """
    v = validators.DoubleValidator()
    result = v.validate("1,1.2,", 6)
    assert result == (0, "1,1.2", 5)


def test_double__to_text(qapp):
    v = validators.DoubleValidator()
    assert v.toText(5.2) == '5.2'


def test_double__to_value(qapp):
    v = validators.DoubleValidator()
    assert v.toValue("5.2") == (5.2, True)


def test_double__to_wrong_value(qapp):
    v = validators.DoubleValidator()
    assert v.toValue("5.a")[1] == False


def test_double__fixup_groups(qapp):
    """The groups can split the value (for example: `1,000,000.05`).

    Test if it is removed.
    """
    v = validators.DoubleValidator()
    result = v.fixup("1,1.2,2")
    assert result == "11.22"


def test_advanceddouble__empty_as_none(qapp):
    """Check that None can be validated properly.
    """
    v = validators.AdvancedDoubleValidator()
    v.setAllowEmpty(True)

    assert v.validate("", 0) == (qt.QValidator.Acceptable, "", 0)
    assert v.toText(None) == ""
    assert v.toValue("") == (None, True)
    assert v.toValue("  ") == (None, True)


def test_advanceddouble__boundary(qapp):
    """Check that the boundaries can be included or excluded.
    """
    v = validators.AdvancedDoubleValidator()
    v.setIncludedBound(False, True)
    v.setRange(0, 100)
    assert v.toValue("0")[1] == False
    assert v.toValue("100")[1] == True
    assert v.validate("0", 1) == (qt.QValidator.Intermediate, "0", 1)
    assert v.validate("100", 3) == (qt.QValidator.Acceptable, "100", 3)


def test_pint__valid(qapp):
    v = validators.DoublePintValidator()
    result = v.validate("1.00 mm", 0)
    assert result == (qt.QValidator.Acceptable, "1.00 mm", 0)


def test_pint__intermediate(qapp):
    v = validators.DoublePintValidator()
    result = v.validate(".00 mm", 0)
    assert result == (qt.QValidator.Intermediate, ".00 mm", 0)


def test_pint__fixup_bare_dot(qapp):
    v = validators.DoublePintValidator()
    result = v.fixup(".00 mm")
    assert result == "0.00 mm"


def test_doublepint__move_dot(qapp):
    """The initial number was 1.22 a '.' is typed at the end

    This moves the dot at another place.
    """
    v = validators.DoublePintValidator()
    result = v.validate("1.22. mm", 5)
    assert result == (qt.QValidator.Acceptable, "122. mm", 4)


def test_doublepint__type_a_group(qapp):
    """The groups can split the value (for example: `1,000,000.05`).

    Make sure a new typed group is removed.
    """
    v = validators.DoublePintValidator()
    result = v.validate("1,1.2, mm", 6)
    assert result == (0, "1,1.2 mm", 5)


def test_doublepint__to_text(qapp):
    v = validators.DoublePintValidator()
    assert v.toText((5.2, "mm")) == '5.2 mm'


def test_doublepint__to_value(qapp):
    v = validators.DoublePintValidator()
    assert v.toValue("5.2") == ((5.2, ""), True)
    assert v.toValue("5.2 mm") == ((5.2, "mm"), True)


def test_doublepint__to_wrong_value(qapp):
    v = validators.DoublePintValidator()
    assert v.toValue("5.a")[1] == False


def test_doublepint__fixup_groups(qapp):
    """The groups can split the value (for example: `1,000,000.05`).

    Test if it is removed.
    """
    v = validators.DoublePintValidator()
    result = v.fixup("1,1.2,2 mm")
    assert result == "11.22 mm"


def test_advanceddoublepint__valid(qapp):
    """Check a valid value.
    """
    v = validators.AdvancedDoublePintValidator()
    v.setAllowEmpty(True)

    assert v.validate("5.0 mm", 0) == (qt.QValidator.Acceptable, "5.0 mm", 0)
    assert v.toText((5.0, "mm")) == "5.0 mm"
    assert v.toValue("5.0 mm") == ((5.0, "mm"), True)


def test_advanceddoublepint__empty_as_none(qapp):
    """Check that None can be validated properly.
    """
    v = validators.AdvancedDoublePintValidator()
    v.setAllowEmpty(True)

    assert v.validate("", 0) == (qt.QValidator.Acceptable, "", 0)
    assert v.toText((None, "mm")) == " mm"
    assert v.toValue("") == (None, True)
    assert v.toValue("  ") == (None, True)


def test_advanceddoublepint__boundary(qapp):
    """Check that the boundaries can be included or excluded.
    """
    v = validators.AdvancedDoublePintValidator()
    v.setIncludedBound(False, True)
    v.setRange(0, 100)
    assert v.toValue("0")[1] == False
    assert v.toValue("100")[1] == True
    assert v.validate("0", 1) == (qt.QValidator.Intermediate, "0", 1)
    assert v.validate("100", 3) == (qt.QValidator.Acceptable, "100", 3)
