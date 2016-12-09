#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""This script is a simple example of how to use the periodic table widgets,
select elements and connect signals.
"""
import sys
from silx.gui import qt
from silx.gui.widgets import PeriodicTable


#   Symbol  Atomic Number   col row ( positions on table )  name  subcategory
_elements = [
    ("H", 1, 1, 1, "hydrogen", "diatomic nonmetal"),
    ("He", 2, 18, 1, "helium", "noble gas"),
    ("Li", 3, 1, 2, "lithium", "alkali metal"),
    ("Be", 4, 2, 2, "beryllium", "alkaline earth metal"),
    ("B", 5, 13, 2, "boron", "metalloid"),
    ("C", 6, 14, 2, "carbon", "polyatomic nonmetal"),
    ("N", 7, 15, 2, "nitrogen", "diatomic nonmetal"),
    ("O", 8, 16, 2, "oxygen", "diatomic nonmetal"),
    ("F", 9, 17, 2, "fluorine", "diatomic nonmetal"),
    ("Ne", 10, 18, 2, "neon", "noble gas"),
    ("Na", 11, 1, 3, "sodium", "alkali metal"),
    ("Mg", 12, 2, 3, "magnesium", "alkaline earth metal"),
    ("Al", 13, 13, 3, "aluminium", "post transition metal"),
    ("Si", 14, 14, 3, "silicon", "metalloid"),
    ("P", 15, 15, 3, "phosphorus", "polyatomic nonmetal"),
    ("S", 16, 16, 3, "sulphur", "polyatomic nonmetal"),
    ("Cl", 17, 17, 3, "chlorine", "diatomic nonmetal"),
    ("Ar", 18, 18, 3, "argon", "noble gas"),
    ("K", 19, 1, 4, "potassium", "alkali metal"),
    ("Ca", 20, 2, 4, "calcium", "alkaline earth metal"),
    ("Sc", 21, 3, 4, "scandium", "transition metal"),
    ("Ti", 22, 4, 4, "titanium", "transition metal"),
    ("V", 23, 5, 4, "vanadium", "transition metal"),
    ("Cr", 24, 6, 4, "chromium", "transition metal"),
    ("Mn", 25, 7, 4, "manganese", "transition metal"),
    ("Fe", 26, 8, 4, "iron", "transition metal"),
    ("Co", 27, 9, 4, "cobalt", "transition metal"),
    ("Ni", 28, 10, 4, "nickel", "transition metal"),
    ("Cu", 29, 11, 4, "copper", "transition metal"),
    ("Zn", 30, 12, 4, "zinc", "transition metal"),
    ("Ga", 31, 13, 4, "gallium", "post transition metal"),
    ("Ge", 32, 14, 4, "germanium", "metalloid"),
    ("As", 33, 15, 4, "arsenic", "metalloid"),
    ("Se", 34, 16, 4, "selenium", "polyatomic nonmetal"),
    ("Br", 35, 17, 4, "bromine", "diatomic nonmetal"),
    ("Kr", 36, 18, 4, "krypton", "noble gas"),
    ("Rb", 37, 1, 5, "rubidium", "alkali metal"),
    ("Sr", 38, 2, 5, "strontium", "alkaline earth metal"),
    ("Y", 39, 3, 5, "yttrium", "transition metal"),
    ("Zr", 40, 4, 5, "zirconium", "transition metal"),
    ("Nb", 41, 5, 5, "niobium", "transition metal"),
    ("Mo", 42, 6, 5, "molybdenum", "transition metal"),
    ("Tc", 43, 7, 5, "technetium", "transition metal"),
    ("Ru", 44, 8, 5, "ruthenium", "transition metal"),
    ("Rh", 45, 9, 5, "rhodium", "transition metal"),
    ("Pd", 46, 10, 5, "palladium", "transition metal"),
    ("Ag", 47, 11, 5, "silver", "transition metal"),
    ("Cd", 48, 12, 5, "cadmium", "transition metal"),
    ("In", 49, 13, 5, "indium", "post transition metal"),
    ("Sn", 50, 14, 5, "tin", "post transition metal"),
    ("Sb", 51, 15, 5, "antimony", "metalloid"),
    ("Te", 52, 16, 5, "tellurium", "metalloid"),
    ("I", 53, 17, 5, "iodine", "diatomic nonmetal"),
    ("Xe", 54, 18, 5, "xenon", "noble gas"),
    ("Cs", 55, 1, 6, "caesium", "alkali metal"),
    ("Ba", 56, 2, 6, "barium", "alkaline earth metal"),
    ("La", 57, 3, 6, "lanthanum", "lanthanide"),
    ("Ce", 58, 4, 9, "cerium", "lanthanide"),
    ("Pr", 59, 5, 9, "praseodymium", "lanthanide"),
    ("Nd", 60, 6, 9, "neodymium", "lanthanide"),
    ("Pm", 61, 7, 9, "promethium", "lanthanide"),
    ("Sm", 62, 8, 9, "samarium", "lanthanide"),
    ("Eu", 63, 9, 9, "europium", "lanthanide"),
    ("Gd", 64, 10, 9, "gadolinium", "lanthanide"),
    ("Tb", 65, 11, 9, "terbium", "lanthanide"),
    ("Dy", 66, 12, 9, "dysprosium", "lanthanide"),
    ("Ho", 67, 13, 9, "holmium", "lanthanide"),
    ("Er", 68, 14, 9, "erbium", "lanthanide"),
    ("Tm", 69, 15, 9, "thulium", "lanthanide"),
    ("Yb", 70, 16, 9, "ytterbium", "lanthanide"),
    ("Lu", 71, 17, 9, "lutetium", "lanthanide"),
    ("Hf", 72, 4, 6, "hafnium", "transition metal"),
    ("Ta", 73, 5, 6, "tantalum", "transition metal"),
    ("W", 74, 6, 6, "tungsten", "transition metal"),
    ("Re", 75, 7, 6, "rhenium", "transition metal"),
    ("Os", 76, 8, 6, "osmium", "transition metal"),
    ("Ir", 77, 9, 6, "iridium", "transition metal"),
    ("Pt", 78, 10, 6, "platinum", "transition metal"),
    ("Au", 79, 11, 6, "gold", "transition metal"),
    ("Hg", 80, 12, 6, "mercury", "transition metal"),
    ("Tl", 81, 13, 6, "thallium", "post transition metal"),
    ("Pb", 82, 14, 6, "lead", "post transition metal"),
    ("Bi", 83, 15, 6, "bismuth", "post transition metal"),
    ("Po", 84, 16, 6, "polonium", "post transition metal"),
    ("At", 85, 17, 6, "astatine", "metalloid"),
    ("Rn", 86, 18, 6, "radon", "noble gas"),
    ("Fr", 87, 1, 7, "francium", "alkali metal"),
    ("Ra", 88, 2, 7, "radium", "alkaline earth metal"),
    ("Ac", 89, 3, 7, "actinium", "actinide"),
    ("Th", 90, 4, 10, "thorium", "actinide"),
    ("Pa", 91, 5, 10, "proactinium", "actinide"),
    ("U", 92, 6, 10, "uranium", "actinide"),
    ("Np", 93, 7, 10, "neptunium", "actinide"),
    ("Pu", 94, 8, 10, "plutonium", "actinide"),
    ("Am", 95, 9, 10, "americium", "actinide"),
    ("Cm", 96, 10, 10, "curium", "actinide"),
    ("Bk", 97, 11, 10, "berkelium", "actinide"),
    ("Cf", 98, 12, 10, "californium", "actinide"),
    ("Es", 99, 13, 10, "einsteinium", "actinide"),
    ("Fm", 100, 14, 10, "fermium", "actinide"),
    ("Md", 101, 15, 10, "mendelevium", "actinide"),
    ("No", 102, 16, 10, "nobelium", "actinide"),
    ("Lr", 103, 17, 10, "lawrencium", "actinide"),
    ("Rf", 104, 4, 7, "rutherfordium", "transition metal"),
    ("Db", 105, 5, 7, "dubnium", "transition metal"),
    ("Sg", 106, 6, 7, "seaborgium", "transition metal"),
    ("Bh", 107, 7, 7, "bohrium", "transition metal"),
    ("Hs", 108, 8, 7, "hassium", "transition metal"),
    ("Mt", 109, 9, 7, "meitnerium")
]

# Named colors may not be available system
# In that case, use RGB hexadecimal code instead
COLORS = {
    "diatomic nonmetal": qt.QColor("chartreuse"),       #7FFF00
    "noble gas": qt.QColor("cyan"),                     #00FFFF
    "alkali metal": qt.QColor("Moccasin"),              #FFE4B5
    "alkaline earth metal": qt.QColor("orange"),        #FFA500
    "polyatomic nonmetal": qt.QColor("aquamarine"), 	#7FFFD4
    "transition metal": qt.QColor("light salmon"),      #FFA07A
    "metalloid": qt.QColor("Dark Sea Green"),           #8FBC8F
    "post transition metal": qt.QColor("light gray"), 	#D3D3D3
    "lanthanide": qt.QColor("light pink"),              #FFB6C1
    "actinide": qt.QColor("Light Coral"),               #F08080
    "": qt.QColor("white"),
}


class MyPeriodicTableItem(PeriodicTable.PeriodicTableItem):
    """Periodic table item without mass but with
    subcategory.
    Colors are assigned based on subcategory.
    """
    def __init__(self, symbol, Z, col, row, name,
                 subcategory=""):
        """
        :param str symbol: Atomic symbol (e.g. H, He, Li...)
        :param int Z: Proton number
        :param int col: 1-based column index of element in periodic table
        :param int row: 1-based row index of element in periodic table
        :param str name: PeriodicTableItem name ("hydrogen", ...)
        :param str subcategory: Elements category, (e.g "noble gas"...)
        """
        # standard element without specifying mass and color
        PeriodicTable.PeriodicTableItem.__init__(
                self, symbol, Z, col, row, name, mass=None, bgcolor=None)
        self.subcategory = subcategory
        # redefine color using category and association dict
        self.bgcolor = COLORS[subcategory]


my_elements = [MyPeriodicTableItem(*info) for info in _elements]

a = qt.QApplication(sys.argv)
a.lastWindowClosed.connect(a.quit)

w = qt.QTabWidget()

pt = PeriodicTable.PeriodicTable(w, selectable=True,
                                 elements=my_elements)
pc = PeriodicTable.PeriodicCombo(w)
pl = PeriodicTable.PeriodicList(w)

pt.setSelection(['H', 'Fe', 'Si'])
pl.setSelectedElements(['H', 'Be', 'F'])
pc.setSelection("Li")


def change_list(items):
    print("New list selection:", [item.symbol for item in items])


def change_combo(item):
    print("New combo selection:", item.symbol)


def click_table(item):
    print("New table click: %s (%s)" % (item.name, item.subcategory))


def change_table(items):
    print("New table selection:", [item.symbol for item in items])


pt.sigElementClicked.connect(click_table)
pt.sigSelectionChanged.connect(change_table)
pl.sigSelectionChanged.connect(change_list)
pc.sigSelectionChanged.connect(change_combo)

# move combo into container widget to preventing it from filling
# the tab inside TabWidget
comboContainer = qt.QWidget(w)
comboContainer.setLayout(qt.QVBoxLayout())
comboContainer.layout().addWidget(pc)

w.addTab(pt, "PeriodicTable")
w.addTab(pl, "PeriodicList")
w.addTab(comboContainer, "PeriodicCombo")
w.show()

a.exec_()

