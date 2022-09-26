# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""This module provides the :func:`getQEventName` utility function."""

from silx.gui import qt


QT_EVENT_NAMES = {
    0: "None",
    114: "ActionAdded",
    113: "ActionChanged",
    115: "ActionRemoved",
    99: "ActivationChange",
    121: "ApplicationActivate",
    # ApplicationActivate: "ApplicationActivated",
    122: "ApplicationDeactivate",
    36: "ApplicationFontChange",
    37: "ApplicationLayoutDirectionChange",
    38: "ApplicationPaletteChange",
    214: "ApplicationStateChange",
    35: "ApplicationWindowIconChange",
    68: "ChildAdded",
    69: "ChildPolished",
    71: "ChildRemoved",
    40: "Clipboard",
    19: "Close",
    200: "CloseSoftwareInputPanel",
    178: "ContentsRectChange",
    82: "ContextMenu",
    183: "CursorChange",
    52: "DeferredDelete",
    60: "DragEnter",
    62: "DragLeave",
    61: "DragMove",
    63: "Drop",
    170: "DynamicPropertyChange",
    98: "EnabledChange",
    10: "Enter",
    150: "EnterEditFocus",
    124: "EnterWhatsThisMode",
    206: "Expose",
    116: "FileOpen",
    8: "FocusIn",
    9: "FocusOut",
    23: "FocusAboutToChange",
    97: "FontChange",
    198: "Gesture",
    202: "GestureOverride",
    188: "GrabKeyboard",
    186: "GrabMouse",
    159: "GraphicsSceneContextMenu",
    164: "GraphicsSceneDragEnter",
    166: "GraphicsSceneDragLeave",
    165: "GraphicsSceneDragMove",
    167: "GraphicsSceneDrop",
    163: "GraphicsSceneHelp",
    160: "GraphicsSceneHoverEnter",
    162: "GraphicsSceneHoverLeave",
    161: "GraphicsSceneHoverMove",
    158: "GraphicsSceneMouseDoubleClick",
    155: "GraphicsSceneMouseMove",
    156: "GraphicsSceneMousePress",
    157: "GraphicsSceneMouseRelease",
    182: "GraphicsSceneMove",
    181: "GraphicsSceneResize",
    168: "GraphicsSceneWheel",
    18: "Hide",
    27: "HideToParent",
    127: "HoverEnter",
    128: "HoverLeave",
    129: "HoverMove",
    96: "IconDrag",
    101: "IconTextChange",
    83: "InputMethod",
    207: "InputMethodQuery",
    169: "KeyboardLayoutChange",
    6: "KeyPress",
    7: "KeyRelease",
    89: "LanguageChange",
    90: "LayoutDirectionChange",
    76: "LayoutRequest",
    11: "Leave",
    151: "LeaveEditFocus",
    125: "LeaveWhatsThisMode",
    88: "LocaleChange",
    176: "NonClientAreaMouseButtonDblClick",
    174: "NonClientAreaMouseButtonPress",
    175: "NonClientAreaMouseButtonRelease",
    173: "NonClientAreaMouseMove",
    177: "MacSizeChange",
    43: "MetaCall",
    102: "ModifiedChange",
    4: "MouseButtonDblClick",
    2: "MouseButtonPress",
    3: "MouseButtonRelease",
    5: "MouseMove",
    109: "MouseTrackingChange",
    13: "Move",
    197: "NativeGesture",
    208: "OrientationChange",
    12: "Paint",
    39: "PaletteChange",
    131: "ParentAboutToChange",
    21: "ParentChange",
    212: "PlatformPanel",
    217: "PlatformSurface",
    75: "Polish",
    74: "PolishRequest",
    123: "QueryWhatsThis",
    106: "ReadOnlyChange",
    199: "RequestSoftwareInputPanel",
    14: "Resize",
    204: "ScrollPrepare",
    205: "Scroll",
    117: "Shortcut",
    51: "ShortcutOverride",
    17: "Show",
    26: "ShowToParent",
    50: "SockAct",
    192: "StateMachineSignal",
    193: "StateMachineWrapped",
    112: "StatusTip",
    100: "StyleChange",
    87: "TabletMove",
    92: "TabletPress",
    93: "TabletRelease",
    171: "TabletEnterProximity",
    172: "TabletLeaveProximity",
    219: "TabletTrackingChange",
    22: "ThreadChange",
    1: "Timer",
    120: "ToolBarChange",
    110: "ToolTip",
    184: "ToolTipChange",
    194: "TouchBegin",
    209: "TouchCancel",
    196: "TouchEnd",
    195: "TouchUpdate",
    189: "UngrabKeyboard",
    187: "UngrabMouse",
    78: "UpdateLater",
    77: "UpdateRequest",
    111: "WhatsThis",
    118: "WhatsThisClicked",
    31: "Wheel",
    132: "WinEventAct",
    24: "WindowActivate",
    103: "WindowBlocked",
    25: "WindowDeactivate",
    34: "WindowIconChange",
    105: "WindowStateChange",
    33: "WindowTitleChange",
    104: "WindowUnblocked",
    203: "WinIdChange",
    126: "ZOrderChange",
    65535: "MaxUser",
}


def getQEventName(eventType):
    """
    Returns the name of a QEvent.

    :param Union[int,qt.QEvent] eventType: A QEvent or a QEvent type.
    :returns: str
    """
    if isinstance(eventType, qt.QEvent):
        eventType = eventType.type()
    if 1000 <= eventType <= 65535:
        return "User_%d" % eventType
    name = QT_EVENT_NAMES.get(eventType, None)
    if name is not None:
        return name
    return "Unknown_%d" % eventType
