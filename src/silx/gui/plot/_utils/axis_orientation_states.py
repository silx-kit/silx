from typing import TypedDict

InversionState = bool


class AxisState(TypedDict):
    icon: str
    state: str
    action: str


Y_AXIS_STATE: dict[InversionState, AxisState] = {
    True: {
        "icon": "plot-ydown",
        "state": "Y-axis origin is at the top",
        "action": "Set Y-axis origin at the top",
    },
    False: {
        "icon": "plot-yup",
        "state": "Y-axis origin is at the bottom",
        "action": "Set Y-axis origin at the bottom",
    },
}

X_AXIS_STATE: dict[InversionState, AxisState] = {
    True: {
        "icon": "plot-xleft",
        "state": "X-axis origin is on the right",
        "action": "Set X-axis origin on the right",
    },
    False: {
        "icon": "plot-xright",
        "state": "X-axis origin is on the left",
        "action": "Set X-axis origin on the left",
    },
}
