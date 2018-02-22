# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2018 European Synchrotron Radiation Facility
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
"""This module provides an implementation of state machines for interaction.

Sample code of a state machine with two states ('idle' and 'active')
with transitions on left button press/release:

.. code-block:: python

   from silx.gui.plot.Interaction import *

   class SampleStateMachine(StateMachine):

       class Idle(State):
           def onPress(self, x, y, btn):
               if btn == LEFT_BTN:
                   self.goto('active')

       class Active(State):
           def enterState(self):
               print('Enabled')  # Handle enter active state here

           def leaveState(self):
               print('Disabled')  # Handle leave active state here

           def onRelease(self, x, y, btn):
               if btn == LEFT_BTN:
                   self.goto('idle')

   def __init__(self):
       # State machine has 2 states
       states = {
           'idle': SampleStateMachine.Idle,
           'active': SampleStateMachine.Active
       }
       super(TwoStates, self).__init__(states, 'idle')
       # idle is the initial state

   stateMachine = SampleStateMachine()

   # Triggers a transition to the Active state:
   stateMachine.handleEvent('press', 0, 0, LEFT_BTN)

   # Triggers a transition to the Idle state:
   stateMachine.handleEvent('release', 0, 0, LEFT_BTN)

See :class:`ClickOrDrag` for another example of a state machine.

See `Renaud Blanch, Michel Beaudouin-Lafon.
Programming Rich Interactions using the Hierarchical State Machine Toolkit.
In Proceedings of AVI 2006. p 51-58.
<http://iihm.imag.fr/en/publication/BB06a/>`_
for a discussion of using (hierarchical) state machines for interaction.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "18/02/2016"


import weakref


# state machine ###############################################################

class State(object):
    """Base class for the states of a state machine.

    This class is meant to be subclassed.
    """

    def __init__(self, machine):
        """State instances should be created by the :class:`StateMachine`.

        They are not intended to be used outside this context.

        :param machine: The state machine instance this state belongs to.
        :type machine: StateMachine
        """
        self._machineRef = weakref.ref(machine)  # Prevent cyclic reference

    @property
    def machine(self):
        """The state machine this state belongs to.

        Useful to access data or methods that are shared across states.
        """
        machine = self._machineRef()
        if machine is not None:
            return machine
        else:
            raise RuntimeError("Associated StateMachine is not valid")

    def goto(self, state, *args, **kwargs):
        """Performs a transition to a new state.

        Extra arguments are passed to the :meth:`enterState` method of the
        new state.

        :param str state: The name of the state to go to.
        """
        self.machine._goto(state, *args, **kwargs)

    def enterState(self, *args, **kwargs):
        """Called when the state machine enters this state.

        Arguments are those provided to the :meth:`goto` method that
        triggered the transition to this state.
        """
        pass

    def leaveState(self):
        """Called when the state machine leaves this state
        (i.e., when :meth:`goto` is called).
        """
        pass


class StateMachine(object):
    """State machine controller.

    This is the entry point of a state machine.
    It is in charge of dispatching received event and handling the
    current active state.
    """

    def __init__(self, states, initState, *args, **kwargs):
        """Create a state machine controller with an initial state.

        Extra arguments are passed to the :meth:`enterState` method
        of the initState.

        :param states: All states of the state machine
        :type states: dict of: {str name: State subclass}
        :param str initState: Key of the initial state in states
        """
        self.states = states

        self.state = self.states[initState](self)
        self.state.enterState(*args, **kwargs)

    def _goto(self, state, *args, **kwargs):
        self.state.leaveState()
        self.state = self.states[state](self)
        self.state.enterState(*args, **kwargs)

    def handleEvent(self, eventName, *args, **kwargs):
        """Process an event with the state machine.

        This method looks up for an event handler in the current state
        and then in the :class:`StateMachine` instance.
        Handler are looked up as 'onEventName' method.
        If a handler is found, it is called with the provided extra
        arguments, and this method returns the return value of the
        handler.
        If no handler is found, this method returns None.

        :param str eventName: Name of the event to handle
        :returns: The return value of the handler or None
        """
        handlerName = 'on' + eventName[0].upper() + eventName[1:]
        try:
            handler = getattr(self.state, handlerName)
        except AttributeError:
            try:
                handler = getattr(self, handlerName)
            except AttributeError:
                handler = None
        if handler is not None:
            return handler(*args, **kwargs)


# clickOrDrag #################################################################

LEFT_BTN = 'left'
"""Left mouse button."""

RIGHT_BTN = 'right'
"""Right mouse button."""

MIDDLE_BTN = 'middle'
"""Middle mouse button."""


class ClickOrDrag(StateMachine):
    """State machine for left and right click and left drag interaction.

    It is intended to be used through subclassing by overriding
    :meth:`click`, :meth:`beginDrag`, :meth:`drag` and :meth:`endDrag`.
    """

    DRAG_THRESHOLD_SQUARE_DIST = 5 ** 2

    class Idle(State):
        def onPress(self, x, y, btn):
            if btn == LEFT_BTN:
                self.goto('clickOrDrag', x, y)
                return True
            elif btn == RIGHT_BTN:
                self.goto('rightClick', x, y)
                return True

    class RightClick(State):
        def onMove(self, x, y):
            self.goto('idle')

        def onRelease(self, x, y, btn):
            if btn == RIGHT_BTN:
                self.machine.click(x, y, btn)
                self.goto('idle')

    class ClickOrDrag(State):
        def enterState(self, x, y):
            self.initPos = x, y

        def onMove(self, x, y):
            dx2 = (x - self.initPos[0]) ** 2
            dy2 = (y - self.initPos[1]) ** 2
            if (dx2 + dy2) >= self.machine.DRAG_THRESHOLD_SQUARE_DIST:
                self.goto('drag', self.initPos, (x, y))

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.click(x, y, btn)
                self.goto('idle')

    class Drag(State):
        def enterState(self, initPos, curPos):
            self.initPos = initPos
            self.machine.beginDrag(*initPos)
            self.machine.drag(*curPos)

        def onMove(self, x, y):
            self.machine.drag(x, y)

        def onRelease(self, x, y, btn):
            if btn == LEFT_BTN:
                self.machine.endDrag(self.initPos, (x, y))
                self.goto('idle')

    def __init__(self):
        states = {
            'idle': ClickOrDrag.Idle,
            'rightClick': ClickOrDrag.RightClick,
            'clickOrDrag': ClickOrDrag.ClickOrDrag,
            'drag': ClickOrDrag.Drag
        }
        super(ClickOrDrag, self).__init__(states, 'idle')

    def click(self, x, y, btn):
        """Called upon a left or right button click.

        To override in a subclass.
        """
        pass

    def beginDrag(self, x, y):
        """Called at the beginning of a drag gesture with left button
        pressed.

        To override in a subclass.
        """
        pass

    def drag(self, x, y):
        """Called on mouse moved during a drag gesture.

        To override in a subclass.
        """
        pass

    def endDrag(self, startPoint, endPoint):
        """Called at the end of a drag gesture when the left button is
        released.

        To override in a subclass.
        """
        pass
