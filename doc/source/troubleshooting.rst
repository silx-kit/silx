Trouble shooting
================

Using OpenGL through ssh
------------------------

Some widgets in :mod:`silx.gui` are using OpenGL2.1:

- Widgets in :mod:`silx.gui.plot3d`, and
- The OpenGL backend of :class:`~silx.gui.plot.PlotWidget` and related widgets in :mod:`silx.gui.plot`.

When running applications based on OpenGL2.1 through ssh, there are a few situations that can prevent the display of OpenGL widgets:

- Make sure to use ``ssh -X`` to enable X11 forwarding.
- OpenGL is disabled with X11 forwarding (the default on Debian 8 and 9). See `Enabling OpenGL forwarding`_.
- Unless the operating system is using `libglvnd <https://github.com/NVIDIA/libglvnd/releases>`_
  (available from Debian 9 backports onward),
  both the server and the client computers must have the same kind of GPU drivers
  (either both using proprietary NVidia drivers or both using open source drivers),
  otherwise only OpenGL1.4 is available.

To get the currently available version of OpenGL, run from the command line::

  glxinfo | grep "OpenGL version string"

On Debian, ``glxinfo`` is available as part of the ``mesa-utils`` package.

Enabling OpenGL forwarding
..........................

"Indirect GLX" must be enabled on the local computer.
If it is disabled, setting it up requires root access.
The way to set it up depends on the configuration of the system (the operating system and the display manager).

- On Debian 8 with kdm display manager, add ``+iglx`` after ``ServerArgsLocal=...`` in ``/etc/kde4/kdm/kdmrc`` and restart the X server.
- On Debian 9 with sddm display manager, dd ``+iglx`` after ``ServerArguments=...``` in ``/etc/sddm.conf`` and restart the X server.
