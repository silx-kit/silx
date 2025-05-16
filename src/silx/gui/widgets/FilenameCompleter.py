from pathlib import Path

from .. import qt


class FilenameCompleter(qt.QCompleter):
    """
    A QCompleter that provides autocompletion for file paths.
    """

    def __init__(
        self, parent: qt.QWidget | None = None, root: str | None = None
    ) -> None:
        super().__init__(parent=parent)

        completer = qt.QCompleter()
        model = qt.QFileSystemModel(completer)
        model.setOption(qt.QFileSystemModel.Option.DontWatchForChanges, True)
        if root is None:
            model.setRootPath(Path(__file__).root)

        completer.setModel(model)
        completer.setCompletionRole(qt.QFileSystemModel.Roles.FileNameRole)
