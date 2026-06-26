from .. import mask as mask_mod


def test_undo_redo():
    shape = (10, 11)
    layers = mask_mod.MaskLayers(shape)

    # Add Mask 0 and Mask 1
    layers.add_empty_layer()
    layers.add_empty_layer()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Undo Mask 1
    layers.undo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Undo Mask 0
    layers.undo()
    excepted = [
        (0, "Mask 0", False, True),
        (1, "Mask 1", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Nothing to undo
    layers.undo()
    excepted = [
        (0, "Mask 0", False, True),
        (1, "Mask 1", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Redo Mask 0
    layers.redo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Redo Mask 1
    layers.redo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Nothing to redo
    layers.redo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
    ]
    _assert_layer_info(layers, excepted)


def test_undo_redo_after_add():
    shape = (10, 11)
    layers = mask_mod.MaskLayers(shape)

    # Add Mask 0 and Mask 1
    layers.add_empty_layer()
    layers.add_empty_layer()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Undo Mask 1
    layers.undo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Add Mask 2
    layers.add_empty_layer()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
        (2, "Mask 2", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Undo Mask 2
    layers.undo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
        (2, "Mask 2", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Redo Mask 2
    layers.redo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", False, True),
        (2, "Mask 2", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Redo Mask 1
    layers.redo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
        (2, "Mask 2", True, True),
    ]
    _assert_layer_info(layers, excepted)


def test_undo_redo_after_remove():
    shape = (10, 11)
    layers = mask_mod.MaskLayers(shape)

    # Add Mask 0, Mask 1 and Mask 2
    layers.add_empty_layer()
    layers.add_empty_layer()
    layers.add_empty_layer()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
        (2, "Mask 2", True, True),
    ]
    _assert_layer_info(layers, excepted)

    # Undo Mask 2
    layers.undo()
    excepted = [
        (0, "Mask 0", True, True),
        (1, "Mask 1", True, True),
        (2, "Mask 2", False, True),
    ]
    _assert_layer_info(layers, excepted)

    # Delete Mask 1 and undo Mask 0
    layers.remove_layer(1)
    layers.undo()
    excepted = [
        (0, "Mask 0", False, True),
        (2, "Mask 2", False, True),
    ]
    _assert_layer_info(layers, excepted)


def _assert_layer_info(layers: mask_mod.MaskLayers, expected: list[tuple]):
    actual = [tuple(info) for info in layers.get_layer_info()]
    assert actual == expected, actual
