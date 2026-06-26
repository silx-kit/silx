from dataclasses import dataclass
from typing import NamedTuple

import numpy

from silx._utils import NP_OPTIONAL_COPY


class MaskLayerInfo(NamedTuple):
    id: int
    name: str
    enabled: bool
    or_operation: bool


@dataclass
class _MaskLayer:
    id: int
    name: str
    mask: numpy.ndarray
    enabled: bool
    or_operation: bool

    def get_info(self) -> MaskLayerInfo:
        return MaskLayerInfo(
            id=self.id,
            name=self.name,
            enabled=self.enabled,
            or_operation=self.or_operation,
        )


class MaskLayers:
    """Manage a collection of uint8 mask layers.

    Supports undo/redo of enabling or disabling layers.
    Does not support undo/redo of modifying or removing layers.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = shape

        self._layers: list[_MaskLayer] = []
        self._next_id = 0

        self._merged_mask = empty_mask(shape)
        self._dirty = False

        # (layer_id, enabled)
        self._undo: list[tuple[int, bool]] = []
        self._redo: list[tuple[int, bool]] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def reset_history(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        while self.can_undo():
            layer_id, enabled = self._undo.pop(-1)
            new_enabled = not enabled

            try:
                self._get_layer(layer_id).enabled = new_enabled
            except KeyError:
                continue

            self._record_redo(layer_id, new_enabled)

            self._dirty = True
            return True

        return False

    def redo(self) -> bool:
        while self.can_redo():
            layer_id, enabled = self._redo.pop(-1)
            new_enabled = not enabled

            try:
                self._get_layer(layer_id).enabled = new_enabled
            except KeyError:
                continue

            self._record_undo(layer_id, new_enabled)

            self._dirty = True
            return True

        return False

    def _record_undo(self, layer_id: int, enabled: bool) -> None:
        self._undo.append((layer_id, enabled))

    def _record_redo(self, layer_id: int, enabled: bool) -> None:
        self._redo.append((layer_id, enabled))

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self._layers.clear()
        self._merged_mask.fill(0)
        self._dirty = False
        self.reset_history()

    def add_layer(
        self,
        mask: numpy.ndarray,
        copy: bool,
        name: str | None = None,
        enabled: bool = True,
        or_operation: bool = True,
    ) -> int:
        mask = _normalize_mask(mask, copy)

        if mask.shape != self._shape:
            raise ValueError(f"Expected shape {self._shape}, got {mask.shape}")

        layer_id = self._next_id
        self._next_id += 1

        self._layers.append(
            _MaskLayer(
                id=layer_id,
                name=name or f"Mask {layer_id}",
                mask=mask,
                enabled=enabled,
                or_operation=or_operation,
            )
        )

        self._dirty = True
        self._record_undo(layer_id, enabled)

        return layer_id

    def add_empty_layer(
        self, name: str | None = None, enabled: bool = True, or_operation: bool = True
    ) -> int:
        mask = empty_mask(self._shape)
        return self.add_layer(
            mask, copy=False, name=name, enabled=enabled, or_operation=or_operation
        )

    def remove_layer(self, layer_id: int) -> None:
        del self._layers[self._find_layer_index(layer_id)]
        self._dirty = True

    def set_layer_enabled(self, layer_id: int, enabled: bool) -> None:
        layer = self._get_layer(layer_id)

        if layer.enabled == enabled:
            return

        layer.enabled = enabled
        self._dirty = True
        self._record_undo(layer_id, enabled)

    def enable_layer(self, layer_id: int) -> None:
        self.set_layer_enabled(layer_id, True)

    def disable_layer(self, layer_id: int) -> None:
        self.set_layer_enabled(layer_id, False)

    def rename_layer(self, layer_id: int, name: str) -> None:
        self._get_layer(layer_id).name = name

    def update_layer(self, layer_id: int, mask: numpy.ndarray, copy: bool) -> None:
        if mask.shape != self._shape:
            raise ValueError(f"Expected shape {self._shape}, got {mask.shape}")

        layer = self._get_layer(layer_id)
        layer.mask = _normalize_mask(mask, copy)

        self._dirty = True

    def update_last_layer(self, mask: numpy.ndarray, copy: bool) -> None:
        if not self._layers:
            raise RuntimeError("No mask layers")

        self.update_layer(self._layers[-1].id, mask=mask, copy=copy)

    def get_layer(self, layer_id: int, copy: bool) -> numpy.ndarray:
        layer = self._get_layer(layer_id)
        return numpy.array(layer.mask, copy=copy)

    def get_last_layer(self, copy: bool) -> numpy.ndarray:
        if not self._layers:
            raise RuntimeError("No mask layers")

        return self.get_layer(self._layers[-1].id, copy=copy)

    def get_layer_info(self) -> list[MaskLayerInfo]:
        return [layer.get_info() for layer in self._layers]

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def get_merged_mask(self, copy: bool) -> numpy.ndarray:
        if self._dirty:
            self._rebuild_cache()

        return numpy.array(self._merged_mask, copy=copy)

    def squash(self, name: str | None = None) -> int:
        merged = self.get_merged_mask(copy=True)
        self.clear()
        return self.add_layer(mask=merged, copy=False, name=name)

    def _rebuild_cache(self) -> None:
        if not self._layers:
            self._merged_mask = empty_mask(self._shape)
            self._dirty = False
            return

        result: numpy.ndarray | None = None

        for layer in self._layers:
            if not layer.enabled:
                continue

            mask = layer.mask

            if result is None:
                result = numpy.array(mask, copy=True)
                continue

            if layer.or_operation:
                result |= mask
            else:
                result &= mask

        if result is None:
            self._merged_mask = empty_mask(self._shape)
        else:
            self._merged_mask = result

        self._dirty = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_layer_index(self, layer_id: int) -> int:
        for i, layer in enumerate(self._layers):
            if layer.id == layer_id:
                return i
        raise KeyError(layer_id)

    def _get_layer(self, layer_id: int) -> _MaskLayer:
        return self._layers[self._find_layer_index(layer_id)]


def empty_mask(shape: tuple[int, ...]) -> numpy.ndarray:
    """
    A uint8 C-contiguous array with zeros.
    """
    return numpy.zeros(shape, order="C", dtype=numpy.uint8)


def full_mask(shape: tuple[int, ...], value: int) -> numpy.ndarray:
    """
    A uint8 C-contiguous array with `value`.
    """
    assert_mask_value(value)
    return numpy.full(shape, value, order="C", dtype=numpy.uint8)


def stencil_mask(
    shape: tuple[int, ...], value: int, stencil: numpy.ndarray
) -> numpy.ndarray:
    """
    A uint8 C-contiguous array with `value` where `stencil==True` and 0 otherwise.
    """
    assert_mask_value(value)
    if stencil.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {stencil.shape}")
    stencil = numpy.asarray(stencil, dtype=bool)
    result = empty_mask(shape)
    result[stencil] = value
    return result


def assert_mask_value(value: int) -> None:
    if not (0 < value < 256):
        raise ValueError(f"value must be in range 1..255, got {value}")


def _normalize_mask(mask: numpy.ndarray, copy: bool) -> numpy.ndarray:
    """
    Convert mask to a uint8 C-contiguous array. Copy when needed or requested.
    """
    return numpy.array(
        mask, dtype=numpy.uint8, copy=copy or NP_OPTIONAL_COPY, order="C", subok=False
    )
