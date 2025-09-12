from __future__ import annotations


from collections import OrderedDict


class LRUCache:
    """
    Least Recently Used cache with a size limit.
    Pop the last element used (set or read) when reach teh cache limit size.

    Inspired by https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
    """

    def __init__(self, maxsize: int = 128):
        if maxsize < 1:
            raise ValueError("cache max size should be higher than 0")
        self._maxsize = maxsize
        self._cache = OrderedDict()

    def __setitem__(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)

        if len(self) > self._maxsize:
            self._cache.popitem(last=False)

    def __getitem__(self, key):
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    # expose API
    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self):
        return iter(self._cache)
