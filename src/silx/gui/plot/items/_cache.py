from __future__ import annotations


from collections import OrderedDict


class LRUCache:
    """
    Least Recently Used cache with a size limit.
    Pop the last element used (set or read) when reach teh cache limit size.

    Inspired by https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
    """

    def __init__(self, maxsize: int | None = 128):
        if maxsize is None and maxsize != -1:
            raise ValueError("cache max size should be -1 or higher than 0")
        self._maxsize = maxsize
        self._cache = OrderedDict()

    def __setitem__(self, key, value):
        self._cache[key] = value
        self._cache.move_to_end(key)

        if self._maxsize is not None and len(self) > self._maxsize:
            self._cache.popitem(last=False)

    def __getitem__(self, key):
        value = self._cache[key]
        self._cache.move_to_end(key)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    @property
    def maxsize(self) -> int | None:
        """
        Return cache maximal number of element kept. If None the cache has no size limit.
        """
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize: int | None) -> None:
        """
        Modify the number of elements kept in the cache.
        If -1 the cache has no size limit.

        .. warning: modifying the maximal size might affect the cache.
        """
        if maxsize is not None and maxsize < 0:
            raise ValueError("cache max size should be None or higher than 0")
        new_cache = OrderedDict()
        max_elmts_kepts = min(len(self._cache), maxsize if maxsize is not None else len(self._cache))
        # Preserve the most recently added element as the final entry in the cache to maintain state continuity between operations.
        items_to_copy = tuple(reversed(self._cache.items()))[:max_elmts_kepts]
        for key, value in reversed(items_to_copy):
            new_cache[key] = value
        self._cache = new_cache
        self._maxsize = maxsize

    # expose API
    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __iter__(self):
        return iter(self._cache)
