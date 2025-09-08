from __future__ import annotations


from collections import OrderedDict


import contextlib as _contextlib


@_contextlib.contextmanager
def _poping_signal(cache: LRUCache):
    old = cache._is_poping_item
    cache._is_poping_item = True
    try:
        yield
    finally:
        cache._is_poping_item = old


class LRUCache(OrderedDict):
    """
    Least Recently Used cache with a size limit.
    Pop the last element used (set or read) when reach teh cache limit size.

    Inspired by https://docs.python.org/3/library/collections.html#ordereddict-examples-and-recipes
    """

    def __init__(self, maxsize: int = 128):
        if maxsize < 1:
            raise ValueError("cache max size should be higher than 0")
        self._maxsize = maxsize
        self._is_poping_item = False
        # needed for python 3.10 because the 'popitem' is calling getitem. Not on upper python version

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

        self.move_to_end(key)

        if len(self) > self._maxsize:
            with _poping_signal(self):
                self.popitem(last=False)
            # this functions call the __getitem__ in python 3.10... which is an issue

    def __getitem__(self, key):
        if not self._is_poping_item:
            self.move_to_end(key)
        return super().__getitem__(key)
