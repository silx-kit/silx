from ..cache import LRUCache


def test_LRU_CACHE():
    """Test the 'LRUCache' class."""
    cache = LRUCache(maxsize=3)
    cache["a"] = 1
    cache["b"] = 2
    cache["c"] = 3
    cache["d"] = 4
    assert len(cache) == 3
    assert "a" not in cache
    assert "d" in cache
    # check reading b and adding e -> c should be pop up.
    cache["b"]
    cache["e"] = "5"
    assert cache["b"] == 2
    assert "c" not in cache
