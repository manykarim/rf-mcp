from robot.api.deco import keyword, library


@library
class CustomLib:
    def __init__(self):
        self._store = {}

    @keyword
    def kw_no_args(self):
        self._store["flag"] = True

    @keyword
    def kw_with_args(self, a, b):
        return f"{a}-{b}"

    @keyword
    def kw_named_args(self, a: int = 1, b: int = 2):
        return a + b

    @keyword
    def kw_mixed_args(self, x, y: int = 0, z: int = 0):
        return int(x) + int(y) + int(z)

    @keyword
    def kw_no_return(self, key, value):
        self._store[key] = value

    @keyword
    def kw_get(self, key):
        return self._store.get(key)

