# optional_imports.py
class OptionalImport:
    def __init__(self, name):
        self.module_available = False
        self.module = None
        self.name = name
        self._attempt_import()

    def _attempt_import(self):
        try:
            self.module = __import__(self.name)
            self.module_available = True
        except ImportError:
            self.module_available = False

    def __getattr__(self, item):
        if not self.module_available:
            raise ImportError(f"{self.name} is required for this feature. Please install it to use this functionality.")
        return getattr(self.module, item)

# Creating the pandas facade
geopandas = OptionalImport('geopandas')
shapely = OptionalImport('shapely')
cartopy = OptionalImport('cartopy')