# optional_imports.py
class OptionalImport:
    def __init__(self, name, submodules = None):
        self.module_available = False
        self.module = None
        self.name = name

        self.submodules = submodules
        
        self._attempt_import()
        self._attempt_import_submods()

    def _attempt_import_submods(self):
        if (not isinstance(self.submodules, type(None))) and self.module_available:
            submodules = self.submodules
            
            if not isinstance(submodules, list):
                submodules = [submodules,]
                
            for mod in submodules:
                __import__(f'{self.name}.{mod}')
            

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
Basemap = OptionalImport('mpl_toolkits.basemap.Basemap')
s3fs = OptionalImport('s3fs')
pyhdf = OptionalImport('pyhdf', submodules = 'SD')
