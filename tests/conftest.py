"""Mirror nbf's isolated wheel environment; never change machine settings."""
import os
for variable in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(variable, None)
