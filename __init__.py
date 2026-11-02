"""Python interpretation of the DAAREM fixed-point accelerator."""

from importlib.metadata import version

from .daarem import daarem

__version__ = version("pydaarem")
__all__ = ["daarem"]
