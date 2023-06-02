# -*- coding: utf-8 -*-

"""Top-level package for proto_sep."""

__author__ = """J. Michael Burgess"""
__email__ = 'jburgess@mpe.mpg.de'


from .utils.configuration import proto_sep_config, show_configuration
from .utils.logging import (
    activate_warnings,
    silence_warnings,
    update_logging_level,
)


from .protostars import Protostar, Group, Region, Catalog
