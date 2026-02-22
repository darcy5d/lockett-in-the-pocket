"""
Core module for AFL Match Predictor.
Contains mapping layers, config, and shared services.
"""

from core.mappings import (
    TeamNameMapper,
    VenueMapper,
    PlayerMapper,
    load_all_mappings,
)

__all__ = [
    "TeamNameMapper",
    "VenueMapper", 
    "PlayerMapper",
    "load_all_mappings",
]
