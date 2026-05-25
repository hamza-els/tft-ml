"""
Canonical StateVector schema. Import from here to ensure consistent structure
across the vision layer, macro engine, and win predictor.
"""

# Re-export from vision.state_extractor to keep a single source of truth.
from src.vision.state_extractor import StateVector, UnitState

__all__ = ["StateVector", "UnitState"]
