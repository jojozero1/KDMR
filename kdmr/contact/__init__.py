"""
Contact estimation module for KDMR.

This module contains:
- GRFProcessor: Ground Reaction Force data processing
- ContactEstimator: Contact sequence estimation from GRF
- ContactMode: Contact mode definitions and utilities
"""

from kdmr.contact.grf_processor import GRFProcessor
from kdmr.contact.contact_estimator import ContactEstimator
from kdmr.contact.contact_mode import ContactMode, ContactSequence

__all__ = [
    "GRFProcessor",
    "ContactEstimator",
    "ContactMode",
    "ContactSequence",
]
