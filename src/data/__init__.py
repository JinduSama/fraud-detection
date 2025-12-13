"""Data generation and fraud injection modules."""
from .generator import CustomerDataGenerator, CustomerRecord
from .fraud_injector import FraudInjector, FraudType
from .fraud_patterns import ExtendedFraudInjector

__all__ = [
    "CustomerDataGenerator",
    "CustomerRecord",
    "FraudInjector",
    "FraudType",
    "ExtendedFraudInjector",
]
