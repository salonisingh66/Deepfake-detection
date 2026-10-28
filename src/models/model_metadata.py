"""Model metadata queries and registry interface."""

from typing import Dict, List, Optional
from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL_ID, ModelMetadata


def get_model_metadata(model_id: str) -> Optional[ModelMetadata]:
    """Retrieve metadata for a specified model ID."""
    return AVAILABLE_MODELS.get(model_id)


def list_available_models() -> List[ModelMetadata]:
    """List all registered models."""
    return list(AVAILABLE_MODELS.values())


def get_default_model_id() -> str:
    """Return the ID of the default production model."""
    return DEFAULT_MODEL_ID
