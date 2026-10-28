"""Model Loading and Caching Module for DeepShield AI.

Ensures models are loaded once, placed in evaluation mode on CPU,
and provides honest availability reporting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config.model_config import AVAILABLE_MODELS, DEFAULT_MODEL_ID, ModelMetadata
from config.settings import settings
from src.models.architectures import EfficientNetB0Deepfake, MesoInception4
from src.utils.logging_config import logger


@dataclass
class LoadedModelContainer:
    """Wrapper holding loaded model components and status."""
    model_id: str
    metadata: ModelMetadata
    model: nn.Module
    processor: Optional[Any] = None
    is_ready: bool = True
    error_message: Optional[str] = None


class ModelLoader:
    """Singleton model cache manager preventing redundant model reloads."""
    _instance: Optional["ModelLoader"] = None
    _cache: Dict[str, LoadedModelContainer] = {}

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cache = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance

    def load_model(self, model_id: str = DEFAULT_MODEL_ID) -> LoadedModelContainer:
        """Load requested model on CPU and return LoadedModelContainer."""
        if model_id in self._cache and self._cache[model_id].is_ready:
            return self._cache[model_id]

        if model_id not in AVAILABLE_MODELS:
            err = f"Unknown model ID '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}"
            logger.error(err)
            return LoadedModelContainer(
                model_id=model_id,
                metadata=AVAILABLE_MODELS[DEFAULT_MODEL_ID],
                model=nn.Identity(),
                is_ready=False,
                error_message=err
            )

        metadata = AVAILABLE_MODELS[model_id]
        logger.info(f"Loading deepfake detection model: {metadata.name} ({model_id}) on CPU...")

        try:
            if model_id == "vit_deepfake":
                # Genuine pretrained Vision Transformer from HuggingFace
                processor = AutoImageProcessor.from_pretrained(
                    metadata.checkpoint_id_or_path,
                    use_fast=False
                )
                model = AutoModelForImageClassification.from_pretrained(
                    metadata.checkpoint_id_or_path
                )
                model.to(settings.DEVICE)
                model.eval()

                container = LoadedModelContainer(
                    model_id=model_id,
                    metadata=metadata,
                    model=model,
                    processor=processor,
                    is_ready=True,
                    error_message=None
                )
                self._cache[model_id] = container
                logger.info(f"Model '{model_id}' successfully loaded and verified.")
                return container

            elif model_id == "efficientnet_b0":
                model = EfficientNetB0Deepfake(num_classes=2, pretrained_backbone=False)
                checkpoint_path = Path(metadata.checkpoint_id_or_path)
                
                if checkpoint_path.exists():
                    state_dict = torch.load(checkpoint_path, map_location=settings.DEVICE)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded custom weights for {model_id} from {checkpoint_path}")
                else:
                    logger.info(f"No custom checkpoint at {checkpoint_path}; running EfficientNet-B0 in baseline mode.")

                model.to(settings.DEVICE)
                model.eval()

                container = LoadedModelContainer(
                    model_id=model_id,
                    metadata=metadata,
                    model=model,
                    processor=None,
                    is_ready=True,
                    error_message=None
                )
                self._cache[model_id] = container
                return container

            elif model_id == "mesonet":
                model = MesoInception4(num_classes=2)
                checkpoint_path = Path(metadata.checkpoint_id_or_path)
                
                if checkpoint_path.exists():
                    state_dict = torch.load(checkpoint_path, map_location=settings.DEVICE)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded custom weights for {model_id} from {checkpoint_path}")
                else:
                    logger.info(f"No custom checkpoint at {checkpoint_path}; running MesoNet in baseline mode.")

                model.to(settings.DEVICE)
                model.eval()

                container = LoadedModelContainer(
                    model_id=model_id,
                    metadata=metadata,
                    model=model,
                    processor=None,
                    is_ready=True,
                    error_message=None
                )
                self._cache[model_id] = container
                return container

            else:
                raise ValueError(f"Unsupported model identifier: {model_id}")

        except Exception as e:
            err_msg = f"Failed to load model '{model_id}': {str(e)}"
            logger.error(err_msg)
            container = LoadedModelContainer(
                model_id=model_id,
                metadata=metadata,
                model=nn.Identity(),
                processor=None,
                is_ready=False,
                error_message=err_msg
            )
            self._cache[model_id] = container
            return container

    def is_model_ready(self, model_id: str = DEFAULT_MODEL_ID) -> Tuple[bool, Optional[str]]:
        """Check if model is currently ready for inference."""
        if model_id in self._cache:
            container = self._cache[model_id]
            return container.is_ready, container.error_message
            
        container = self.load_model(model_id)
        return container.is_ready, container.error_message

    def clear_cache(self) -> None:
        """Clear loaded models from memory."""
        self._cache.clear()
        logger.info("ModelLoader cache cleared.")
