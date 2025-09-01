"""
Voice Recognition Service Package

This package contains modular services for voice recognition including:
- Service state management
- Audio preprocessing
- ASR (Automatic Speech Recognition)
- Feature extraction
- Speaker enrollment
- Speaker diarization
- Audio-text alignment
- Output generation
- Base64 handling
- Pipeline orchestration

All services are designed for production deployment with hardcoded configurations.
"""

from .service_state import ServiceState
from .service_preprocess import PreprocessService
from .service_asr import ASRService
from .service_features import FeatureService
from .service_enroll import EnrollmentService
from .service_diarize import DiarizationService
from .service_align import AlignmentService
from .service_output import OutputService
from .service_base64 import Base64Service
from .service_pipeline import PipelineService

__all__ = [
    'ServiceState',
    'PreprocessService',
    'ASRService',
    'FeatureService',
    'EnrollmentService',
    'DiarizationService',
    'AlignmentService',
    'OutputService',
    'Base64Service',
    'PipelineService'
]
