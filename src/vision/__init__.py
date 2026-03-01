"""Визуальный модуль HybridQA-Net."""

try:
    from .backbone import VisionBackbone
    from .gradcam import GradCAMPlusPlus
    from .preprocessor import ImagePreprocessor
    __all__ = ["VisionBackbone", "GradCAMPlusPlus", "ImagePreprocessor"]
except ImportError:
    # cv2 / libGL not available in this environment — heavy vision deps deferred
    __all__ = []
