"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

from .rtv4 import RTv4

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rotated_dfine_decoder import RotatedDFINETransformer
from .rotated_matcher import RotatedHungarianMatcher
from .rotated_criterion import RotatedRTv4Criterion
from .rotated_postprocessor import RotatedPostProcessor
from .obb_visualization import draw_obbs, save_obb_visualization
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .rtv4_criterion import RTv4Criterion

from .dinov3_teacher import DINOv3TeacherModel
