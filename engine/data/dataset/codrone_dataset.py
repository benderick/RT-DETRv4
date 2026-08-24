"""CODrone specialization of the shared DOTA OBB dataset adapter."""

from __future__ import annotations

import re

from .dota_obb_dataset import DotaOBBDetection
from ...core import register


CODRONE_CLASSES = (
    "car", "people", "motor", "truck", "traffic-sign", "traffic-light",
    "boat", "bus", "bicycle", "tricycle", "ship", "bridge",
)

_CLASS_ALIASES = {
    "traffic_sign": "traffic-sign",
    "traffic sign": "traffic-sign",
    "traffic_light": "traffic-light",
    "traffic light": "traffic-light",
}


@register()
class CODroneDetection(DotaOBBDetection):
    """Read CODrone full images or materialized DOTA-style tiles."""

    def __init__(
        self,
        root: str,
        transforms=None,
        image_dir: str = "images",
        ann_dir: str = "annfile",
        classes=CODRONE_CLASSES,
        filter_empty_gt: bool = False,
    ):
        super().__init__(
            root,
            transforms,
            image_dir=image_dir,
            ann_dir=ann_dir,
            classes=classes,
            class_aliases=_CLASS_ALIASES,
            filter_empty_gt=filter_empty_gt,
            dataset_name="CODrone",
        )

    def get_image_metadata(self, image_id: int):
        """Decode CODrone acquisition factors from its stable image name."""

        name = self.image_ids[int(image_id)].lower()
        illumination = re.search(r"(?:^|_)(day|night)(?:_|$)", name)
        altitude = re.search(r"_(\d+(?:\.\d+)?)m(?:_|$)", name)
        view = re.search(r"_(\d+(?:\.\d+)?)c(?:_|$)", name)
        frame = re.search(r"_frame_(\d+)(?:_|$)", name)
        return {
            "illumination": illumination.group(1) if illumination else None,
            "altitude_m": float(altitude.group(1)) if altitude else None,
            "view_angle_deg": float(view.group(1)) if view else None,
            "frame_index": int(frame.group(1)) if frame else None,
        }
