"""UAV-ROD specialization of the shared DOTA OBB dataset adapter."""

from __future__ import annotations

import re

from .dota_obb_dataset import DotaOBBDetection
from ...core import register


UAV_ROD_CLASSES = ("car",)


@register()
class UAVRODDetection(DotaOBBDetection):
    """Read a converted UAV-ROD train or test split."""

    def __init__(
        self,
        root: str,
        transforms=None,
        image_dir: str = "images",
        ann_dir: str = "annfile",
        classes=UAV_ROD_CLASSES,
        filter_empty_gt: bool = False,
    ):
        super().__init__(
            root,
            transforms,
            image_dir=image_dir,
            ann_dir=ann_dir,
            classes=classes,
            filter_empty_gt=filter_empty_gt,
            dataset_name="UAV-ROD",
        )

    def get_image_metadata(self, image_id: int):
        """Expose source-video and frame identity encoded in UAV-ROD names."""

        name = self.image_ids[int(image_id)]
        match = re.fullmatch(r"(.+)_(\d+)", name)
        return {
            "video_id": match.group(1) if match else None,
            "frame_index": int(match.group(2)) if match else None,
        }
