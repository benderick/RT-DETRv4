"""Opt-in input-information control; the detector retains its eight-channel stem.

Withheld bands are constant zero at both training and evaluation. This is not
an inference-time ablation of an eight-band-trained detector or an RGB model.
"""
import torch

from ...core import register
from .moda_dataset import MODADetection


@register()
class MODAInformationControl(MODADetection):
    def __init__(self, root, transforms=None, split_file=None, expected_labels=None,
                 filter_empty_gt=False, retained_bands=(4, 2, 1)):
        if (not isinstance(retained_bands, (list, tuple)) or not retained_bands
                or any(type(b) is not int or not 0 <= b < 8 for b in retained_bands)
                or len(set(retained_bands)) != len(retained_bands)):
            raise ValueError("retained_bands must be distinct integer band IDs in [0,7]")
        # Preserve physical channel positions, regardless of display RGB order.
        self.retained_bands = tuple(sorted(retained_bands))
        super().__init__(root, transforms, split_file, expected_labels, filter_empty_gt)

    def _load_image(self, index):
        source = super()._load_image(index)
        image = torch.zeros_like(source)
        image[list(self.retained_bands)] = source[list(self.retained_bands)]
        return image

    def get_dataset_provenance(self):
        result = super().get_dataset_provenance()
        result.update(
            input_information_control="constant_zero_from_training_through_evaluation",
            retained_source_bands=list(self.retained_bands),
            zeroed_source_bands=[b for b in range(8) if b not in self.retained_bands],
            source_band_positions_preserved=True,
        )
        return result
