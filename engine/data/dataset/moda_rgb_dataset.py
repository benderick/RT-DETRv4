"""Read MODA pseudo RGB as three channels without changing source files."""
from ...core import register
from .moda_dataset import MODADetection


@register()
class MODARGBDetection(MODADetection):
    source_bands = (4, 2, 1)

    def _load_image(self, index):
        source = super()._load_image(index)
        return source[list(self.source_bands)].contiguous()

    def get_dataset_provenance(self):
        result = super().get_dataset_provenance()
        result.update(
            input_channels=3,
            input_layout="CWH->CHW[B4,B2,B1]",
            retained_source_bands=list(self.source_bands),
        )
        return result
