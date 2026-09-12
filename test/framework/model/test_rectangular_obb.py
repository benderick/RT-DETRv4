import unittest
import torch
from torch import nn

from engine.data.transforms.rotated_transforms import RotatedConvertToTensor
from engine.rtv4 import RotatedDFINETransformer, RotatedPostProcessor
from engine.rtv4.rotated_box_ops import rotated_iou
from engine.rtv4.dfine_decoder import MSDeformableAttention
from engine.diagnostics.obb_diagnostics import _restore_target_boxes
from test.framework.model.test_model_pipeline import build_criterion


class RectangularOBBTest(unittest.TestCase):
    def test_isotropic_normalization_preserves_geometry_and_restoration(self):
        boxes=torch.tensor([[45.,25.,20.,18.,.7],[48.,29.,22.,17.,.9]])
        target=dict(boxes=boxes.clone(),size=torch.tensor([96,64]),scale_factor=torch.ones(2),padding=torch.zeros(4))
        _,target,_=RotatedConvertToTensor(box_coordinate_mode="isotropic")((torch.zeros(8,64,96,dtype=torch.uint8),target,None))
        normalized=target["boxes"]
        torch.testing.assert_close(rotated_iou(boxes,boxes,model_space=False),rotated_iou(normalized,normalized,model_space=True))
        restored=RotatedPostProcessor().restore_boxes(normalized[None],[target])[0]
        torch.testing.assert_close(restored,boxes)
        torch.testing.assert_close(_restore_target_boxes(target),boxes)

    def test_rotated_attention_maps_isotropic_points_to_feature_axes(self):
        attention=MSDeformableAttention(32,4,1,1)
        attention.box_coordinate_mode="isotropic"
        attention.diagnostic_mode=True
        nn.init.zeros_(attention.sampling_offsets.weight)
        nn.init.zeros_(attention.sampling_offsets.bias)
        attention(torch.zeros(1,1,32),torch.tensor([[[[.5,.25,.2,.1,.3]]]]),
                  [torch.zeros(1,4,8,8)],[(2,4)])
        expected=torch.tensor([.5,.5])
        torch.testing.assert_close(attention.last_sampling_locations,expected.expand_as(attention.last_sampling_locations))

    def test_rectangular_decoder_regular_aux_dn_loss_backward(self):
        for mode in ("o2_adr","direct_angle"):
            model=RotatedDFINETransformer(num_classes=3,hidden_dim=32,num_queries=20,feat_channels=[32]*3,
                feat_strides=[8,16,32],num_levels=3,num_points=[2]*3,nhead=4,num_layers=2,
                dim_feedforward=64,num_denoising=10,reg_max=8,refinement_mode=mode,
                ocd_mode="box" if mode=="o2_adr" else "standard",box_coordinate_mode="isotropic")
            features=[torch.randn(2,32,h,w) for h,w in ((8,12),(4,6),(2,3))]
            target=[dict(labels=torch.tensor([1]),boxes=torch.tensor([[.5,.25,.2,.18,.3]])),
                    dict(labels=torch.empty(0,dtype=torch.long),boxes=torch.empty(0,5))]
            output=model(features,target)
            loss=sum(build_criterion(mode)(output,target).values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__": unittest.main()
