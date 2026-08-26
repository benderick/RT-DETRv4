import unittest

import torch

from engine.rtv4.obb.methods.o2.adr import adr_target_residual
from test.framework.model.test_model_pipeline import build_criterion, build_tiny_model


class ConvergenceTest(unittest.TestCase):
    def _fit_fixed_features(self, refinement_mode, steps, *, aux_loss):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_tiny_model(
            aux_loss=aux_loss, num_denoising=0,
            refinement_mode=refinement_mode).to(device).train()
        criterion = build_criterion(refinement_mode=refinement_mode).to(device)
        features = [torch.randn(1, 32, 8, 8, device=device),
                    torch.randn(1, 32, 4, 4, device=device),
                    torch.randn(1, 32, 2, 2, device=device)]
        targets = [{"labels": torch.tensor([1], device=device),
                    "boxes": torch.tensor([[.48, .52, .22, .09, .93]], device=device)}]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        history = []
        pre_box_gradient_seen = False
        for _ in range(steps):
            optimizer.zero_grad()
            losses = criterion(model(features, targets), targets)
            loss = sum(losses.values())
            loss.backward()
            pre_box_gradient = model.pre_bbox_head.layers[-1].weight.grad
            pre_box_gradient_seen |= bool(
                pre_box_gradient is not None and pre_box_gradient.abs().sum() > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
        self.assertLess(sum(history[-3:]) / 3, history[0] * 0.65, history)
        return model, criterion, features, targets, pre_box_gradient_seen

    def test_tiny_o2_overfits_fixed_features(self):
        # ADR is defined around a detached traditional-head pre-box.  The
        # production path therefore requires auxiliary supervision of that
        # pre-box; disabling it creates a fixed grid anchor whose target may
        # lie outside the finite ADR codebook and is not a valid convergence
        # test of O²-DFINE.
        model, criterion, features, targets, pre_box_gradient_seen = \
            self._fit_fixed_features(
                refinement_mode="o2_adr", steps=60, aux_loss=True)
        self.assertTrue(pre_box_gradient_seen)

        outputs = model(features, targets)
        source_indices, target_indices = criterion.matcher(
            outputs, targets)["indices"][0]
        self.assertEqual(len(source_indices), 1)
        source_indices = source_indices.to(outputs["ref_points"].device)
        target_indices = target_indices.to(targets[0]["boxes"].device)
        residual = adr_target_residual(
            outputs["ref_points"][0].index_select(0, source_indices),
            targets[0]["boxes"].index_select(0, target_indices),
        )
        self.assertLessEqual(float(residual.abs().max()), 1.0)

    def test_tiny_direct_angle_overfits_fixed_features(self):
        self._fit_fixed_features(
            refinement_mode="direct_angle", steps=20, aux_loss=False)


if __name__ == "__main__":
    unittest.main()
