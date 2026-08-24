import unittest

import torch

from test.framework.model.test_model_pipeline import build_criterion, build_tiny_model


class ConvergenceTest(unittest.TestCase):
    def _assert_fixed_feature_convergence(self, refinement_mode, steps):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_tiny_model(
            aux_loss=False, num_denoising=0,
            refinement_mode=refinement_mode).to(device).train()
        criterion = build_criterion(refinement_mode=refinement_mode).to(device)
        features = [torch.randn(1, 32, 8, 8, device=device),
                    torch.randn(1, 32, 4, 4, device=device),
                    torch.randn(1, 32, 2, 2, device=device)]
        targets = [{"labels": torch.tensor([1], device=device),
                    "boxes": torch.tensor([[.48, .52, .22, .09, .93]], device=device)}]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        history = []
        for _ in range(steps):
            optimizer.zero_grad()
            losses = criterion(model(features, targets), targets)
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
        self.assertLess(sum(history[-3:]) / 3, history[0] * 0.65, history)

    def test_tiny_o2_overfits_fixed_features(self):
        # Six ADR distributions carry more logits than the scalar-angle head,
        # so give the primary architecture a correspondingly longer tiny fit.
        self._assert_fixed_feature_convergence(
            refinement_mode="o2_adr", steps=35)

    def test_tiny_direct_angle_overfits_fixed_features(self):
        self._assert_fixed_feature_convergence(
            refinement_mode="direct_angle", steps=20)


if __name__ == "__main__":
    unittest.main()
