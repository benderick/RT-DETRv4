import unittest

import torch

from test.test_model_pipeline import build_criterion, build_tiny_model


class ConvergenceTest(unittest.TestCase):
    def test_tiny_dfine_obb_overfits_fixed_features(self):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_tiny_model(aux_loss=False, num_denoising=0).to(device).train()
        criterion = build_criterion().to(device)
        features = [torch.randn(1, 32, 8, 8, device=device),
                    torch.randn(1, 32, 4, 4, device=device),
                    torch.randn(1, 32, 2, 2, device=device)]
        targets = [{"labels": torch.tensor([1], device=device),
                    "boxes": torch.tensor([[.48, .52, .22, .09, .93]], device=device)}]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        history = []
        for _ in range(20):
            optimizer.zero_grad()
            losses = criterion(model(features, targets), targets)
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
        self.assertTrue(all(torch.isfinite(torch.tensor(history))))
        self.assertLess(sum(history[-3:]) / 3, history[0] * 0.65, history)


if __name__ == "__main__":
    unittest.main()
