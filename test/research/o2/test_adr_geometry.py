import math
import unittest

import torch

from engine.rtv4.obb.methods.o2.adr import (
    adr_orthogonality_error,
    adr_target_residual,
    adr_to_rbox,
    adr_values_to_corners,
    apply_adr_residuals,
    rbox_to_adr,
)
from engine.rtv4.rotated_box_ops import aligned_kld_loss, rbox_to_corners


def _corner_hausdorff(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Order-independent maximum corner error for aligned box batches."""

    distances = torch.cdist(first, second)
    return torch.maximum(
        distances.amin(dim=-1).amax(dim=-1),
        distances.amin(dim=-2).amax(dim=-1),
    )


class O2ADRGeometryTest(unittest.TestCase):
    def assert_same_geometry(self, first, second, *, atol):
        first_corners = rbox_to_corners(first, normalized_angle=True)
        second_corners = rbox_to_corners(second, normalized_angle=True)
        error = _corner_hausdorff(first_corners, second_corners)
        self.assertLessEqual(float(error.max()), atol)

    def test_zero_residual_is_identity_at_axis_boundaries(self):
        # These are radians before conversion to the model's theta/pi unit.
        angle_epsilon = 1e-5
        radians = [
            -angle_epsilon, 0.0, angle_epsilon,
            math.pi / 2 - angle_epsilon, math.pi / 2,
            math.pi / 2 + angle_epsilon,
            math.pi - angle_epsilon, math.pi, math.pi + angle_epsilon,
        ]
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                machine_angle = 4 * torch.finfo(dtype).eps
                tested_radians = radians + [
                    -machine_angle, machine_angle,
                    math.pi / 2 - machine_angle, math.pi / 2 + machine_angle,
                    math.pi - machine_angle, math.pi + machine_angle,
                ]
                boxes = torch.tensor(
                    [[0.73, 0.81, 0.20, 0.09, angle / math.pi]
                     for angle in tested_radians],
                    dtype=dtype,
                )
                recovered = adr_to_rbox(boxes, torch.zeros((len(boxes), 6), dtype=dtype))
                tolerance = 2e-6 if dtype == torch.float32 else 2e-12
                self.assert_same_geometry(recovered, boxes, atol=tolerance)

    def test_float32_near_axis_zero_residual_stress(self):
        generator = torch.Generator().manual_seed(19)
        count = 10_000
        boxes = torch.empty((count, 5), dtype=torch.float32)
        boxes[..., :2] = torch.rand((count, 2), generator=generator)
        sides = 0.01 + 0.39 * torch.rand((count, 2), generator=generator)
        boxes[..., 2] = sides.max(dim=-1).values
        boxes[..., 3] = sides.min(dim=-1).values
        axes = torch.randint(0, 3, (count,), generator=generator).float() * 0.5
        boxes[..., 4] = axes + (
            torch.rand(count, generator=generator) - 0.5
        ) * (2e-4 / math.pi)
        recovered = adr_to_rbox(boxes, torch.zeros((count, 6)))
        error = _corner_hausdorff(
            rbox_to_corners(recovered, normalized_angle=True),
            rbox_to_corners(boxes, normalized_angle=True),
        )
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertLessEqual(float(error.max()), 2e-6)

    def test_square_and_near_square_zero_residual_across_angles_and_centers(self):
        generator = torch.Generator().manual_seed(1618)
        count = 4096
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                boxes = torch.empty((count, 5), dtype=dtype)
                boxes[..., :2] = 0.01 + 0.98 * torch.rand(
                    (count, 2), generator=generator, dtype=dtype
                )
                side = 0.015 + 0.30 * torch.rand(count, generator=generator, dtype=dtype)
                # Include exact squares and several near-square regimes where
                # long-edge canonicalization is most sensitive.
                relative_gap = torch.tensor(
                    [0.0, 1e-7, 1e-5, 1e-3, 1e-2], dtype=dtype
                ).repeat((count + 4) // 5)[:count]
                boxes[..., 2] = side * (1 + relative_gap)
                boxes[..., 3] = side
                # A deterministic dense sweep covers every orientation while
                # deliberately crossing both normalized axis boundaries.
                boxes[..., 4] = torch.linspace(-0.01, 1.01, count, dtype=dtype)
                recovered = adr_to_rbox(boxes, torch.zeros((count, 6), dtype=dtype))
                error = _corner_hausdorff(
                    rbox_to_corners(recovered, normalized_angle=True),
                    rbox_to_corners(boxes, normalized_angle=True),
                )
                self.assertTrue(torch.isfinite(recovered).all())
                tolerance = 2e-6 if dtype == torch.float32 else 8e-13
                self.assertLessEqual(float(error.max()), tolerance)

    def test_encoding_and_decoding_are_translation_invariant_near_an_axis(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                first = torch.tensor(
                    [[0.25, 0.25, 0.20, 0.10, (math.pi / 2 - 5e-5) / math.pi]],
                    dtype=dtype,
                )
                second = first.clone()
                second[..., :2] = torch.tensor([0.75, 0.83], dtype=dtype)
                first_values, first_scale = rbox_to_adr(first)
                second_values, second_scale = rbox_to_adr(second)
                torch.testing.assert_close(first_values, second_values, atol=0, rtol=0)
                torch.testing.assert_close(first_scale, second_scale, atol=0, rtol=0)

                residual = torch.tensor(
                    [[0.13, -0.07, 0.04, 0.11, -0.08, 0.09]], dtype=dtype
                )
                first_decoded = adr_to_rbox(first, residual)
                second_decoded = adr_to_rbox(second, residual)
                expected_shift = second[..., :2] - first[..., :2]
                torch.testing.assert_close(
                    second_decoded[..., :2] - first_decoded[..., :2],
                    expected_shift,
                    atol=2e-7 if dtype == torch.float32 else 2e-15,
                    rtol=0,
                )
                torch.testing.assert_close(
                    second_decoded[..., 2:], first_decoded[..., 2:],
                    atol=2e-6 if dtype == torch.float32 else 2e-14,
                    rtol=0,
                )

    def test_axis_crossing_exposes_the_known_gliding_offset_chart_seam(self):
        # This is intentionally a characterization test, not a claim that
        # ADR is continuous in parameter space.  The physical boxes on either
        # side of zero are arbitrarily close, while the globally named
        # top/right offset fractions select opposite chart endpoints.
        epsilon = 1e-6
        boxes = torch.tensor([
            [.5, .5, .30, .10, -epsilon / math.pi],
            [.5, .5, .30, .10, +epsilon / math.pi],
        ], dtype=torch.float64)
        values, scale = rbox_to_adr(boxes)
        fractions = values[:, 4:] / scale
        self.assertGreater(float((fractions[0] - fractions[1]).abs().min()), .99)
        geometry_gap = _corner_hausdorff(
            rbox_to_corners(boxes[:1], normalized_angle=True),
            rbox_to_corners(boxes[1:], normalized_angle=True),
        )
        self.assertLess(float(geometry_gap), 1e-6)

    def test_axis_tie_starts_both_offsets_at_the_external_corner(self):
        boxes = torch.tensor([[.5, .5, .30, .10, 0.]], dtype=torch.float64)
        values, scale = rbox_to_adr(boxes)
        torch.testing.assert_close(
            values[:, 4:] / scale,
            torch.zeros((1, 2), dtype=torch.float64),
        )

    def test_equivalent_width_height_parameterizations_have_one_adr_encoding(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                canonical = torch.tensor(
                    [[0.43, 0.61, 0.31, 0.11, 0.137],
                     [0.52, 0.37, 0.24, 0.24, 0.731]],
                    dtype=dtype,
                )
                equivalent = canonical.clone()
                equivalent[..., 2] = canonical[..., 3]
                equivalent[..., 3] = canonical[..., 2]
                equivalent[..., 4] = canonical[..., 4] + 0.5
                values1, scale1 = rbox_to_adr(canonical)
                values2, scale2 = rbox_to_adr(equivalent)
                tolerance = 2e-7 if dtype == torch.float32 else 5e-16
                torch.testing.assert_close(values1, values2, atol=tolerance, rtol=0)
                torch.testing.assert_close(scale1, scale2, atol=tolerance, rtol=0)
                self.assert_same_geometry(
                    adr_to_rbox(canonical, torch.zeros_like(values1)),
                    adr_to_rbox(equivalent, torch.zeros_like(values2)),
                    atol=2e-6 if dtype == torch.float32 else 2e-14,
                )

    def test_target_residual_round_trip_for_random_valid_boxes(self):
        generator = torch.Generator().manual_seed(2718)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                count = 256
                reference = torch.rand((count, 5), generator=generator, dtype=dtype)
                target = torch.rand((count, 5), generator=generator, dtype=dtype)
                reference[..., :2] = 0.2 + 0.6 * reference[..., :2]
                target[..., :2] = 0.2 + 0.6 * target[..., :2]
                for boxes in (reference, target):
                    sides = 0.03 + 0.32 * boxes[..., 2:4]
                    boxes[..., 2] = sides.max(dim=-1).values
                    boxes[..., 3] = sides.min(dim=-1).values
                residual = adr_target_residual(reference, target)
                recovered = adr_to_rbox(reference, residual)
                self.assert_same_geometry(
                    recovered,
                    target,
                    atol=4e-6 if dtype == torch.float32 else 5e-13,
                )

                target_values = apply_adr_residuals(reference, residual)
                self.assertLessEqual(
                    float(adr_orthogonality_error(target_values).max()),
                    2e-5 if dtype == torch.float32 else 2e-12,
                )

    def test_six_values_use_standard_equal_diagonal_rectangle_completion(self):
        centers = torch.tensor([[.4, .6], [.7, .2]], dtype=torch.float64)
        values = torch.tensor([
            [.17, .11, .23, .19, .31, -.07],
            [.09, .21, .14, .08, -.04, .26],
        ], dtype=torch.float64)
        observed = adr_values_to_corners(centers, values)

        left, top, right, bottom, epsilon, eta = values.unbind(-1)
        width = (left + right).clamp_min(1e-7)
        height = (top + bottom).clamp_min(1e-7)
        refined_center = torch.stack((
            centers[:, 0] + .5 * (right - left),
            centers[:, 1] + .5 * (bottom - top),
        ), dim=-1)
        diagonal = torch.stack((
            torch.stack((width / 2 - epsilon, -height / 2), dim=-1),
            torch.stack((width / 2, height / 2 - eta), dim=-1),
        ), dim=1)
        radius = torch.linalg.vector_norm(diagonal, dim=-1)
        diagonal = diagonal * (radius.max(dim=-1, keepdim=True).values / radius).unsqueeze(-1)
        first, second = diagonal.unbind(dim=1)
        expected = torch.stack((first, second, -first, -second), dim=1)
        expected += refined_center[:, None]
        torch.testing.assert_close(observed, expected, atol=1e-14, rtol=0)

        edge1 = observed[:, 1] - observed[:, 0]
        edge2 = observed[:, 2] - observed[:, 1]
        torch.testing.assert_close(
            (edge1 * edge2).sum(dim=-1), torch.zeros(2, dtype=torch.float64),
            atol=1e-14, rtol=0)

    def test_random_six_values_always_decode_to_an_orthogonal_rectangle(self):
        generator = torch.Generator().manual_seed(31415)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                count = 1024
                reference = torch.rand((count, 5), generator=generator, dtype=dtype)
                reference[..., :2] = 0.2 + 0.6 * reference[..., :2]
                sides = 0.02 + 0.38 * reference[..., 2:4]
                reference[..., 2] = sides.max(dim=-1).values
                reference[..., 3] = sides.min(dim=-1).values
                residual = 2.5 * torch.randn((count, 6), generator=generator, dtype=dtype)
                decoded = adr_to_rbox(reference, residual)
                self.assertTrue(torch.isfinite(decoded).all())
                self.assertTrue((decoded[..., 2:4] > 0).all())

                corners = rbox_to_corners(decoded, normalized_angle=True)
                edge1 = corners[..., 1, :] - corners[..., 0, :]
                edge2 = corners[..., 2, :] - corners[..., 1, :]
                cosine = (edge1 * edge2).sum(dim=-1).abs() / (
                    torch.linalg.vector_norm(edge1, dim=-1)
                    * torch.linalg.vector_norm(edge2, dim=-1)
                ).clamp_min(torch.finfo(dtype).tiny)
                # Arbitrary invalid six-values can decode to a thin but still
                # legal rectangle; reconstructing its corners then magnifies
                # round-off in this normalized cosine check.
                tolerance = 1e-4 if dtype == torch.float32 else 2e-12
                self.assertLessEqual(float(cosine.max()), tolerance)

    def test_signed_side_distance_keeps_center_correction_and_gradient(self):
        # alpha becomes negative while alpha + gamma stays positive.  A
        # per-edge clamp would move the result to x=0.55; the joint span
        # representation correctly preserves the requested x=0.575 centre.
        reference = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]], dtype=torch.float64)
        residual = torch.tensor(
            [[-0.75, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        decoded = adr_to_rbox(reference, residual)
        self.assertAlmostEqual(float(decoded[0, 0]), 0.575, places=12)
        decoded[0, 0].backward()
        self.assertTrue(torch.isfinite(residual.grad).all())
        self.assertAlmostEqual(float(residual.grad[0, 0]), -0.1, places=12)

    def test_equal_diagonal_adr_decode_backward_is_finite_for_both_dtypes(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                reference = torch.tensor(
                    [[0.47, 0.53, 0.27, 0.08, 0.19],
                     [0.63, 0.29, 0.18, 0.17, 0.81]],
                    dtype=dtype,
                )
                residual = torch.tensor(
                    [[-0.9, 0.3, 0.2, -0.1, 1.3, -0.7],
                     [0.4, -1.1, -0.2, 0.5, -0.8, 1.5]],
                    dtype=dtype,
                    requires_grad=True,
                )
                decoded = adr_to_rbox(reference, residual)
                loss = decoded[..., :4].square().sum() + torch.sin(
                    2 * math.pi * decoded[..., 4]
                ).sum()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(residual.grad)
                self.assertTrue(torch.isfinite(residual.grad).all())

    def test_amp_mixed_dtype_contract_has_finite_backward(self):
        torch.manual_seed(5)
        # In the real AMP decoder, anchor/reference geometry remains float32
        # while the distribution head emits float16 logits/residuals.  Normal
        # PyTorch promotion therefore keeps geometry in the reference dtype;
        # adr.py must not contain a hidden dtype conversion of its own.
        reference = torch.rand((4096, 5), dtype=torch.float32)
        reference[:, 2:4] = reference[:, 2:4] * 0.2 + 0.001
        reference[:, 2:4] = reference[:, 2:4].sort(
            dim=-1, descending=True).values
        residual = (
            torch.rand((4096, 6), dtype=torch.float16) * 2 - 1
        ).requires_grad_()
        decoded = adr_to_rbox(reference, residual)
        self.assertEqual(decoded.dtype, reference.dtype)
        loss = decoded.square().sum(dim=-1).mean()
        gradient, = torch.autograd.grad(loss, residual)
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(gradient).all())

    def test_codebook_residual_to_kld_stress_has_finite_backward(self):
        # This exercises the complete equal-diagonal ADR -> OBB -> closed-form
        # KLD backward chain over the full published codebook, not merely a
        # box-valued surrogate loss.
        generator = torch.Generator().manual_seed(0)
        count = 32_768
        reference = torch.rand((count, 5), generator=generator)
        reference[..., :2] = 0.01 + 0.98 * reference[..., :2]
        sides = 1e-5 + 0.2 * reference[..., 2:4]
        reference[..., 2:4] = sides.sort(dim=-1, descending=True).values
        residual = (
            2 * torch.rand((count, 6), generator=generator) - 1
        ).requires_grad_()
        decoded = adr_to_rbox(reference, residual)
        loss = aligned_kld_loss(
            decoded, reference, sqrt=False, fun="log1p", tau=1.0)
        gradient, = torch.autograd.grad(loss.mean(), residual)
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
