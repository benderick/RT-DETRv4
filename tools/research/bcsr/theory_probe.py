#!/usr/bin/env python3
"""Falsifiable local-model checks for the BCSR design review, not AP evidence."""
import argparse
import json
import math
from pathlib import Path

import torch


def boundary_model(q, samples=4):
    """Boundary points, normals and normal-displacement Jacobian [4K,5].

    q=(cx,cy,log(w),log(h),theta). Increment coordinates are
    (dcx/s,dcy/s,dlog(w),dlog(h),dtheta), with s=sqrt(w*h) fixed per step.
    This is a local geometric model, not a learned image feature model.
    """
    w, h = q[2:4].exp()
    u = torch.stack((q[4].cos(), q[4].sin()))
    v = torch.stack((-q[4].sin(), q[4].cos()))
    normals = torch.stack((-v, u, v, -u))
    tangents = torch.stack((u, v, -u, -v))
    lengths = torch.stack((w, h, w, h))
    radii = torch.stack((h, w, h, w)) / 2
    t = (torch.arange(samples, dtype=q.dtype, device=q.device)+.5)/samples-.5
    offsets = normals[:, None]*radii[:, None, None] + tangents[:, None]*lengths[:, None, None]*t[None, :, None]
    normal = normals[:, None].expand_as(offsets).reshape(-1, 2)
    offset = offsets.reshape(-1, 2)
    turn = torch.stack((-offset[:, 1], offset[:, 0]), -1)
    jacobian = torch.cat((normal*torch.sqrt(w*h),
        ((normal @ u)*(offset @ u))[:, None],
        ((normal @ v)*(offset @ v))[:, None],
        (normal*turn).sum(-1, keepdim=True)), -1)
    return q[:2]+offset, normal, jacobian


def solve_local(jacobian, residual, precision, damping=0.):
    information = jacobian.T @ precision @ jacobian
    system = information + damping*torch.eye(5, dtype=jacobian.dtype)
    return torch.linalg.solve(system, jacobian.T @ precision @ residual)


def run_probe():
    torch.manual_seed(0)
    q = torch.tensor([10., 20., math.log(40), math.log(20), .31], dtype=torch.float64)
    points, normals, jacobian = boundary_model(q)
    scale = torch.exp(q[2:4].sum()/2)
    finite_difference = []
    for column in range(5):
        step = torch.zeros(5, dtype=q.dtype)
        step[column] = 1e-6*(scale if column < 2 else 1.)
        plus = boundary_model(q+step)[0]
        minus = boundary_model(q-step)[0]
        finite_difference.append(((plus-minus)*normals).sum(-1)/(2e-6))
    numerical = torch.stack(finite_difference, -1)
    torch.testing.assert_close(jacobian, numerical, atol=1e-7, rtol=1e-7)

    edge_mean = jacobian.reshape(4, 4, 5).mean(1)
    t = (torch.arange(4, dtype=q.dtype)+.5)/4-.5
    first_moment = (jacobian.reshape(4, 4, 5)*t[None, :, None]).mean(1)
    mean_and_moment = torch.cat((edge_mean, first_moment))
    rank = lambda a: int(torch.linalg.matrix_rank(a))
    assert (rank(jacobian), rank(edge_mean), rank(mean_and_moment)) == (5, 4, 5)
    assert edge_mean[:, 4].abs().max() < 1e-12

    variance = torch.linspace(.2, 2., len(points), dtype=q.dtype)
    precision = torch.diag(variance.reciprocal())
    information = jacobian.T @ precision @ jacobian
    covariance = torch.linalg.inv(information)
    truth = torch.tensor([.01, -.015, .02, -.01, .025], dtype=q.dtype)
    noise = torch.randn(20000, len(points), dtype=q.dtype)*variance.sqrt()
    gain = torch.linalg.solve(information, jacobian.T @ precision)
    estimated = (jacobian @ truth+noise) @ gain.T
    mse = (estimated-truth).square().mean(0)
    torch.testing.assert_close(mse, covariance.diag(), rtol=.05, atol=0.)

    # Row reindexing and frame/gauge changes must reindex the estimated update.
    residual = jacobian @ truth+noise[0]
    update = solve_local(jacobian, residual, precision, damping=.3)
    permutation = torch.randperm(len(points))
    torch.testing.assert_close(update, solve_local(jacobian[permutation], residual[permutation],
        precision[permutation][:, permutation], damping=.3), atol=1e-12, rtol=1e-12)
    phi = .7
    rotation = q.new_tensor([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
    rotated_q = q.clone()
    rotated_q[:2] = rotation @ q[:2]
    rotated_q[4] += phi
    transformed_jacobian = boundary_model(rotated_q)[2]
    transform = torch.eye(5, dtype=q.dtype)
    transform[:2, :2] = rotation
    torch.testing.assert_close(transformed_jacobian @ transform, jacobian, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(solve_local(transformed_jacobian, residual, precision, damping=.3),
        transform @ update, atol=1e-12, rtol=1e-12)
    swapped_q = q.clone()
    swapped_q[2:4] = q[[3, 2]]
    swapped_q[4] += math.pi/2
    swapped_points, _, swapped_jacobian = boundary_model(swapped_q)
    mapping = torch.cdist(swapped_points, points).argmin(1)
    assert len(mapping.unique()) == len(points)
    exchange = torch.eye(5, dtype=q.dtype)[[0, 1, 3, 2, 4]]
    torch.testing.assert_close(swapped_jacobian @ exchange, jacobian[mapping], atol=1e-12, rtol=1e-12)

    objective = lambda x: float(.5*(residual-jacobian @ x) @ precision @ (residual-jacobian @ x)+.15*x.square().sum())
    assert objective(update) <= objective(torch.zeros_like(update))
    zero = solve_local(jacobian, residual, torch.zeros_like(precision), damping=.3)
    assert torch.equal(zero, torch.zeros_like(zero))

    # Independent unbiased band-wise estimates: heterogeneous edge reliability.
    edge_variance = torch.tensor([[1., 9.], [9., 1.]], dtype=q.dtype)
    local_weights = edge_variance.reciprocal()
    local_weights /= local_weights.sum(-1, keepdim=True)
    shared_weights = edge_variance.sum(0).reciprocal()
    shared_weights /= shared_weights.sum()
    local_risk = float((local_weights.square()*edge_variance).sum())
    shared_risk = float((shared_weights.square()*edge_variance).sum())
    assert local_risk < shared_risk

    # Eight correlated copies are not eight independent information sources.
    rho, bands = .9, 8
    correlated = (1-rho)*torch.eye(bands, dtype=q.dtype)+rho*torch.ones(bands, bands, dtype=q.dtype)
    one = torch.ones(bands, dtype=q.dtype)
    true_variance = float(1/(one @ torch.linalg.solve(correlated, one)))
    claimed_independent_variance = 1/bands
    assert true_variance > 7*claimed_independent_variance
    return dict(scope="synthetic_local_geometry_and_estimation_only_not_detector_accuracy",
        jacobian_max_error=float((jacobian-numerical).abs().max()),
        rank_pointwise=rank(jacobian), rank_edge_mean=rank(edge_mean),
        rank_mean_plus_first_moment=rank(mean_and_moment),
        mean_angle_sensitivity=float(edge_mean[:, 4].abs().max()),
        gls_mse_over_expected=(mse/covariance.diag()).tolist(),
        frame_and_gauge_checks=True, zero_evidence_zero_update=True,
        fixed_local_objective_before=objective(torch.zeros_like(update)),
        fixed_local_objective_after=objective(update),
        local_band_weighting_risk=local_risk, best_shared_weighting_risk=shared_risk,
        correlated_band_variance=true_variance,
        independence_assumed_variance=claimed_independent_variance,
        covariance_underestimation_factor=true_variance/claimed_independent_variance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("logs/research/bcsr/cleanup_review/theory_probe.json"))
    args = parser.parse_args()
    report = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
