#!/usr/bin/env python3
"""Predict final hydrogel coordinates by learning final-minus-initial deltas.

This is a cleaner geometry model than the hand-built relaxers:

    initial_xyz -> GNN -> delta_xyz
    predicted_final_xyz = initial_xyz + delta_xyz

For the local SimCode proof-of-theory data, each run has:
    gel_coordinates_placed*.txt
    gel_coordinates_crosslinked*.txt
    ConnectionMatrix_crosslinked*.txt

The model trains on paired initial/final coordinates. It uses a lightweight
message-passing network over local initial-neighborhood edges so every domain
can learn from nearby domains and from its molecule/domain identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch import nn


SCRIPT_VERSION = "delta_xyz_gnn_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/predict a delta-XYZ coordinate GNN.")
    parser.add_argument("--project-root", default=".", help="Folder containing SimCode text outputs.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--nd", type=int, default=4)
    parser.add_argument("--box-size-nm", type=float, default=150.0)
    parser.add_argument("--neighbor-radius-nm", type=float, default=9.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bond-loss-weight", type=float, default=0.35)
    parser.add_argument("--bond-target-source", choices=["true", "radius"], default="true")
    parser.add_argument("--particle-radius-nm", type=float, default=2.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@dataclass
class GraphSample:
    name: str
    initial_xyz: np.ndarray
    final_xyz: np.ndarray
    conn_path: Path
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    target_delta: np.ndarray
    bond_edges: np.ndarray
    bond_length_targets: np.ndarray


class DeltaGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, steps: int):
        super().__init__()
        self.steps = steps
        self.node_in = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                for _ in range(steps)
            ]
        )
        self.update_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(steps)
            ]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        h = self.node_in(node_features)
        src, dst = edge_index
        for message_mlp, update_mlp in zip(self.edge_mlps, self.update_mlps):
            msg = message_mlp(torch.cat([h[src], h[dst], edge_features], dim=1))
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msg)
            degree = torch.zeros((h.shape[0], 1), dtype=h.dtype, device=h.device)
            degree.index_add_(0, dst, torch.ones((dst.shape[0], 1), dtype=h.dtype, device=h.device))
            agg = agg / degree.clamp_min(1.0)
            h = h + update_mlp(torch.cat([h, agg], dim=1))
        return self.out(h)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    project = Path(args.project_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    analysis = output / "Analysis" / "delta_xyz_gnn"
    models = output / "Models"
    pred_dir = output / "PredictedFrames"
    figures = output / "Figures" / "delta_xyz_gnn"
    for path in (analysis, models, pred_dir, figures):
        path.mkdir(parents=True, exist_ok=True)

    samples = load_samples(project, args)
    if len(samples) < 3:
        raise ValueError(f"Need at least 3 paired runs; found {len(samples)}")
    train_idx, val_idx, test_idx = split_indices(len(samples), args.seed)

    model = DeltaGNN(
        node_dim=samples[0].node_features.shape[1],
        edge_dim=samples[0].edge_features.shape[1],
        hidden_dim=args.hidden_dim,
        steps=args.message_steps,
    )
    device = choose_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_state = None
    best_epoch = 0
    best_val = math.inf
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, samples, train_idx, device, args, optimizer)
        val_loss = run_epoch(model, samples, val_idx, device, args, optimizer=None)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if epoch == 1 or epoch % 10 == 0:
            print(f"Delta XYZ GNN epoch {epoch:03d}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best validation loss {best_val:.6f} at epoch {best_epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(analysis / "delta_xyz_training_history.csv", index=False)

    rows = []
    for split_name, indices in [("train", train_idx), ("validation", val_idx), ("test", test_idx), ("all", np.arange(len(samples)))]:
        rows.append(evaluate_split(model, samples, indices, split_name, device, args))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(analysis / "delta_xyz_metrics.csv", index=False)

    save_predictions(model, samples, pred_dir, device, args)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_state_dict": model.state_dict(),
            "node_dim": samples[0].node_features.shape[1],
            "edge_dim": samples[0].edge_features.shape[1],
            "hidden_dim": args.hidden_dim,
            "message_steps": args.message_steps,
            "box_size_nm": args.box_size_nm,
            "nd": args.nd,
            "neighbor_radius_nm": args.neighbor_radius_nm,
            "max_neighbors": args.max_neighbors,
            "bond_loss_weight": args.bond_loss_weight,
            "bond_target_source": args.bond_target_source,
        },
        models / "delta_xyz_gnn.pt",
    )
    (analysis / "delta_xyz_run_summary.json").write_text(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "n_samples": len(samples),
                "train": len(train_idx),
                "validation": len(val_idx),
                "test": len(test_idx),
                "best_epoch": best_epoch,
                "best_validation_loss": best_val,
                "device": str(device),
                "bond_loss_weight": float(args.bond_loss_weight),
                "bond_target_source": args.bond_target_source,
            },
            indent=2,
        )
    )
    if args.render:
        render_predictions(project, pred_dir, figures, args)
    print(f"Saved metrics: {analysis / 'delta_xyz_metrics.csv'}")
    print(f"Saved model: {models / 'delta_xyz_gnn.pt'}")
    print(f"Saved predictions: {pred_dir}")


def load_samples(project: Path, args: argparse.Namespace) -> list[GraphSample]:
    placed = sorted(
        [p for p in project.glob("gel_coordinates_placed*.txt") if "_run" in p.stem],
        key=run_sort_key,
    )
    samples = []
    for placed_path in placed:
        final_path = crosslinked_path_for(placed_path)
        conn_path = connection_path_for(placed_path)
        if not final_path.exists() or not conn_path.exists():
            continue
        initial = load_txt(placed_path, dtype=float)
        final = load_txt(final_path, dtype=float)
        n = min(len(initial), len(final))
        initial = initial[:n]
        final = final[:n]
        if n % args.nd != 0:
            n = (n // args.nd) * args.nd
            initial = initial[:n]
            final = final[:n]
        # crosslink_gel() begins by sorting molecules farthest-first. The
        # placement file is saved before that sort, while the crosslinked file
        # is saved after it. Without mirroring this reorder, row i in initial
        # does not correspond to row i in final, so delta_xyz is meaningless.
        initial = sort_molecules_by_distance_from_center(initial, args.nd)
        initial = coordinates_to_box_frame(initial, args.box_size_nm)
        final = coordinates_to_box_frame(final, args.box_size_nm)
        edge_index, edge_features = local_graph(initial, args.neighbor_radius_nm, args.max_neighbors, args.box_size_nm)
        node_features = node_features_for(initial, args.nd, args.box_size_nm)
        target_delta = (final - initial) / args.box_size_nm
        conn = load_txt(conn_path, dtype=int)
        bond_edges = connection_edges(conn, len(initial))
        if len(bond_edges):
            if args.bond_target_source == "radius":
                bond_lengths = np.full(len(bond_edges), 2.0 * args.particle_radius_nm, dtype=np.float32)
            else:
                bond_lengths = np.linalg.norm(final[bond_edges[:, 0]] - final[bond_edges[:, 1]], axis=1).astype(np.float32)
        else:
            bond_lengths = np.zeros((0,), dtype=np.float32)
        samples.append(
            GraphSample(
                name=placed_path.stem,
                initial_xyz=initial.astype(np.float32),
                final_xyz=final.astype(np.float32),
                conn_path=conn_path,
                node_features=node_features.astype(np.float32),
                edge_index=edge_index.astype(np.int64),
                edge_features=edge_features.astype(np.float32),
                target_delta=target_delta.astype(np.float32),
                bond_edges=bond_edges.astype(np.int64),
                bond_length_targets=bond_lengths.astype(np.float32),
            )
        )
    return samples


def run_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_run(\d+)", path.stem)
    # Put the bare gel_coordinates_placed.txt last because it may be from a
    # different run size than run2-run10.
    return (int(match.group(1)) if match else 9999, path.name)


def crosslinked_path_for(placed: Path) -> Path:
    match = re.search(r"_run(\d+)", placed.stem)
    if match:
        return placed.with_name(f"gel_coordinates_crosslinked_run{match.group(1)}.txt")
    return placed.with_name("gel_coordinates_crosslinked.txt")


def connection_path_for(placed: Path) -> Path:
    match = re.search(r"_run(\d+)", placed.stem)
    if match:
        return placed.with_name(f"ConnectionMatrix_crosslinked_run{match.group(1)}.txt")
    return placed.with_name("ConnectionMatrix_crosslinked.txt")


def load_txt(path: Path, dtype=float) -> np.ndarray:
    arr = np.loadtxt(path, skiprows=1, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def connection_edges(conn: np.ndarray, domain_count: int) -> np.ndarray:
    edges = set()
    for a in range(min(domain_count, conn.shape[0])):
        for value in conn[a]:
            b = int(value) - 1
            if 0 <= b < domain_count and b != a:
                edges.add(tuple(sorted((a, b))))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def sort_molecules_by_distance_from_center(points: np.ndarray, nd: int) -> np.ndarray:
    """Mirror gel_network.sort_by_distance_from_center for coordinates only."""
    points = np.asarray(points, dtype=np.float32)
    nm = points.shape[0] // nd
    trimmed = points[: nm * nd]
    molecules = trimmed.reshape(nm, nd, 3)
    coms = np.full((nm, 3), np.nan, dtype=np.float32)
    for i in range(nm):
        pts = molecules[i]
        ok = np.isfinite(pts).all(axis=1)
        if ok.any():
            coms[i] = pts[ok].mean(axis=0)
    dist = np.where(np.isnan(coms[:, 0]), 0.0, np.linalg.norm(coms, axis=1))
    order = np.argsort(dist)[::-1]
    sorted_points = np.full_like(trimmed, np.nan)
    for new_i, old_i in enumerate(order):
        sorted_points[new_i * nd : new_i * nd + nd] = trimmed[old_i * nd : old_i * nd + nd]
    if len(points) > nm * nd:
        sorted_points = np.vstack([sorted_points, points[nm * nd :]])
    return sorted_points


def coordinates_to_box_frame(points: np.ndarray, box_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        return points.copy()
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    if np.all(mins >= -1e-4) and np.all(maxs <= box_size + 1e-4):
        origin = np.zeros(3, dtype=np.float32)
    elif np.all(mins >= -0.65 * box_size) and np.all(maxs <= 0.65 * box_size):
        origin = np.full(3, -box_size / 2.0, dtype=np.float32)
    else:
        origin = np.median(finite, axis=0).astype(np.float32) - box_size / 2.0
    return np.mod(points - origin, box_size).astype(np.float32)


def local_graph(points: np.ndarray, radius: float, max_neighbors: int, box_size: float) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    edges = []
    feats = []
    for i, point in enumerate(points):
        idxs = tree.query_ball_point(point, r=radius)
        idxs = [j for j in idxs if j != i]
        idxs.sort(key=lambda j: float(np.linalg.norm(points[j] - point)))
        for j in idxs[:max_neighbors]:
            diff = points[j] - point
            dist = float(np.linalg.norm(diff))
            edges.append((i, j))
            feats.append([diff[0] / box_size, diff[1] / box_size, diff[2] / box_size, dist / radius])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    return np.asarray(edges, dtype=np.int64).T, np.asarray(feats, dtype=np.float32)


def node_features_for(initial: np.ndarray, nd: int, box_size: float) -> np.ndarray:
    n = len(initial)
    nm = max(n // nd, 1)
    mol = (np.arange(n) // nd).astype(np.float32)
    dom = (np.arange(n) % nd).astype(np.float32)
    centered = initial / box_size
    return np.column_stack(
        [
            centered,
            mol / max(nm - 1, 1),
            dom / max(nd - 1, 1),
            np.sin(2.0 * np.pi * dom / max(nd, 1)),
            np.cos(2.0 * np.pi * dom / max(nd, 1)),
        ]
    )


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def sample_to_device(sample: GraphSample, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(sample.node_features, dtype=torch.float32, device=device),
        torch.as_tensor(sample.edge_index, dtype=torch.long, device=device),
        torch.as_tensor(sample.edge_features, dtype=torch.float32, device=device),
        torch.as_tensor(sample.target_delta, dtype=torch.float32, device=device),
        torch.as_tensor(sample.initial_xyz, dtype=torch.float32, device=device),
        torch.as_tensor(sample.bond_edges, dtype=torch.long, device=device),
        torch.as_tensor(sample.bond_length_targets, dtype=torch.float32, device=device),
    )


def run_epoch(model: DeltaGNN, samples: list[GraphSample], indices: np.ndarray, device: torch.device, args: argparse.Namespace, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    order = np.array(indices, copy=True)
    if is_train:
        np.random.shuffle(order)
    for idx in order:
        node, edge_index, edge_feat, target, initial_xyz, bond_edges, bond_targets = sample_to_device(samples[int(idx)], device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(node, edge_index, edge_feat)
        coord_loss = torch.nn.functional.smooth_l1_loss(pred, target, beta=0.02)
        loss = coord_loss
        if args.bond_loss_weight > 0 and bond_edges.numel() > 0:
            pred_final = initial_xyz + pred * args.box_size_nm
            pred_lengths = torch.linalg.norm(pred_final[bond_edges[:, 0]] - pred_final[bond_edges[:, 1]], dim=1)
            bond_loss = torch.nn.functional.smooth_l1_loss(pred_lengths, bond_targets, beta=0.5) / max(args.box_size_nm, 1.0)
            loss = loss + args.bond_loss_weight * bond_loss
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_split(model: DeltaGNN, samples: list[GraphSample], indices: np.ndarray, split: str, device: torch.device, args: argparse.Namespace) -> dict:
    model.eval()
    rmse = []
    centered_rmse = []
    rg_mae = []
    bond_mae = []
    bond_max = []
    with torch.no_grad():
        for idx in indices:
            sample = samples[int(idx)]
            pred_final = predict_final_xyz(model, sample, device, args)
            err = pred_final - sample.final_xyz
            rmse.append(float(np.sqrt(np.mean(np.sum(err**2, axis=1)))))
            pred_c = pred_final - pred_final.mean(axis=0)
            true_c = sample.final_xyz - sample.final_xyz.mean(axis=0)
            centered_rmse.append(float(np.sqrt(np.mean(np.sum((pred_c - true_c) ** 2, axis=1)))))
            rg_mae.append(abs(radius_of_gyration(pred_final) - radius_of_gyration(sample.final_xyz)))
            if len(sample.bond_edges):
                pred_lengths = np.linalg.norm(
                    pred_final[sample.bond_edges[:, 0]] - pred_final[sample.bond_edges[:, 1]],
                    axis=1,
                )
                bond_mae.append(float(np.mean(np.abs(pred_lengths - sample.bond_length_targets))))
                bond_max.append(float(np.max(pred_lengths)))
    return {
        "split": split,
        "n_states": int(len(indices)),
        "coordinate_rmse_nm": float(np.mean(rmse)) if rmse else None,
        "centered_coordinate_rmse_nm": float(np.mean(centered_rmse)) if centered_rmse else None,
        "radius_of_gyration_mae_nm": float(np.mean(rg_mae)) if rg_mae else None,
        "bond_length_mae_nm": float(np.mean(bond_mae)) if bond_mae else None,
        "bond_length_max_nm": float(np.mean(bond_max)) if bond_max else None,
    }


def predict_final_xyz(model: DeltaGNN, sample: GraphSample, device: torch.device, args: argparse.Namespace) -> np.ndarray:
    node, edge_index, edge_feat, _, _, _, _ = sample_to_device(sample, device)
    pred_delta = model(node, edge_index, edge_feat).detach().cpu().numpy() * args.box_size_nm
    pred_final = sample.initial_xyz + pred_delta
    return np.clip(pred_final, 0.0, args.box_size_nm).astype(np.float32)


def radius_of_gyration(points: np.ndarray) -> float:
    c = points.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((points - c) ** 2, axis=1))))


def save_predictions(model: DeltaGNN, samples: list[GraphSample], pred_dir: Path, device: torch.device, args: argparse.Namespace) -> None:
    rows = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            pred_final = predict_final_xyz(model, sample, device, args)
            coords_path = pred_dir / f"{sample.name}_delta_gnn_predicted_final.txt"
            npz_path = pred_dir / f"{sample.name}_delta_gnn_predicted_final.npz"
            np.savetxt(coords_path, pred_final, header="x(nm) y(nm) z(nm)", comments="", fmt="%.6f")
            np.savez_compressed(
                npz_path,
                initial_xyz=sample.initial_xyz,
                true_final_xyz=sample.final_xyz,
                predicted_final_xyz=pred_final,
                xyz=pred_final,
                source_name=sample.name,
                nd=np.asarray(args.nd, dtype=np.int32),
                box_size_nm=np.asarray(args.box_size_nm, dtype=np.float32),
            )
            rows.append({"state": sample.name, "predicted_coordinates_txt": str(coords_path), "predicted_npz": str(npz_path)})
    pd.DataFrame(rows).to_csv(pred_dir / "delta_xyz_prediction_manifest.csv", index=False)


def render_predictions(project: Path, pred_dir: Path, figures: Path, args: argparse.Namespace) -> None:
    visualizer = project / "visualize_cluster_size.py"
    if not visualizer.exists():
        print(f"Skipping render: missing {visualizer}")
        return
    env = dict(os.environ)
    env["PYVISTA_OFF_SCREEN"] = "true"
    env["MPLCONFIGDIR"] = "/private/tmp/mpl-cache"
    for coords in sorted(pred_dir.glob("*_delta_gnn_predicted_final.txt")):
        name = coords.name.replace("_delta_gnn_predicted_final.txt", "")
        conn = connection_path_for(project / f"{name}.txt")
        if not conn.exists():
            # Prediction geometry is the thing being tested here. If the true
            # connection file is missing, skip rendering instead of faking bonds.
            continue
        out = figures / f"{name}_delta_gnn_predicted_final.png"
        cmd = [
            str(Path(sys_executable())),
            str(visualizer),
            str(coords),
            "--connections",
            str(conn),
            "--nd",
            str(args.nd),
            "--radius",
            "0.8",
            "--show-bonds",
            "--fast-points",
            "--pore-sphere",
            "0",
            "--screenshot",
            str(out),
        ]
        subprocess.run(cmd, cwd=str(project), env=env, check=True)


def sys_executable() -> str:
    import sys

    return sys.executable


if __name__ == "__main__":
    main()
