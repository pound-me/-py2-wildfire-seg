from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from baseline_runtime import PROJECT_ROOT, TotalLoss, build_dataset, build_model, load_config
from custom_losses import build_training_criterion


COLORS = ["#202020", "#8a8a8a", "#ef3b2c"]


def balanced_subset(features, labels, num_classes, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for class_index in range(num_classes):
        indices = np.flatnonzero(labels == class_index)
        if indices.size > per_class:
            indices = rng.choice(indices, size=per_class, replace=False)
        selected.append(indices)
    selected_indices = np.concatenate(selected)
    return features[selected_indices], labels[selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fixed validation DFM features and separability evidence."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pidnet_s_dfm_mproto_p4.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--max-points-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12007)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fused-feature export.")
    config = load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    device = torch.device(config["DEVICE"])
    model = build_model(config, augment=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    criterion = build_training_criterion(TotalLoss(config), config)
    criterion_state = checkpoint.get("criterion_state_dict")
    if criterion_state is None:
        raise RuntimeError("Checkpoint does not contain an EMA prototype bank.")
    criterion.load_state_dict(criterion_state)
    if not bool(criterion.bank.initialized.all()):
        missing = torch.nonzero(
            ~criterion.bank.initialized,
            as_tuple=False,
        ).tolist()
        raise RuntimeError(
            "Cannot export prototype evidence with uninitialized prototypes: "
            f"{missing}"
        )

    dataset = build_dataset(config, split="val")
    fixed_subset_samples = dataset.files[
        : min(max(args.max_images, 0), len(dataset))
    ]
    loader = DataLoader(
        dataset,
        batch_size=int(config["BATCHSIZE"]),
        shuffle=False,
        num_workers=0,
    )
    feature_parts, label_parts = [], []
    processed_images = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if processed_images >= args.max_images:
                break
            images, labels, component_maps = batch[0], batch[1], batch[4]
            remaining = args.max_images - processed_images
            images = images[:remaining]
            labels = labels[:remaining]
            component_maps = component_maps[:remaining]
            images = images.to(device=device, dtype=torch.float32)
            fused_feature = model(images)[3]
            sampled = criterion.bank.sample_features(
                fused_feature,
                labels,
                component_maps,
                sampling_seed=args.seed + batch_index,
            )
            feature_parts.append(sampled.features.detach().cpu().numpy())
            label_parts.append(sampled.class_ids.detach().cpu().numpy())
            processed_images += images.shape[0]

    features = np.concatenate(feature_parts, axis=0).astype(np.float32)
    labels = np.concatenate(label_parts, axis=0).astype(np.int64)
    features, labels = balanced_subset(
        features,
        labels,
        int(config["NUM_CLASSES"]),
        args.max_points_per_class,
        args.seed,
    )
    normalized = F.normalize(torch.from_numpy(features), dim=1).numpy()

    class_means = []
    centroid_distances = {}
    for class_index, class_name in enumerate(config["CLS_NAMES"]):
        class_features = normalized[labels == class_index]
        if class_features.shape[0] == 0:
            raise RuntimeError(f"No exported points for class {class_name}.")
        mean = class_features.mean(axis=0)
        mean /= max(np.linalg.norm(mean), 1e-12)
        class_means.append(mean)
        centroid_distances[class_name] = float(
            np.mean(1.0 - class_features @ mean)
        )
    class_means = np.stack(class_means, axis=0)
    cosine_similarity = class_means @ class_means.T
    cosine_distance = 1.0 - cosine_similarity
    silhouette = float(silhouette_score(normalized, labels, metric="cosine"))

    perplexity = min(30.0, max(5.0, (normalized.shape[0] - 1) / 3.0))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
    ).fit_transform(normalized)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else checkpoint_path.parent / "fused_feature_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "fused_features.npz",
        features=features,
        normalized_features=normalized,
        labels=labels,
        tsne=embedding,
        class_means=class_means,
        prototype_bank=criterion.bank.prototypes.detach().cpu().numpy(),
        prototype_initialized=criterion.bank.initialized.detach().cpu().numpy(),
    )

    plt.figure(figsize=(7, 6))
    for class_index, class_name in enumerate(config["CLS_NAMES"]):
        mask = labels == class_index
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=7,
            alpha=0.65,
            c=COLORS[class_index],
            label=class_name,
            linewidths=0,
        )
    plt.legend(frameon=False)
    plt.xticks([])
    plt.yticks([])
    plt.title("DFM fused-feature t-SNE (fixed validation subset)")
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_three_class.png", dpi=300)
    plt.close()

    prototype_flat = criterion.bank.prototypes.detach().cpu().reshape(
        -1, criterion.bank.feature_channels
    )
    prototype_cosine = (prototype_flat @ prototype_flat.T).numpy()
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": "validation",
        "fixed_subset_images": processed_images,
        "fixed_subset_samples": fixed_subset_samples,
        "fixed_seed": args.seed,
        "points_per_class": {
            class_name: int((labels == class_index).sum())
            for class_index, class_name in enumerate(config["CLS_NAMES"])
        },
        "silhouette_score_cosine": silhouette,
        "mean_cosine_distance_to_class_centroid": centroid_distances,
        "class_mean_cosine_similarity": cosine_similarity.tolist(),
        "class_mean_cosine_distance": cosine_distance.tolist(),
        "prototype_cosine_matrix": prototype_cosine.tolist(),
        "interpretation": (
            "t-SNE is qualitative only; quantitative conclusions should use "
            "IoU, cosine distances, silhouette score, and prototype statistics."
        ),
    }
    (output_dir / "separability_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Feature archive: {output_dir / 'fused_features.npz'}")
    print(f"t-SNE: {output_dir / 'tsne_three_class.png'}")


if __name__ == "__main__":
    main()
