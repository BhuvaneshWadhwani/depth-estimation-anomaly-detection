import torch
import torchvision
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, AutoModelForDepthEstimation, AutoModel
import matplotlib.colors as mcolors
import time
from sklearn.metrics import roc_auc_score


####################### Task 1: Setup and Data Preparation #######################

####### Data Information #######
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}")


NUM_IMAGES = 100

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent



####################### NYU_Depth_v2 #######################

nyu_ds = load_dataset("tanganke/nyuv2", split="val")
nyu_ds = nyu_ds.select(range(NUM_IMAGES))

nyu_ds = nyu_ds.with_format("torch")

print(f"NYU validation images used: {len(nyu_ds)}")

nyu_image = np.array(nyu_ds[0]['image'])
nyu_image = nyu_image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
nyu_depth = np.array(nyu_ds[0]['depth']).squeeze()  # Remove channel dimension if it exists

print(f"Image count: {len(nyu_ds)}")
print(f"Image shape: {nyu_image.shape}")
print(f"Depth shape: {nyu_depth.shape}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(nyu_image)
plt.title("NYU RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(nyu_depth, cmap="inferno")
plt.title("Ground Truth Depth Map")
plt.axis("off")
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nyu_example_visualization.png")
plt.close()

print("Saved visualization as: nyu_example_visualization.png")



####################### ADE20K #######################

ADE_DIR = BASE_DIR / "data" / "ADEChallengeData2016"

ADE_IMG_DIR = ADE_DIR / "images" / "validation"
ADE_MASK_DIR = ADE_DIR / "annotations" / "validation"

ade_image_paths = sorted(ADE_IMG_DIR.glob("*.jpg"))[:NUM_IMAGES]
ade_mask_paths = sorted(ADE_MASK_DIR.glob("*.png"))[:NUM_IMAGES]

print(f"ADE20K validation images used: {len(ade_image_paths)}")
print(f"ADE20K validation masks used: {len(ade_mask_paths)}")

ade_image = np.array(Image.open(ade_image_paths[0]).convert("RGB"))
ade_mask = np.array(Image.open(ade_mask_paths[0]))

print(f"ADE20K example image shape: {ade_image.shape}")
print(f"ADE20K example mask shape: {ade_mask.shape}")

ADE_NUM_CLASSES = 150
print(f"ADE20K number of classes: {ADE_NUM_CLASSES}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(ade_image)
plt.title("ADE20K RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ade_mask, cmap="tab20")
plt.title("Ground Truth Segmentation Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ade20k_example_visualization.png", dpi=200)
plt.close()

print("Saved visualization as: ade20k_example_visualization.png")


####################### MVTec-AD #######################
MVTEC_DIR = BASE_DIR / "data" / "mvtec_anomaly_detection"

MVTEC_CATEGORIES = ["leather", "bottle", "transistor"]

for category in MVTEC_CATEGORIES:
    category_dir = MVTEC_DIR / category

    train_good_dir = category_dir / "train" / "good"
    test_good_dir = category_dir / "test" / "good"
    test_dir = category_dir / "test"

    train_good = sorted(train_good_dir.glob("*.png"))
    test_good = sorted(test_good_dir.glob("*.png"))

    defective_test = []
    for defect_folder in test_dir.iterdir():
        if defect_folder.is_dir() and defect_folder.name != "good":
            defective_test.extend(sorted(defect_folder.glob("*.png")))

    print(f"\nMVTec category: {category}")
    print(f"Nominal train images: {len(train_good)}")
    print(f"Nominal test images: {len(test_good)}")
    print(f"Defective test images: {len(defective_test)}")

# Example visualization: one defective image with its mask
example_category = "bottle"

example_defect_dir = next(
    d for d in (MVTEC_DIR / example_category / "test").iterdir()
    if d.is_dir() and d.name != "good"
)

example_image_path = sorted(example_defect_dir.glob("*.png"))[0]

example_mask_dir = MVTEC_DIR / example_category / "ground_truth" / example_defect_dir.name
example_mask_path = sorted(example_mask_dir.glob("*.png"))[0]

mvtec_image = np.array(Image.open(example_image_path).convert("RGB"))
mvtec_mask = np.array(Image.open(example_mask_path).convert("L"))

print(f"\nMVTec example image shape: {mvtec_image.shape}")
print(f"MVTec example mask shape: {mvtec_mask.shape}")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(mvtec_image)
plt.title("MVTec Defective Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mvtec_mask, cmap="gray")
plt.title("Ground Truth Anomaly Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "mvtec_example_visualization.png", dpi=200)
plt.close()

print("Saved visualization as: mvtec_example_visualization.png")


####################### Task 2: Semantic Segmentation #######################
SEG_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
IGNORE_INDEX = 0

seg_device = "cpu"
seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL_NAME).to(seg_device)
seg_processor = AutoImageProcessor.from_pretrained(SEG_MODEL_NAME)
seg_model.eval()

for param in seg_model.parameters():
    param.requires_grad = False

intersection = torch.zeros(ADE_NUM_CLASSES, dtype=torch.float32)
union = torch.zeros(ADE_NUM_CLASSES, dtype=torch.float32)

pred_masks_for_vis = []
gt_masks_for_vis = []
rgb_images_for_vis = []

with torch.no_grad():
    for idx, (img_path, mask_path) in enumerate(zip(ade_image_paths, ade_mask_paths)):
        image_pil = Image.open(img_path).convert("RGB")
        gt_raw = np.array(Image.open(mask_path))

        inputs = seg_processor(images=image_pil, return_tensors="pt").to(seg_device)

        outputs = seg_model(**inputs)
        logits = outputs.logits

        logits_up = F.interpolate(
            logits,
            size=gt_raw.shape,
            mode="bilinear",
            align_corners=False
        )

        pred = logits_up.argmax(dim=1).squeeze(0).cpu()

        gt_raw_tensor = torch.tensor(gt_raw, dtype=torch.long)

        valid_mask = gt_raw_tensor != IGNORE_INDEX
        gt = gt_raw_tensor - 1

        for class_id in range(ADE_NUM_CLASSES):
            pred_c = (pred == class_id) & valid_mask
            gt_c = (gt == class_id) & valid_mask

            intersection[class_id] += (pred_c & gt_c).sum().item()
            union[class_id] += (pred_c | gt_c).sum().item()

        if idx < 5:
            rgb_images_for_vis.append(np.array(image_pil))
            pred_masks_for_vis.append(pred.numpy())

            gt_vis = gt.numpy()
            gt_vis[gt_raw == IGNORE_INDEX] = -1
            gt_masks_for_vis.append(gt_vis)

valid_classes = union > 0
ious = intersection[valid_classes] / union[valid_classes]
miou = ious.mean().item()

print(f"ADE20K mIoU over {len(ade_image_paths)} validation images: {miou:.4f}")

# Visualization: 5 examples, RGB / prediction / ground truth

rng = np.random.default_rng(42)
colors = rng.random((ADE_NUM_CLASSES, 3))
cmap = mcolors.ListedColormap(colors)

fig, axes = plt.subplots(5, 3, figsize=(12, 18))

for i in range(5):
    axes[i, 0].imshow(rgb_images_for_vis[i])
    axes[i, 0].set_title("RGB Input")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(pred_masks_for_vis[i], cmap=cmap, vmin=0, vmax=ADE_NUM_CLASSES - 1)
    axes[i, 1].set_title("Predicted Mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(gt_masks_for_vis[i], cmap=cmap, vmin=0, vmax=ADE_NUM_CLASSES - 1)
    axes[i, 2].set_title("Ground Truth Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ade20k_segmentation_examples.png", dpi=200)
plt.close()

print("Saved segmentation visualization as: ade20k_segmentation_examples.png")




####################### Task 3: Monocular Depth Estimation #######################
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

depth_device = "cpu"
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(depth_device)
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
depth_model.eval()

for param in depth_model.parameters():
    param.requires_grad = False

absrel_sum = 0.0
valid_pixel_count = 0

depth_rgb_vis = []
depth_pred_vis = []
depth_gt_vis = []

total_inference_time = 0.0

with torch.no_grad():
    for idx in range(len(nyu_ds)):
        example = nyu_ds[idx]

        image_tensor = example["image"]   # [3, H, W]
        gt_depth = example["depth"].squeeze().float()   # [H, W]

        image_np = image_tensor.numpy().transpose(1, 2, 0)

        if image_np.max() <= 1.0:
            image_np = (image_np * 255).clip(0, 255)

        image_np = image_np.astype(np.uint8)
        image_pil = Image.fromarray(image_np)

        inputs = depth_processor(images=image_pil, return_tensors="pt").to(depth_device)

        start_time = time.perf_counter()
        outputs = depth_model(**inputs)
        end_time = time.perf_counter()

        total_inference_time += end_time - start_time

        pred_depth = outputs.predicted_depth

        pred_depth = F.interpolate(
            pred_depth.unsqueeze(1),
            size=gt_depth.shape,
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu()

        valid_mask = gt_depth > 0

        pred_valid = pred_depth[valid_mask]
        gt_valid = gt_depth[valid_mask]

        scale = torch.median(gt_valid) / torch.median(pred_valid)
        pred_aligned = scale * pred_depth

        absrel = torch.abs(pred_aligned[valid_mask] - gt_valid) / gt_valid

        absrel_sum += absrel.sum().item()
        valid_pixel_count += valid_mask.sum().item()

        if idx < 5:
            depth_rgb_vis.append(image_np)
            depth_pred_vis.append(pred_aligned.numpy())
            depth_gt_vis.append(gt_depth.numpy())

mean_absrel = absrel_sum / valid_pixel_count
runtime_per_image = total_inference_time / len(nyu_ds)

print(f"NYU Depth Anything V2 mean AbsRel over {len(nyu_ds)} images: {mean_absrel:.4f}")
print(f"Total depth inference time: {total_inference_time:.2f} seconds")
print(f"Runtime per image: {runtime_per_image:.4f} seconds/image")

# Visualization: 5 examples, RGB / predicted aligned depth / ground truth depth

fig, axes = plt.subplots(5, 3, figsize=(12, 18))

for i in range(5):
    row_min = min(depth_pred_vis[i].min(), depth_gt_vis[i].min())
    row_max = max(depth_pred_vis[i].max(), depth_gt_vis[i].max())

    axes[i, 0].imshow(depth_rgb_vis[i])
    axes[i, 0].set_title("RGB Input")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(depth_pred_vis[i], cmap="inferno", vmin=row_min, vmax=row_max)
    axes[i, 1].set_title("Predicted Aligned Depth")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(depth_gt_vis[i], cmap="inferno", vmin=row_min, vmax=row_max)
    axes[i, 2].set_title("Ground Truth Depth")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "nyu_depth_estimation_examples.png", dpi=200)
plt.close()

print("Saved depth visualization as: nyu_depth_estimation_examples.png")



####################### Task 4: Anomaly Detection with DINOv2 Features #######################

DINO_MODEL_NAME = "facebook/dinov2-small"
MVTEC_CATEGORIES = ["leather", "bottle", "transistor"]

dino_device = "cpu"
dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(dino_device)
dino_model.eval()

for p in dino_model.parameters():
    p.requires_grad = False


def load_rgb(path):
    return Image.open(path).convert("RGB")


def extract_patch_features(image_path):
    image = load_rgb(image_path)
    inputs = dino_processor(images=image, return_tensors="pt").to(dino_device)

    with torch.no_grad():
        outputs = dino_model(**inputs)

    tokens = outputs.last_hidden_state[:, 1:, :]  # remove CLS token
    tokens = F.normalize(tokens.squeeze(0), p=2, dim=1)

    num_patches = tokens.shape[0]
    grid_size = int(num_patches ** 0.5)

    return tokens.cpu(), grid_size, image.size[::-1], np.array(image)


def min_cosine_distance_to_memory(test_features, memory_bank, chunk_size=4096):
    min_distances = []

    for start in range(0, test_features.shape[0], chunk_size):
        chunk = test_features[start:start + chunk_size]

        similarities = chunk @ memory_bank.T
        max_similarities = similarities.max(dim=1).values
        distances = 1.0 - max_similarities

        min_distances.append(distances)

    return torch.cat(min_distances, dim=0)


all_category_aurocs = []

for category in MVTEC_CATEGORIES:
    print(f"\nDINOv2 MVTec category: {category}")

    category_dir = MVTEC_DIR / category
    train_good_dir = category_dir / "train" / "good"
    test_dir = category_dir / "test"
    gt_dir = category_dir / "ground_truth"

    train_good_paths = sorted(train_good_dir.glob("*.png"))

    memory_features = []

    memory_start = time.perf_counter()

    for path in train_good_paths:
        feats, _, _, _ = extract_patch_features(path)
        memory_features.append(feats)

    memory_bank = torch.cat(memory_features, dim=0)
    memory_bank = F.normalize(memory_bank, p=2, dim=1)

    memory_time = time.perf_counter() - memory_start

    print(f"Memory bank size: {memory_bank.shape[0]} patches")
    print(f"Memory bank feature dimension: {memory_bank.shape[1]}")
    print(f"Memory bank build time: {memory_time:.2f} seconds")

    test_image_paths = []
    test_labels = []
    test_defect_types = []

    for defect_type_dir in sorted(test_dir.iterdir()):
        if not defect_type_dir.is_dir():
            continue

        defect_type = defect_type_dir.name
        paths = sorted(defect_type_dir.glob("*.png"))

        for p in paths:
            test_image_paths.append(p)
            test_defect_types.append(defect_type)
            test_labels.append(0 if defect_type == "good" else 1)

    image_scores = []
    vis_items = []

    inference_start = time.perf_counter()

    for image_path, label, defect_type in zip(test_image_paths, test_labels, test_defect_types):
        test_feats, grid_size, input_hw, rgb_np = extract_patch_features(image_path)

        patch_scores = min_cosine_distance_to_memory(test_feats, memory_bank)
        image_score = patch_scores.max().item()
        image_scores.append(image_score)

        heatmap_small = patch_scores.reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)

        heatmap = F.interpolate(
            heatmap_small,
            size=input_hw,
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        if len(vis_items) < 3 and (label == 1 or len(vis_items) == 0):
            mask_np = None

            if label == 1:
                mask_dir = gt_dir / defect_type
                mask_candidates = sorted(mask_dir.glob(image_path.stem + "_mask.png"))

                if len(mask_candidates) > 0:
                    mask_np = np.array(Image.open(mask_candidates[0]).convert("L"))

            vis_items.append({
                "rgb": rgb_np,
                "heatmap": heatmap,
                "mask": mask_np,
                "label": label,
                "defect_type": defect_type
            })

    inference_time = time.perf_counter() - inference_start

    auroc = roc_auc_score(test_labels, image_scores)
    all_category_aurocs.append(auroc)

    print(f"{category} image-level AUROC: {auroc:.4f}")
    print(f"{category} total inference time: {inference_time:.2f} seconds")

    # Visualization: 3 examples per category
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i, item in enumerate(vis_items):
        rgb = item["rgb"]
        heatmap = item["heatmap"]
        mask = item["mask"]

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"RGB: {item['defect_type']}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb)
        axes[i, 1].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[i, 1].set_title("Anomaly Heatmap Overlay")
        axes[i, 1].axis("off")

        if mask is not None:
            axes[i, 2].imshow(mask, cmap="gray")
            axes[i, 2].set_title("GT Mask")
        else:
            axes[i, 2].imshow(np.zeros(rgb.shape[:2]), cmap="gray")
            axes[i, 2].set_title("GT Mask: none")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"mvtec_dinov2_{category}_heatmaps.png", dpi=200)
    plt.close()

    print(f"Saved heatmap visualization: mvtec_dinov2_{category}_heatmaps.png")

mean_auroc = np.mean(all_category_aurocs)
print(f"\nMean MVTec image-level AUROC across categories: {mean_auroc:.4f}")

