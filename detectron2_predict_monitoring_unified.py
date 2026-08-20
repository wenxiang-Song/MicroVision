# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Unified Detectron2 prediction + monitoring + optional anomaly detection for images or videos.

Input behavior:
- If --input_path is a directory, run batch image prediction.
- If --input_path is a video file, run sampled video prediction.
- --show_label_score controls whether confidence percentages are drawn after labels
  for both image and video visualization outputs.

Video behavior:
- If --max_predict_frames is 0, every input frame is predicted and written.
- If --max_predict_frames is positive, frames are sampled evenly from the full video.
- Only sampled frames are predicted and written, so the output MP4 frame count equals
  the sampled frame count when the input has enough frames.
- If --save_pred_frames is 1, every successfully predicted sampled frame is also
  saved as an individually named JPEG image under the output directory.
- Output MP4 uses ffmpeg H.264 + yuv420p + faststart for better VS Code/browser compatibility.

Anomaly behavior:
- --enable_anomaly 1: apply label / morphology / confidence anomaly rules.
- --enable_anomaly 0: ignore anomaly-rule parameters and draw normal class colors only.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-detectron2")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode

# Reuse the existing image anomaly script as the shared utility module.
# Keep detectron2_batch_predict_monitoring_anomaly.py in the same project folder.
import detectron2_batch_predict_monitoring_anomaly as image_anomaly


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "configs").is_dir() and (SCRIPT_DIR / "detectron2").is_dir():
    PROJECT_ROOT = SCRIPT_DIR
else:
    PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CFG_PATH = PROJECT_ROOT / "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "finetune_weights/model_final_copypaste.pth"

CLASS_NAMES = image_anomaly.CLASS_NAMES
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTS = image_anomaly.IMAGE_EXTS
MORPH_METRICS = image_anomaly.MORPH_METRICS
TARGET_GROUP_VALID_LABELS = image_anomaly.TARGET_GROUP_VALID_LABELS

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v"}


def parse_bool_int(value):
    value = str(value).strip().lower()
    if value in ["1", "true", "yes", "y"]:
        return True
    if value in ["0", "false", "no", "n"]:
        return False
    raise ValueError(f"Invalid bool value: {value}. Use 1/0, true/false, yes/no.")


def parse_args():
    parser = argparse.ArgumentParser(
        "Unified Detectron2 image/video prediction + monitoring + optional anomaly detection"
    )

    # Unified input/output
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input image directory or input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )

    # Basic prediction parameters
    parser.add_argument(
        "--weights_path",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Detectron2 model weights path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument(
        "--show_label_score",
        "--show_label_percentage",
        dest="show_label_score",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Images and videos. Show confidence percentage after each class label: "
            "1=yes, 0=show class label only."
        ),
    )

    # Video-only parameters
    parser.add_argument(
        "--max_predict_frames",
        type=int,
        default=200,
        help=(
            "Video only. Maximum sampled frames to predict and write. "
            "0 means every frame. If input_path is a directory, this is ignored."
        ),
    )
    parser.add_argument(
        "--output_video_name",
        type=str,
        default="predicted_monitoring_anomaly_h264.mp4",
        help="Video only. Fixed output video name.",
    )
    parser.add_argument(
        "--fallback_fps",
        type=float,
        default=25.0,
        help="Video only. FPS used when the input video does not report valid FPS.",
    )
    parser.add_argument(
        "--draw_summary",
        type=int,
        default=1,
        help="Video only. Draw frame-level monitoring panel: 1=yes, 0=no.",
    )
    parser.add_argument(
        "--save_pred_frames",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Video only. Save every successfully predicted sampled frame as an "
            "individual image: 1=yes, 0=no."
        ),
    )
    parser.add_argument(
        "--pred_frames_dir_name",
        type=str,
        default="predicted_frames",
        help="Video only. Subdirectory name for individually saved prediction frames.",
    )
    parser.add_argument(
        "--pred_frame_jpeg_quality",
        type=int,
        default=95,
        help="Video only. JPEG quality for saved prediction frames, from 1 to 100.",
    )

    # Anomaly switch
    parser.add_argument(
        "--enable_anomaly",
        type=int,
        default=1,
        choices=[0, 1],
        help="1 enables anomaly detection; 0 disables and ignores anomaly-rule parameters.",
    )

    # Anomaly condition 1: label anomaly
    parser.add_argument(
        "--target_group",
        type=str,
        default="Crystal",
        choices=["Crystal", "Droplet", "Microsphere", "crystal", "droplet", "microsphere"],
        help="Target group for label anomaly detection. Ignored when --enable_anomaly 0.",
    )

    # Anomaly condition 2: morphology anomaly
    parser.add_argument(
        "--morph_labels",
        type=str,
        default="none",
        help='Labels for morphology anomaly detection, e.g. "rod,plate". Ignored when --enable_anomaly 0.',
    )
    parser.add_argument(
        "--morph_exclude_edge",
        type=str,
        default="1",
        help="Whether to exclude edge-touching instances from morphology detection.",
    )
    parser.add_argument(
        "--morph_exclude_contact",
        type=str,
        default="1",
        help="Whether to exclude mask-contact instances from morphology detection.",
    )

    # Important morphology ranges. Use none to ignore.
    parser.add_argument("--aspect_ratio_min", type=str, default="none")
    parser.add_argument("--aspect_ratio_max", type=str, default="none")
    parser.add_argument("--relative_area_min", type=str, default="none")
    parser.add_argument("--relative_area_max", type=str, default="none")
    parser.add_argument("--circularity_min", type=str, default="none")
    parser.add_argument("--circularity_max", type=str, default="none")
    parser.add_argument("--relative_diameter_min", type=str, default="none")
    parser.add_argument("--relative_diameter_max", type=str, default="none")

    # Additional morphology ranges, default none.
    parser.add_argument("--filling_ratio_min", type=str, default="none")
    parser.add_argument("--filling_ratio_max", type=str, default="none")
    parser.add_argument("--eccentricity_min", type=str, default="none")
    parser.add_argument("--eccentricity_max", type=str, default="none")
    parser.add_argument("--relative_major_axis_length_min", type=str, default="none")
    parser.add_argument("--relative_major_axis_length_max", type=str, default="none")
    parser.add_argument("--relative_minor_axis_length_min", type=str, default="none")
    parser.add_argument("--relative_minor_axis_length_max", type=str, default="none")

    # Anomaly condition 3: confidence anomaly
    parser.add_argument(
        "--abnormal_conf_thresh",
        type=float,
        default=0.5,
        help="Instances with confidence below this threshold are abnormal. Ignored when --enable_anomaly 0.",
    )

    # Abnormal visualization color
    parser.add_argument(
        "--abnormal_color",
        type=str,
        default="#FF0000",
        help="Mask/box color for abnormal instances, e.g. '#FF0000'. Ignored when --enable_anomaly 0.",
    )
    parser.add_argument(
        "--hide_abnormal",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Do not draw abnormal instances in output images/videos: 1=yes, 0=no. "
            "This only changes visualization; metrics and counts still include them. "
            "Ignored when --enable_anomaly 0."
        ),
    )

    return parser.parse_args()


def reset_output_dir(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"[INFO] Removing old output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_predictor(weights_path, device, score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(str(CFG_PATH))
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def build_metadata(prefix="microvision_unified"):
    metadata_name = f"{prefix}_{os.getpid()}"
    metadata = MetadataCatalog.get(metadata_name)
    metadata.set(thing_classes=CLASS_NAMES)
    return metadata


def get_anomaly_config(args):
    enable_anomaly = bool(int(args.enable_anomaly))
    hide_abnormal = bool(int(args.hide_abnormal))

    if not enable_anomaly:
        if hide_abnormal:
            print("[WARN] --hide_abnormal 1 is ignored because --enable_anomaly is 0.")
        return {
            "enable_anomaly": False,
            "hide_abnormal": False,
            "target_group": None,
            "morph_labels": [],
            "morph_ranges": {metric: {"min": None, "max": None} for metric in MORPH_METRICS},
            "morph_exclude_edge": True,
            "morph_exclude_contact": True,
            "abnormal_conf_thresh": None,
            "abnormal_color": image_anomaly.normalize_hex_color(args.abnormal_color),
        }

    target_group = args.target_group.strip().lower()
    if target_group not in TARGET_GROUP_VALID_LABELS:
        raise ValueError(
            f"Invalid --target_group: {args.target_group}. Choose Crystal, Droplet, or Microsphere."
        )

    morph_labels = image_anomaly.parse_label_list(args.morph_labels)
    morph_ranges = image_anomaly.build_morph_ranges(args)
    morph_exclude_edge = image_anomaly.parse_bool_int(args.morph_exclude_edge)
    morph_exclude_contact = image_anomaly.parse_bool_int(args.morph_exclude_contact)
    abnormal_conf_thresh = float(args.abnormal_conf_thresh)
    abnormal_color = image_anomaly.normalize_hex_color(args.abnormal_color)

    # Keep low-confidence statistics consistent with the confidence anomaly threshold.
    image_anomaly.LOW_CONFIDENCE_THRESH = abnormal_conf_thresh

    return {
        "enable_anomaly": True,
        "hide_abnormal": hide_abnormal,
        "target_group": target_group,
        "morph_labels": morph_labels,
        "morph_ranges": morph_ranges,
        "morph_exclude_edge": morph_exclude_edge,
        "morph_exclude_contact": morph_exclude_contact,
        "abnormal_conf_thresh": abnormal_conf_thresh,
        "abnormal_color": abnormal_color,
    }


def maybe_apply_anomaly(instance_rows, instances, anomaly_cfg):
    if not anomaly_cfg["enable_anomaly"]:
        return instance_rows, None

    instance_rows = image_anomaly.apply_anomaly_detection_to_rows(
        instance_rows=instance_rows,
        target_group=anomaly_cfg["target_group"],
        morph_labels=anomaly_cfg["morph_labels"],
        morph_ranges=anomaly_cfg["morph_ranges"],
        morph_exclude_edge=anomaly_cfg["morph_exclude_edge"],
        morph_exclude_contact=anomaly_cfg["morph_exclude_contact"],
        abnormal_conf_thresh=anomaly_cfg["abnormal_conf_thresh"],
    )

    abnormal_flags = image_anomaly.extract_abnormal_flags_for_instances(
        instances=instances,
        image_instance_rows=instance_rows,
    )
    return instance_rows, abnormal_flags


def prepare_visualization_instances(instances, abnormal_flags, hide_abnormal):
    """Optionally remove abnormal instances from visualization only."""
    if not hide_abnormal or abnormal_flags is None:
        return instances, abnormal_flags

    if len(abnormal_flags) != len(instances):
        raise ValueError(
            "abnormal_flags and instances must have the same length: "
            f"{len(abnormal_flags)} != {len(instances)}"
        )

    device = instances.scores.device if instances.has("scores") else "cpu"
    keep_mask = torch.as_tensor(
        [not bool(flag) for flag in abnormal_flags],
        dtype=torch.bool,
        device=device,
    )
    return instances[keep_mask], None


def configure_label_score_display(instances, show_label_score):
    """Return a visualization-only Instances object with optional score labels."""
    if bool(int(show_label_score)) or not instances.has("scores"):
        return instances

    visualization_instances = type(instances)(instances.image_size)
    for field_name, field_value in instances.get_fields().items():
        if field_name != "scores":
            visualization_instances.set(field_name, field_value)
    return visualization_instances


# ============================================================
# Image pipeline
# ============================================================

def collect_images(input_dir):
    input_dir = Path(input_dir)
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(input_dir.rglob(f"*{ext}"))
        image_paths.extend(input_dir.rglob(f"*{ext.upper()}"))
    image_paths = sorted(list(set(image_paths)))
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {input_dir}")
    return image_paths


def save_image_visualization(
    image_bgr,
    instances,
    metadata,
    save_path,
    abnormal_flags,
    abnormal_color,
    hide_abnormal,
    show_label_score,
):
    instances, abnormal_flags = prepare_visualization_instances(
        instances=instances,
        abnormal_flags=abnormal_flags,
        hide_abnormal=hide_abnormal,
    )
    instances = configure_label_score_display(instances, show_label_score)
    visualizer = image_anomaly.CustomVisualizer(
        image_bgr[:, :, ::-1],
        metadata=metadata,
        class_names=CLASS_NAMES,
        class_color_map=image_anomaly.class_color_map,
        abnormal_flags=abnormal_flags,
        abnormal_color=abnormal_color,
        mask_alpha=image_anomaly.MASK_ALPHA,
        box_line_width=image_anomaly.BOX_LINE_WIDTH,
        label_font_size_ratio=image_anomaly.LABEL_FONT_SIZE_RATIO,
        label_font_size_min=image_anomaly.LABEL_FONT_SIZE_MIN,
        label_font_size_max=image_anomaly.LABEL_FONT_SIZE_MAX,
        label_font_color=image_anomaly.LABEL_FONT_COLOR,
        label_bg_color=image_anomaly.LABEL_BG_COLOR,
        label_bg_alpha=image_anomaly.LABEL_BG_ALPHA,
        label_bg_pad=image_anomaly.LABEL_BG_PAD,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = visualizer.draw_instance_predictions(instances)
    vis_rgb = vis_output.get_image()
    vis_bgr = vis_rgb[:, :, ::-1]
    cv2.imwrite(str(save_path), vis_bgr)


def run_image_prediction(args, predictor, metadata, anomaly_cfg):
    input_dir = Path(args.input_path)
    output_dir = reset_output_dir(args.output_dir)

    pred_image_dir = output_dir / "pred_images"
    pred_image_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    failed_images = []
    all_instance_rows = []
    all_image_summary_rows = []

    print("=" * 80)
    print("Unified Detectron2 Image Prediction + Monitoring")
    print("=" * 80)
    print(f"Input dir       : {input_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Pred image dir  : {pred_image_dir}")
    print(f"Weights         : {args.weights_path}")
    print(f"Device          : {args.device}")
    print(f"Score thresh    : {args.score_thresh}")
    print(f"Show label score: {bool(int(args.show_label_score))}")
    print(f"Num images      : {len(image_paths)}")
    print(f"Anomaly enabled : {anomaly_cfg['enable_anomaly']}")
    if anomaly_cfg["enable_anomaly"]:
        print(f"Target group    : {anomaly_cfg['target_group']}")
        print(f"Morph labels    : {anomaly_cfg['morph_labels'] if anomaly_cfg['morph_labels'] else 'none'}")
        print(f"Morph ranges    : {anomaly_cfg['morph_ranges']}")
        print(f"Hide abnormal   : {anomaly_cfg['hide_abnormal']}")
        print(f"Abnormal color  : {anomaly_cfg['abnormal_color']}")
    print("=" * 80)

    for img_path in tqdm(image_paths, desc="Predicting images", unit="img", ncols=100):
        img_path = Path(img_path)
        rel_path = img_path.relative_to(input_dir)
        save_path = (pred_image_dir / rel_path).with_suffix(".jpg")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        im_bgr = cv2.imread(str(img_path))
        if im_bgr is None:
            failed_images.append(f"{img_path}\tcv2.imread failed")
            continue

        height, width = im_bgr.shape[:2]

        try:
            outputs = predictor(im_bgr)
            instances = outputs["instances"]

            instance_rows = image_anomaly.extract_instance_metrics(
                instances=instances,
                image_path=img_path,
                input_dir=input_dir,
                height=height,
                width=width,
            )

            instance_rows, abnormal_flags = maybe_apply_anomaly(
                instance_rows=instance_rows,
                instances=instances,
                anomaly_cfg=anomaly_cfg,
            )

            save_image_visualization(
                image_bgr=im_bgr,
                instances=instances,
                metadata=metadata,
                save_path=save_path,
                abnormal_flags=abnormal_flags,
                abnormal_color=anomaly_cfg["abnormal_color"],
                hide_abnormal=anomaly_cfg["hide_abnormal"],
                show_label_score=args.show_label_score,
            )

            image_summary_row = image_anomaly.summarize_image_metrics(
                instance_rows=instance_rows,
                image_path=img_path,
                input_dir=input_dir,
                height=height,
                width=width,
            )

            all_instance_rows.extend(instance_rows)
            all_image_summary_rows.append(image_summary_row)

        except Exception as exc:
            failed_images.append(f"{img_path}\t{exc}")

    instance_csv = output_dir / "instance_metrics.csv"
    image_summary_csv = output_dir / "image_summary_metrics.csv"

    instance_df = pd.DataFrame(all_instance_rows)
    image_summary_df = pd.DataFrame(all_image_summary_rows)

    if len(instance_df) > 0 and "label" in instance_df.columns:
        instance_df["label"] = pd.Categorical(
            instance_df["label"],
            categories=CLASS_NAMES,
            ordered=True,
        )
        sort_cols = [c for c in ["relative_path", "instance_index"] if c in instance_df.columns]
        if sort_cols:
            instance_df = instance_df.sort_values(sort_cols).reset_index(drop=True)

    if len(image_summary_df) > 0 and "relative_path" in image_summary_df.columns:
        image_summary_df = image_summary_df.sort_values(["relative_path"]).reset_index(drop=True)

    instance_df.to_csv(instance_csv, index=False, encoding="utf-8-sig")
    image_summary_df.to_csv(image_summary_csv, index=False, encoding="utf-8-sig")

    if failed_images:
        failed_txt = output_dir / "failed_images.txt"
        with open(failed_txt, "w", encoding="utf-8") as f:
            for item in failed_images:
                f.write(str(item) + "\n")
    else:
        failed_txt = None

    print("\n" + "=" * 80)
    print("Image prediction finished")
    print("=" * 80)
    print(f"Total images processed     : {len(image_paths)}")
    print(f"Image summary rows         : {len(image_summary_df)}")
    print(f"Instance metric rows       : {len(instance_df)}")
    print(f"Prediction images saved to : {pred_image_dir}")
    print(f"Instance CSV               : {instance_csv}")
    print(f"Image summary CSV          : {image_summary_csv}")
    print(f"Failed images              : {len(failed_images)}")
    if failed_txt is not None:
        print(f"Failed image list          : {failed_txt}")
    print("=" * 80)


# ============================================================
# Video pipeline
# ============================================================

def get_valid_fps(capture, fallback_fps):
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 1e-6:
        fps = float(fallback_fps)
    return fps


class FFmpegVideoWriter:
    """Write VS Code / browser friendly MP4 using ffmpeg H.264."""

    def __init__(self, output_video_path, fps, width, height, crf=18, preset="medium"):
        self.output_video_path = str(output_video_path)
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.released = False

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "没有检测到 ffmpeg，请先安装 ffmpeg：\n"
                "conda install -c conda-forge ffmpeg\n"
                "或者：sudo apt-get install ffmpeg"
            )

        if self.width % 2 != 0 or self.height % 2 != 0:
            raise ValueError(f"Output size must be even, got {self.width} x {self.height}")

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-crf", str(crf),
            "-preset", preset,
            self.output_video_path,
        ]

        self.process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr):
        if frame_bgr is None:
            return
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        frame_bgr = np.ascontiguousarray(frame_bgr)
        self.process.stdin.write(frame_bgr.tobytes())

    def release(self):
        if self.released:
            return
        self.released = True
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            stderr = self.process.stderr.read().decode("utf-8", errors="ignore")
            if self.process.stderr is not None:
                self.process.stderr.close()
            print(stderr)
            raise RuntimeError("ffmpeg 视频编码失败")
        if self.process.stderr is not None:
            self.process.stderr.close()

    def __del__(self):
        if getattr(self, "released", True):
            return
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def build_predict_frame_indices(frame_count, max_predict_frames):
    if frame_count <= 0:
        return None
    if max_predict_frames is None or max_predict_frames <= 0:
        return list(range(frame_count))
    if frame_count <= max_predict_frames:
        return list(range(frame_count))
    sampled = np.linspace(0, frame_count - 1, num=max_predict_frames)
    return [int(round(x)) for x in sampled]


def build_video_output_paths(input_video, output_dir, output_video_name):
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_video.stem
    return {
        "video": output_dir / output_video_name,
        "instance_csv": output_dir / f"instance_metrics_{stem}.csv",
        "frame_summary_csv": output_dir / f"frame_summary_metrics_{stem}.csv",
        "video_summary_csv": output_dir / f"video_summary_metrics_{stem}.csv",
        "failed_frames_csv": output_dir / f"failed_frames_{stem}.csv",
    }


def extract_frame_instance_metrics(instances, input_video, frame_index, time_sec, height, width):
    video_path = Path(input_video)
    frame_id = f"{video_path.stem}_frame_{frame_index:06d}"
    image_area = float(height * width)

    if len(instances) == 0 or not instances.has("pred_masks"):
        return []

    instances_cpu = instances.to("cpu")
    pred_masks = instances_cpu.pred_masks.numpy().astype(bool)
    pred_classes = instances_cpu.pred_classes.numpy().astype(int)
    scores = instances_cpu.scores.numpy().astype(float)

    masks = [pred_masks[i] for i in range(len(pred_masks))]
    contact_flags, contact_counts, contact_distance_px = image_anomaly.compute_contact_info(
        masks=masks,
        height=height,
        width=width,
    )

    rows = []
    for i, mask in enumerate(masks):
        metrics = image_anomaly.compute_mask_metrics(mask, height, width)
        if metrics is None:
            continue

        class_id = int(pred_classes[i])
        label = CLASS_NAMES[class_id]
        confidence = float(scores[i])

        row = {
            "video_name": video_path.name,
            "video_path": str(video_path),
            "frame_id": frame_id,
            "frame_index": int(frame_index),
            "time_sec": round(float(time_sec), 6),
            "image_width": int(width),
            "image_height": int(height),
            "image_area": image_area,
            "instance_index": int(i),
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
            "is_low_confidence": int(confidence < image_anomaly.LOW_CONFIDENCE_THRESH),
            "is_edge_touching": int(image_anomaly.is_edge_touching(mask, height, width)),
            "edge_ratio": image_anomaly.EDGE_RATIO,
            "is_mask_contact": int(contact_flags[i]),
            "contact_instance_count": int(contact_counts[i]),
            "contact_distance_px": int(contact_distance_px),
        }
        row.update(metrics)
        rows.append(row)

    return rows


def build_empty_frame_summary(input_video, frame_index, time_sec, height, width):
    video_path = Path(input_video)
    frame_id = f"{video_path.stem}_frame_{frame_index:06d}"
    row = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "frame_id": frame_id,
        "frame_index": int(frame_index),
        "time_sec": round(float(time_sec), 6),
        "image_width": int(width),
        "image_height": int(height),
        "image_area": float(height * width),
        "total_instance_count": 0,
        "density": 0.0,
        "total_low_confidence_count": 0,
        "total_low_confidence_ratio": 0.0,
        "total_abnormal_count": 0,
        "total_abnormal_ratio": 0.0,
        "label_abnormal_count": 0,
        "label_abnormal_ratio": 0.0,
        "morphology_abnormal_count": 0,
        "morphology_abnormal_ratio": 0.0,
        "confidence_abnormal_count": 0,
        "confidence_abnormal_ratio": 0.0,
    }
    for label in CLASS_NAMES:
        row[f"{label}_count"] = 0
        row[f"{label}_instance_ratio"] = 0.0
        row[f"{label}_area_sum"] = 0.0
        row[f"{label}_area_ratio"] = 0.0
        row[f"{label}_low_confidence_count"] = 0
        row[f"{label}_low_confidence_ratio"] = 0.0
        row[f"{label}_abnormal_count"] = 0
        row[f"{label}_abnormal_ratio"] = 0.0
        row[f"{label}_label_abnormal_count"] = 0
        row[f"{label}_morphology_abnormal_count"] = 0
        row[f"{label}_confidence_abnormal_count"] = 0
        for metric in MORPH_METRICS:
            row[f"{label}_mean_{metric}"] = np.nan
    return row


def summarize_frame_metrics(instance_rows, input_video, frame_index, time_sec, height, width):
    base_row = build_empty_frame_summary(
        input_video=input_video,
        frame_index=frame_index,
        time_sec=time_sec,
        height=height,
        width=width,
    )
    if len(instance_rows) == 0:
        return base_row

    df = pd.DataFrame(instance_rows)
    total_count = int(len(df))
    total_area = float(df["area"].sum())
    image_area = float(height * width)
    total_low_conf_count = int(df["is_low_confidence"].sum())

    base_row["total_instance_count"] = total_count
    base_row["density"] = image_anomaly.safe_divide(total_area, image_area)
    base_row["total_low_confidence_count"] = total_low_conf_count
    base_row["total_low_confidence_ratio"] = image_anomaly.safe_divide(total_low_conf_count, total_count)

    if "is_abnormal" in df.columns:
        base_row["total_abnormal_count"] = int(df["is_abnormal"].sum())
        base_row["total_abnormal_ratio"] = image_anomaly.safe_divide(base_row["total_abnormal_count"], total_count)
        base_row["label_abnormal_count"] = int(df["is_label_abnormal"].sum())
        base_row["label_abnormal_ratio"] = image_anomaly.safe_divide(base_row["label_abnormal_count"], total_count)
        base_row["morphology_abnormal_count"] = int(df["is_morphology_abnormal"].sum())
        base_row["morphology_abnormal_ratio"] = image_anomaly.safe_divide(base_row["morphology_abnormal_count"], total_count)
        base_row["confidence_abnormal_count"] = int(df["is_confidence_abnormal"].sum())
        base_row["confidence_abnormal_ratio"] = image_anomaly.safe_divide(base_row["confidence_abnormal_count"], total_count)

    for label in CLASS_NAMES:
        sub = df[df["label"] == label]
        label_count = int(len(sub))
        label_area_sum = float(sub["area"].sum()) if label_count > 0 else 0.0
        label_low_conf_count = int(sub["is_low_confidence"].sum()) if label_count > 0 else 0

        base_row[f"{label}_count"] = label_count
        base_row[f"{label}_instance_ratio"] = image_anomaly.safe_divide(label_count, total_count)
        base_row[f"{label}_area_sum"] = label_area_sum
        base_row[f"{label}_area_ratio"] = image_anomaly.safe_divide(label_area_sum, total_area)
        base_row[f"{label}_low_confidence_count"] = label_low_conf_count
        base_row[f"{label}_low_confidence_ratio"] = image_anomaly.safe_divide(label_low_conf_count, label_count)

        if label_count > 0 and "is_abnormal" in sub.columns:
            base_row[f"{label}_abnormal_count"] = int(sub["is_abnormal"].sum())
            base_row[f"{label}_abnormal_ratio"] = image_anomaly.safe_divide(base_row[f"{label}_abnormal_count"], label_count)
            base_row[f"{label}_label_abnormal_count"] = int(sub["is_label_abnormal"].sum())
            base_row[f"{label}_morphology_abnormal_count"] = int(sub["is_morphology_abnormal"].sum())
            base_row[f"{label}_confidence_abnormal_count"] = int(sub["is_confidence_abnormal"].sum())

        for metric in MORPH_METRICS:
            base_row[f"{label}_mean_{metric}"] = float(sub[metric].mean()) if label_count > 0 else np.nan

    return base_row


def extract_frame_counts_from_summary(summary_row):
    return {label: int(summary_row.get(f"{label}_count", 0)) for label in CLASS_NAMES}


def visualize_frame(
    frame_bgr,
    metadata,
    instances,
    abnormal_flags,
    abnormal_color,
    hide_abnormal,
    show_label_score,
):
    instances, abnormal_flags = prepare_visualization_instances(
        instances=instances,
        abnormal_flags=abnormal_flags,
        hide_abnormal=hide_abnormal,
    )
    instances = configure_label_score_display(instances, show_label_score)
    visualizer = image_anomaly.CustomVisualizer(
        frame_bgr[:, :, ::-1],
        metadata=metadata,
        class_names=CLASS_NAMES,
        class_color_map=image_anomaly.class_color_map,
        abnormal_flags=abnormal_flags,
        abnormal_color=abnormal_color,
        mask_alpha=image_anomaly.MASK_ALPHA,
        box_line_width=image_anomaly.BOX_LINE_WIDTH,
        label_font_size_ratio=image_anomaly.LABEL_FONT_SIZE_RATIO,
        label_font_size_min=image_anomaly.LABEL_FONT_SIZE_MIN,
        label_font_size_max=image_anomaly.LABEL_FONT_SIZE_MAX,
        label_font_color=image_anomaly.LABEL_FONT_COLOR,
        label_bg_color=image_anomaly.LABEL_BG_COLOR,
        label_bg_alpha=image_anomaly.LABEL_BG_ALPHA,
        label_bg_pad=image_anomaly.LABEL_BG_PAD,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
    )
    vis_output = visualizer.draw_instance_predictions(instances)
    return np.ascontiguousarray(vis_output.get_image()[:, :, ::-1])


def split_summary_pairs(pairs, max_chars=54):
    lines = []
    current = ""
    for pair in pairs:
        candidate = pair if not current else f"{current}  {pair}"
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = pair
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_monitoring_panel(frame_bgr, frame_index, time_sec, summary_row, enable_anomaly):
    height, width = frame_bgr.shape[:2]
    total_count = int(summary_row.get("total_instance_count", 0))
    abnormal_count = int(summary_row.get("total_abnormal_count", 0))

    if enable_anomaly:
        lines = [f"frame {frame_index} | {time_sec:.2f}s | total {total_count} | abnormal {abnormal_count}"]
    else:
        lines = [f"frame {frame_index} | {time_sec:.2f}s | total {total_count}"]

    class_counts = extract_frame_counts_from_summary(summary_row)
    nonzero_pairs = [f"{name}:{class_counts[name]}" for name in CLASS_NAMES if class_counts[name] > 0]
    if nonzero_pairs:
        lines.extend(split_summary_pairs(nonzero_pairs))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.85, min(height, width) / 1200.0))
    thickness = max(1, int(round(font_scale * 2)))
    pad = max(8, int(round(min(height, width) * 0.012)))
    line_gap = max(6, int(round(font_scale * 8)))

    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    panel_width = min(width - 2 * pad, max(size[0] for size in text_sizes) + 2 * pad)
    panel_height = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1) + 2 * pad

    x1, y1 = pad, pad
    x2, y2 = min(width - pad, x1 + panel_width), min(height - pad, y1 + panel_height)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    frame_bgr = cv2.addWeighted(overlay, 0.78, frame_bgr, 0.22, 0)

    y = y1 + pad
    for line, size in zip(lines, text_sizes):
        y += size[1]
        cv2.putText(frame_bgr, line, (x1 + pad, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        y += line_gap

    return frame_bgr


def build_video_summary(
    input_video,
    output_video,
    frame_count,
    input_fps,
    output_fps,
    width,
    height,
    max_predict_frames,
    processed_frames,
    predicted_frames,
    failed_frames,
    instance_rows,
):
    video_path = Path(input_video)
    row = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "output_video": str(output_video),
        "input_fps": float(input_fps),
        "output_fps": float(output_fps),
        "frame_count": int(frame_count) if frame_count > 0 else np.nan,
        "duration_sec": image_anomaly.safe_divide(frame_count, input_fps) if frame_count > 0 else np.nan,
        "image_width": int(width),
        "image_height": int(height),
        "max_predict_frames": int(max_predict_frames),
        "processed_frames": int(processed_frames),
        "predicted_frames": int(predicted_frames),
        "reused_frames": 0,
        "raw_copied_frames": 0,
        "failed_frames": int(failed_frames),
        "total_instance_count": 0,
        "instances_per_predicted_frame": 0.0,
        "total_abnormal_count": 0,
        "total_abnormal_ratio": 0.0,
        "label_abnormal_count": 0,
        "label_abnormal_ratio": 0.0,
        "morphology_abnormal_count": 0,
        "morphology_abnormal_ratio": 0.0,
        "confidence_abnormal_count": 0,
        "confidence_abnormal_ratio": 0.0,
    }

    for label in CLASS_NAMES:
        row[f"{label}_count"] = 0
        row[f"{label}_instance_ratio"] = 0.0
        row[f"{label}_abnormal_count"] = 0
        row[f"{label}_abnormal_ratio"] = 0.0
        row[f"{label}_label_abnormal_count"] = 0
        row[f"{label}_morphology_abnormal_count"] = 0
        row[f"{label}_confidence_abnormal_count"] = 0
        for metric in MORPH_METRICS:
            row[f"{label}_mean_{metric}"] = np.nan

    if len(instance_rows) == 0:
        return row

    df = pd.DataFrame(instance_rows)
    total_count = int(len(df))
    row["total_instance_count"] = total_count
    row["instances_per_predicted_frame"] = image_anomaly.safe_divide(total_count, predicted_frames)

    if "is_abnormal" in df.columns:
        row["total_abnormal_count"] = int(df["is_abnormal"].sum())
        row["total_abnormal_ratio"] = image_anomaly.safe_divide(row["total_abnormal_count"], total_count)
        row["label_abnormal_count"] = int(df["is_label_abnormal"].sum())
        row["label_abnormal_ratio"] = image_anomaly.safe_divide(row["label_abnormal_count"], total_count)
        row["morphology_abnormal_count"] = int(df["is_morphology_abnormal"].sum())
        row["morphology_abnormal_ratio"] = image_anomaly.safe_divide(row["morphology_abnormal_count"], total_count)
        row["confidence_abnormal_count"] = int(df["is_confidence_abnormal"].sum())
        row["confidence_abnormal_ratio"] = image_anomaly.safe_divide(row["confidence_abnormal_count"], total_count)

    for label in CLASS_NAMES:
        sub = df[df["label"] == label]
        label_count = int(len(sub))
        row[f"{label}_count"] = label_count
        row[f"{label}_instance_ratio"] = image_anomaly.safe_divide(label_count, total_count)
        if label_count > 0 and "is_abnormal" in sub.columns:
            row[f"{label}_abnormal_count"] = int(sub["is_abnormal"].sum())
            row[f"{label}_abnormal_ratio"] = image_anomaly.safe_divide(row[f"{label}_abnormal_count"], label_count)
            row[f"{label}_label_abnormal_count"] = int(sub["is_label_abnormal"].sum())
            row[f"{label}_morphology_abnormal_count"] = int(sub["is_morphology_abnormal"].sum())
            row[f"{label}_confidence_abnormal_count"] = int(sub["is_confidence_abnormal"].sum())
        for metric in MORPH_METRICS:
            row[f"{label}_mean_{metric}"] = float(sub[metric].mean()) if label_count > 0 else np.nan

    return row


def save_video_csv_outputs(output_paths, instance_rows, frame_summary_rows, video_summary_row):
    instance_df = pd.DataFrame(instance_rows)
    frame_summary_df = pd.DataFrame(frame_summary_rows)
    video_summary_df = pd.DataFrame([video_summary_row])

    if len(instance_df) > 0 and "label" in instance_df.columns:
        instance_df["label"] = pd.Categorical(instance_df["label"], categories=CLASS_NAMES, ordered=True)
        instance_df = instance_df.sort_values(["frame_index", "instance_index"]).reset_index(drop=True)

    if len(frame_summary_df) > 0 and "frame_index" in frame_summary_df.columns:
        frame_summary_df = frame_summary_df.sort_values(["frame_index"]).reset_index(drop=True)

    instance_df.to_csv(output_paths["instance_csv"], index=False, encoding="utf-8-sig")
    frame_summary_df.to_csv(output_paths["frame_summary_csv"], index=False, encoding="utf-8-sig")
    video_summary_df.to_csv(output_paths["video_summary_csv"], index=False, encoding="utf-8-sig")


def run_video_prediction(args, predictor, metadata, anomaly_cfg):
    input_video = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_video_output_paths(input_video, output_dir, args.output_video_name)

    save_pred_frames = int(args.save_pred_frames) == 1
    pred_frame_dir = None
    if save_pred_frames:
        pred_frames_dir_name = Path(args.pred_frames_dir_name)
        if pred_frames_dir_name.is_absolute() or ".." in pred_frames_dir_name.parts:
            raise ValueError(
                "--pred_frames_dir_name must be a relative subdirectory name "
                "without '..'."
            )
        if not 1 <= int(args.pred_frame_jpeg_quality) <= 100:
            raise ValueError("--pred_frame_jpeg_quality must be between 1 and 100.")

        pred_frame_dir = output_dir / pred_frames_dir_name
        if pred_frame_dir.exists():
            print(f"[INFO] Removing old prediction frame dir: {pred_frame_dir}")
            shutil.rmtree(pred_frame_dir)
        pred_frame_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    input_fps = get_valid_fps(capture, args.fallback_fps)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Could not get valid frame count from: {input_video}")

    sampled_frame_indices = build_predict_frame_indices(frame_count, args.max_predict_frames)
    if sampled_frame_indices is None or len(sampled_frame_indices) == 0:
        capture.release()
        raise RuntimeError("No valid frames selected for processing.")

    planned_frames = len(sampled_frame_indices)
    input_duration = frame_count / input_fps
    output_fps = planned_frames / input_duration

    capture.set(cv2.CAP_PROP_POS_FRAMES, sampled_frame_indices[0])
    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Could not read first sampled frame from: {input_video}")

    input_height, input_width = first_frame.shape[:2]
    output_width = input_width - (input_width % 2)
    output_height = input_height - (input_height % 2)
    if output_width <= 0 or output_height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid output frame size: {output_width} x {output_height}")

    writer = FFmpegVideoWriter(
        output_video_path=output_paths["video"],
        fps=output_fps,
        width=output_width,
        height=output_height,
        crf=18,
        preset="medium",
    )

    print("=" * 80)
    print("Unified Detectron2 Video Prediction + Monitoring")
    print("=" * 80)
    print(f"Input video        : {input_video}")
    print(f"Output dir         : {output_dir}")
    print(f"Output video       : {output_paths['video']}")
    print(f"Weights            : {args.weights_path}")
    print(f"Device             : {args.device}")
    print(f"Score thresh       : {args.score_thresh}")
    print(f"Show label score   : {bool(int(args.show_label_score))}")
    print(f"Input FPS          : {input_fps:.3f}")
    print(f"Output FPS         : {output_fps:.3f}")
    print(f"Input frame count  : {frame_count}")
    print(f"Output frame count : {planned_frames}")
    print(f"Save pred frames   : {save_pred_frames}")
    if save_pred_frames:
        print(f"Pred frame dir     : {pred_frame_dir}")
        print(f"Pred JPEG quality  : {args.pred_frame_jpeg_quality}")
    print(f"Input duration     : {input_duration:.3f}s")
    print(f"Frame size         : {input_width} x {input_height} -> {output_width} x {output_height}")
    print(f"Anomaly enabled    : {anomaly_cfg['enable_anomaly']}")
    if anomaly_cfg["enable_anomaly"]:
        print(f"Target group       : {anomaly_cfg['target_group']}")
        print(f"Morph labels       : {anomaly_cfg['morph_labels'] if anomaly_cfg['morph_labels'] else 'none'}")
        print(f"Morph ranges       : {anomaly_cfg['morph_ranges']}")
        print(f"Hide abnormal      : {anomaly_cfg['hide_abnormal']}")
        print(f"Abnormal color     : {anomaly_cfg['abnormal_color']}")
    print("=" * 80)

    all_instance_rows = []
    all_frame_summary_rows = []
    failed_rows = []
    processed_frames = 0
    predicted_frames = 0
    saved_frame_images = 0
    failed_frames = 0
    total_instances = 0
    last_vis_bgr = None

    pbar = tqdm(total=planned_frames, desc="Predicting sampled frames", unit="frame", ncols=100)

    for output_index, frame_index in enumerate(sampled_frame_indices):
        time_sec = frame_index / input_fps
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()

        if not ok or frame_bgr is None:
            failed_frames += 1
            failed_rows.append({
                "frame_index": frame_index,
                "time_sec": round(float(time_sec), 6),
                "error": "cv2.VideoCapture read failed",
            })
            if last_vis_bgr is not None:
                writer.write(last_vis_bgr)
                processed_frames += 1
            pbar.update(1)
            continue

        try:
            if frame_bgr.shape[1] != output_width or frame_bgr.shape[0] != output_height:
                frame_bgr = cv2.resize(frame_bgr, (output_width, output_height))

            outputs = predictor(frame_bgr)
            instances = outputs["instances"]

            instance_rows = extract_frame_instance_metrics(
                instances=instances,
                input_video=input_video,
                frame_index=frame_index,
                time_sec=time_sec,
                height=output_height,
                width=output_width,
            )

            instance_rows, abnormal_flags = maybe_apply_anomaly(
                instance_rows=instance_rows,
                instances=instances,
                anomaly_cfg=anomaly_cfg,
            )

            frame_summary_row = summarize_frame_metrics(
                instance_rows=instance_rows,
                input_video=input_video,
                frame_index=frame_index,
                time_sec=time_sec,
                height=output_height,
                width=output_width,
            )

            instances_cpu = instances.to("cpu")
            vis_bgr = visualize_frame(
                frame_bgr=frame_bgr,
                metadata=metadata,
                instances=instances_cpu,
                abnormal_flags=abnormal_flags,
                abnormal_color=anomaly_cfg["abnormal_color"],
                hide_abnormal=anomaly_cfg["hide_abnormal"],
                show_label_score=args.show_label_score,
            )

            if int(args.draw_summary) == 1:
                vis_bgr = draw_monitoring_panel(
                    frame_bgr=vis_bgr,
                    frame_index=frame_index,
                    time_sec=time_sec,
                    summary_row=frame_summary_row,
                    enable_anomaly=anomaly_cfg["enable_anomaly"],
                )

            if save_pred_frames:
                prediction_number = output_index + 1
                source_frame_number = frame_index + 1
                frame_image_path = pred_frame_dir / (
                    f"prediction_{prediction_number:06d}_"
                    f"source_frame_{source_frame_number:06d}.jpg"
                )
                image_saved = cv2.imwrite(
                    str(frame_image_path),
                    vis_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.pred_frame_jpeg_quality)],
                )
                if not image_saved:
                    raise RuntimeError(
                        f"Could not save predicted frame image: {frame_image_path}"
                    )
                saved_frame_images += 1

            writer.write(vis_bgr)
            last_vis_bgr = vis_bgr

            all_instance_rows.extend(instance_rows)
            all_frame_summary_rows.append(frame_summary_row)
            processed_frames += 1
            predicted_frames += 1
            total_instances += len(instance_rows)

        except Exception as exc:
            failed_frames += 1
            failed_rows.append({
                "frame_index": frame_index,
                "time_sec": round(float(time_sec), 6),
                "error": str(exc),
            })
            if last_vis_bgr is not None:
                writer.write(last_vis_bgr)
                processed_frames += 1

        pbar.update(1)

    pbar.close()
    capture.release()
    writer.release()

    video_summary_row = build_video_summary(
        input_video=input_video,
        output_video=output_paths["video"],
        frame_count=frame_count,
        input_fps=input_fps,
        output_fps=output_fps,
        width=output_width,
        height=output_height,
        max_predict_frames=args.max_predict_frames,
        processed_frames=processed_frames,
        predicted_frames=predicted_frames,
        failed_frames=failed_frames,
        instance_rows=all_instance_rows,
    )

    save_video_csv_outputs(output_paths, all_instance_rows, all_frame_summary_rows, video_summary_row)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(output_paths["failed_frames_csv"], index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Video prediction finished")
    print("=" * 80)
    print(f"Processed frames  : {processed_frames}")
    print(f"Predicted frames  : {predicted_frames}")
    if save_pred_frames:
        print(f"Saved frame images: {saved_frame_images}")
        print(f"Pred frame dir    : {pred_frame_dir}")
    print(f"Output frames     : {planned_frames}")
    print(f"Total instances   : {total_instances}")
    print(f"Failed frames     : {failed_frames}")
    print(f"Output video      : {output_paths['video']}")
    print(f"Instance CSV      : {output_paths['instance_csv']}")
    print(f"Frame summary CSV : {output_paths['frame_summary_csv']}")
    print(f"Video summary CSV : {output_paths['video_summary_csv']}")
    if failed_rows:
        print(f"Failed frame CSV  : {output_paths['failed_frames_csv']}")
    print("=" * 80)


def detect_input_mode(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return "image_dir"
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        return "video"
    if input_path.is_file() and input_path.suffix.lower() in set(IMAGE_EXTS):
        raise ValueError(
            "当前统一脚本的图像模式输入是文件夹，不是单张图片。"
            "请传入图片文件夹，或传入视频文件。"
        )
    raise ValueError(
        f"Unsupported --input_path: {input_path}. "
        "Use an image directory or a video file such as .mp4."
    )


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    weights_path = Path(args.weights_path)

    assert weights_path.is_file(), f"Weights not found: {weights_path}"
    assert CFG_PATH.is_file(), f"Config not found: {CFG_PATH}"
    assert input_path.exists(), f"Input path not found: {input_path}"

    mode = detect_input_mode(input_path)
    anomaly_cfg = get_anomaly_config(args)

    predictor = build_predictor(
        weights_path=str(weights_path),
        device=args.device,
        score_thresh=args.score_thresh,
    )
    metadata = build_metadata(prefix=f"microvision_unified_{mode}")

    if mode == "image_dir":
        run_image_prediction(args, predictor, metadata, anomaly_cfg)
    elif mode == "video":
        run_video_prediction(args, predictor, metadata, anomaly_cfg)
    else:
        raise RuntimeError(f"Unknown input mode: {mode}")


if __name__ == "__main__":
    main()
