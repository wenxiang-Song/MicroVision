# MicroVision 模型使用说明 / MicroVision Model Usage Guide

## 1. 模型使用：图像与视频分析、数据统计和异常检测运行命令 / Model Usage: Command for Image and Video Analysis, Statistical Summarization, and Anomaly Detection

```bash
!python /absolute/path/detectron2_predict_monitoring_unified.py \
  --input_path /absolute/path/input_folder_or_video \
  --output_dir /absolute/path/output_folder \
  --weights_path /absolute/path/model_final.pth \
  --device cuda:0 \
  --score_thresh 0.50 \
  --show_label_score 1 \
  --max_predict_frames 200 \
  --output_video_name predicted_monitoring_anomaly_h264.mp4 \
  --fallback_fps 25 \
  --draw_summary 1 \
  --save_pred_frames 0 \
  --pred_frames_dir_name predicted_frames \
  --pred_frame_jpeg_quality 95 \
  --enable_anomaly 1 \
  --target_group Crystal \
  --morph_labels none \
  --morph_exclude_edge 1 \
  --morph_exclude_contact 1 \
  --aspect_ratio_min none \
  --aspect_ratio_max none \
  --relative_area_min none \
  --relative_area_max none \
  --circularity_min none \
  --circularity_max none \
  --relative_diameter_min none \
  --relative_diameter_max none \
  --filling_ratio_min none \
  --filling_ratio_max none \
  --eccentricity_min none \
  --eccentricity_max none \
  --relative_major_axis_length_min none \
  --relative_major_axis_length_max none \
  --relative_minor_axis_length_min none \
  --relative_minor_axis_length_max none \
  --abnormal_conf_thresh 0.70 \
  --abnormal_color "#FF0000" \
  --hide_abnormal 0
```

在 Jupyter Notebook 中可使用上述 `!python` 形式；在 Linux 终端中运行时，删除命令开头的 `!`。

Use the `!python` form shown above in a Jupyter Notebook. When running the command in a Linux terminal, remove the leading `!`.

---

## 2. 使用说明 / Usage Instructions

### 2.1 脚本功能 / Script Overview

`detectron2_predict_monitoring_unified.py` 是统一的 MicroVision 预测与监测脚本，支持以下功能：

- 批量处理显微图像文件夹；
- 对视频均匀抽帧并执行实例分割；
- 计算实例级、图像级、帧级和视频级统计数据；
- 执行可选的类别、形态和置信度异常检测；
- 输出带标注的图像、视频和 CSV 数据文件。

`detectron2_predict_monitoring_unified.py` is the unified MicroVision prediction and monitoring script. It supports:

- Batch processing of microscopy image directories;
- Uniform frame sampling and instance segmentation of videos;
- Instance-, image-, frame-, and video-level statistical analysis;
- Optional label-, morphology-, and confidence-based anomaly detection;
- Export of annotated images, videos, and CSV data files.

运行前应确保 `detectron2_batch_predict_monitoring_anomaly.py` 与统一预测脚本位于同一项目目录中。

Before running the script, ensure that `detectron2_batch_predict_monitoring_anomaly.py` is located in the same project directory as the unified prediction script.

---

### 2.2 输入与输出参数 / Input and Output Parameters

#### `--input_path`

指定输入数据路径。支持以下两种输入形式：

- 输入图像文件夹时，脚本自动进入批量图像预测模式；
- 输入视频文件时，脚本自动进入视频预测模式。

图像模式会递归读取 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff` 和 `.webp` 文件。视频模式支持 `.mp4`、`.avi`、`.mov`、`.mkv`、`.mpeg`、`.mpg` 和 `.m4v` 文件。

该参数不直接支持单张图像文件。如果只需要预测一张图像，应先将其放入单独的文件夹中。

Specifies the input path. Two input types are supported:

- An image directory activates batch image prediction;
- A video file activates video prediction.

Image directories are searched recursively for `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp` files. Supported video formats are `.mp4`, `.avi`, `.mov`, `.mkv`, `.mpeg`, `.mpg`, and `.m4v`.

A single image file cannot be provided directly. To process one image, place it in a separate directory.

#### `--output_dir`

指定预测结果保存目录。

- 图像模式下，如果输出目录已经存在，脚本会删除整个目录并重新创建；
- 视频模式下，不会删除整个输出目录，但同名视频和 CSV 文件可能被覆盖；
- 启用逐帧保存时，已有的预测帧子目录会被删除并重新创建。

不要将其他重要文件保存在图像模式所使用的输出目录中。

Specifies the output directory.

- In image mode, an existing output directory is deleted and recreated;
- In video mode, the entire directory is retained, but files with identical names may be overwritten;
- When individual frame export is enabled, the existing prediction-frame subdirectory is deleted and recreated.

Do not store unrelated important files in an output directory used for image-mode prediction.

#### `--weights_path`

指定 Detectron2 模型权重文件路径，通常为 `.pth` 文件。如果不提供该参数，脚本使用项目默认权重。

Specifies the Detectron2 model weights, usually a `.pth` file. If this parameter is omitted, the default project weights are used.

#### `--device`

指定模型推理设备，例如：

```text
cuda:0    使用第1张GPU / Use the first GPU
cuda:3    使用第4张GPU / Use the fourth GPU
cpu       使用CPU / Use the CPU
```

使用 GPU 时，应确保对应编号的 GPU 存在且具有足够显存。

When using a GPU, ensure that the selected device exists and has sufficient memory.

---

### 2.3 基础预测参数 / Basic Prediction Parameters

#### `--score_thresh`

指定模型保留预测实例的最低置信度阈值，通常设置在 0–1 之间。只有置信度不低于该值的预测实例才会进入结果可视化、实例数量统计、形态参数计算和异常检测。

降低阈值通常会提高召回率，但可能增加误检；提高阈值通常会减少误检，但可能增加漏检。

Specifies the minimum confidence threshold for retaining a predicted instance, typically between 0 and 1. Only instances with confidence scores equal to or greater than this threshold are included in visualization, counting, morphological measurement, and anomaly detection.

A lower threshold generally increases recall but may introduce more false positives. A higher threshold generally reduces false positives but may increase missed detections.

#### `--show_label_score`

控制是否在类别标签后显示预测置信度：

```text
1    显示类别和置信度，例如 droplet 96% / Display the class and confidence
0    只显示类别，例如 droplet / Display the class only
```

该参数只影响可视化，不影响预测、统计或异常判断。

This parameter affects visualization only and does not change prediction, statistical analysis, or anomaly detection.

---

### 2.4 视频专用参数 / Video-Specific Parameters

#### `--max_predict_frames`

指定视频最多预测的帧数，默认值为 `200`。

```text
0          预测视频中的全部帧 / Process every frame
正整数     在完整视频中均匀抽取指定数量的帧 / Uniformly sample the specified number of frames
```

例如，设置为 `200` 表示从整个视频的时间范围内均匀抽取约 200 帧，而不是只预测前 200 帧。该参数在图像模式下无效。

For example, `200` means that approximately 200 frames are sampled uniformly across the entire video, rather than processing only the first 200 frames. This parameter is ignored in image mode.

#### `--output_video_name`

指定输出视频文件名，建议使用 `.mp4` 后缀。默认文件名为：

```text
predicted_monitoring_anomaly_h264.mp4
```

Specifies the output video filename. The `.mp4` extension is recommended. The default filename is:

```text
predicted_monitoring_anomaly_h264.mp4
```

#### `--fallback_fps`

当输入视频无法提供有效帧率时，指定使用的备用帧率，默认值为 `25.0`。如果输入视频帧率有效，该参数不会覆盖原始帧率信息。

Specifies the fallback frame rate used when the input video does not report a valid FPS. The default is `25.0`. If the input video provides a valid FPS, this parameter does not override it.

#### `--draw_summary`

控制是否在输出视频中绘制当前帧的统计信息面板：

```text
1    显示统计面板 / Display the summary panel
0    不显示统计面板 / Do not display the summary panel
```

该参数只影响视频可视化，不影响 CSV 统计结果。

This parameter affects video visualization only and does not change the CSV results.

#### `--save_pred_frames`

控制是否将每个成功预测的抽样帧单独保存为 JPEG 图像：

```text
1    保存 / Save frames
0    不保存 / Do not save frames
```

Controls whether each successfully processed sampled frame is also saved as an individual JPEG image.

#### `--pred_frames_dir_name`

指定单独预测帧的保存子目录名称，默认值为 `predicted_frames`。该名称必须是相对于 `output_dir` 的子目录，不能使用绝对路径，也不能包含 `..`。

Specifies the subdirectory used to store individually exported prediction frames. The default is `predicted_frames`. It must be a relative subdirectory under `output_dir` and cannot be an absolute path or contain `..`.

#### `--pred_frame_jpeg_quality`

指定单独保存预测帧时的 JPEG 质量，取值范围为 1–100，默认值为 `95`。数值越大，图像质量和文件体积通常越高。

Specifies the JPEG quality for individually saved prediction frames. The valid range is 1–100, and the default is `95`. Higher values generally produce better image quality and larger files.

---

### 2.5 异常检测总开关 / Anomaly Detection Switch

#### `--enable_anomaly`

控制是否启用异常检测：

```text
1    启用异常检测 / Enable anomaly detection
0    关闭异常检测 / Disable anomaly detection
```

设置为 `0` 时，类别、形态和置信度异常规则均不执行，但常规实例分割和形态数据统计仍会正常进行。

When set to `0`, label-, morphology-, and confidence-based anomaly rules are ignored. Standard instance segmentation and morphological measurements are still performed.

---

### 2.6 类别异常检测 / Label-Based Anomaly Detection

#### `--target_group`

指定当前样品所属的目标体系。可选值及对应允许类别如下：

```text
Crystal:
agg, block, plate, rod, bubble

Droplet:
droplet, bubble

Microsphere:
microsphere, bubble, agg
```

当模型预测类别不属于指定体系允许的类别时，该实例会被标记为类别异常。

Specifies the target particle system. A predicted instance is marked as a label anomaly if its class is not included in the permitted classes of the selected target system.

---

### 2.7 形态异常检测对象 / Classes Subjected to Morphological Screening

#### `--morph_labels`

指定需要进行形态异常检查的类别。例如：

```text
--morph_labels droplet
--morph_labels "rod,plate"
--morph_labels none
```

支持的类别为：

```text
agg, block, rod, plate, bubble, droplet, microsphere
```

未列入该参数的类别仍会被识别和统计，但不会进行形态异常判断。

Specifies which classes are subjected to morphological anomaly screening. Classes not listed here are still detected and included in the statistical outputs, but they are not evaluated using morphological anomaly rules.

#### `--morph_exclude_edge`

控制是否跳过接触图像边缘的实例：

```text
1    不对边缘实例进行形态异常判断 / Exclude edge-touching instances
0    边缘实例也参与形态异常判断 / Include edge-touching instances
```

被跳过的实例仍然保留在预测结果、实例计数和 CSV 文件中。

Excluded instances remain in the predictions, counts, and CSV outputs.

#### `--morph_exclude_contact`

控制是否跳过与其他实例发生掩膜接触的目标：

```text
1    不对接触实例进行形态异常判断 / Exclude contacting instances
0    接触实例也参与形态异常判断 / Include contacting instances
```

该参数只影响形态异常判断，不会删除实例。

This parameter affects morphological screening only and does not remove any instance.

---

### 2.8 形态学阈值 / Morphological Thresholds

所有形态参数均通过上下限定义正常范围：

All morphological parameters define a normal range using lower and upper limits:

```text
*_min    正常范围下限 / Lower bound of the normal range
*_max    正常范围上限 / Upper bound of the normal range
none     不设置该限制 / Do not apply this limit
```

异常判定规则如下：

The anomaly criteria are:

```text
value < min    异常 / Abnormal
value > max    异常 / Abnormal
```

必须保证 `min <= max`。如果某项指标的上下限均为 `none`，则不使用该指标进行异常判断。

The specified limits must satisfy `min <= max`. If both limits are `none`, that metric is not used for anomaly detection.

#### `--aspect_ratio_min` / `--aspect_ratio_max`

长径比定义为：

The aspect ratio is defined as:

$$
\mathrm{Aspect\ ratio}=\frac{\mathrm{major\ axis\ length}}{\mathrm{minor\ axis\ length}}
$$

该值通常不小于 1。越接近 1，目标越接近等轴形态；数值越大，目标越细长。

The value is generally not less than 1. Values close to 1 indicate an approximately equiaxed shape, whereas larger values indicate a more elongated shape.

#### `--relative_area_min` / `--relative_area_max`

相对面积定义为：

The relative area is defined as:

$$
\mathrm{Relative\ area}=\frac{\mathrm{mask\ area}}{\mathrm{image\ area}}
$$

该值是比例而不是百分数。例如，`0.01` 表示实例面积占整幅图像面积的 1%。

The value is expressed as a fraction rather than a percentage. For example, `0.01` means that the instance occupies 1% of the image area.

#### `--circularity_min` / `--circularity_max`

圆度定义为：

Circularity is defined as:

$$
\mathrm{Circularity}=\frac{4\pi A}{P^2}
$$

其中，$A$ 为掩膜面积，$P$ 为轮廓周长。理想圆形的圆度接近 1；数值越低，通常表示目标越细长或轮廓越不规则。

Here, $A$ is the mask area and $P$ is the perimeter. A perfect circle approaches 1, whereas lower values generally indicate a more elongated or irregular boundary.

#### `--relative_diameter_min` / `--relative_diameter_max`

等效直径和相对等效直径定义为：

The equivalent diameter and relative equivalent diameter are defined as:

$$
D_{\mathrm{eq}}=\sqrt{\frac{4A}{\pi}}
$$

$$
D_{\mathrm{relative}}=\frac{D_{\mathrm{eq}}}{\sqrt{H\times W}}
$$

该指标利用图像尺寸对等效直径进行归一化，因此没有像素单位。

This metric normalizes the equivalent diameter by the image dimensions and is therefore dimensionless.

#### `--filling_ratio_min` / `--filling_ratio_max`

填充度定义为：

The filling ratio is defined as:

$$
\mathrm{Filling\ ratio}=\frac{\mathrm{mask\ area}}{\mathrm{bounding\ box\ area}}
$$

该值通常位于 0–1 之间。数值越高，说明目标越充分填充其外接矩形。

The value is generally between 0 and 1. Higher values indicate that the object more completely fills its bounding box.

#### `--eccentricity_min` / `--eccentricity_max`

偏心率定义为：

Eccentricity is defined as:

$$
e=\sqrt{1-\frac{\lambda_{\min}}{\lambda_{\max}}}
$$

接近 0 表示目标接近等轴或圆形，接近 1 表示目标更加细长。

Values close to 0 indicate an approximately circular or equiaxed object, whereas values close to 1 indicate a more elongated object.

#### `--relative_major_axis_length_min` / `--relative_major_axis_length_max`

相对长轴长度定义为：

The relative major-axis length is defined as:

$$
L_{\mathrm{major,relative}}=\frac{L_{\mathrm{major}}}{\sqrt{H\times W}}
$$

该指标表示目标主方向长度相对于图像尺度的归一化结果。

This metric represents the object's major-axis length normalized by the image dimensions.

#### `--relative_minor_axis_length_min` / `--relative_minor_axis_length_max`

相对短轴长度定义为：

The relative minor-axis length is defined as:

$$
L_{\mathrm{minor,relative}}=\frac{L_{\mathrm{minor}}}{\sqrt{H\times W}}
$$

该指标表示目标次方向长度相对于图像尺度的归一化结果。

This metric represents the object's minor-axis length normalized by the image dimensions.

---

### 2.9 置信度异常检测 / Confidence-Based Anomaly Detection

#### `--abnormal_conf_thresh`

指定置信度异常阈值。当实例满足以下条件时，被标记为置信度异常：

Specifies the confidence threshold for anomaly detection. An instance is marked as a confidence anomaly when:

```text
confidence < abnormal_conf_thresh
```

该规则只作用于已经通过 `score_thresh` 保留的实例。例如：

This rule is applied only to instances already retained by `score_thresh`. For example:

```text
score_thresh = 0.50
abnormal_conf_thresh = 0.70

confidence < 0.50
预测结果被过滤 / The prediction is discarded

0.50 <= confidence < 0.70
实例被保留，但标记为异常 / The instance is retained but marked as abnormal

confidence >= 0.70
不触发置信度异常 / No confidence anomaly is triggered
```

如果 `score_thresh` 和 `abnormal_conf_thresh` 设置为相同数值，通常不会产生置信度异常实例。

If `score_thresh` and `abnormal_conf_thresh` are set to the same value, confidence anomalies will generally not occur.

---

### 2.10 异常结果显示 / Visualization of Abnormal Instances

#### `--abnormal_color`

指定异常实例的掩膜和边界框颜色，使用十六进制颜色代码。例如：

Specifies the mask and bounding-box color used for abnormal instances. Hexadecimal color codes are accepted. For example:

```text
"#FF0000"    红色 / Red
"#FFA500"    橙色 / Orange
"#808080"    灰色 / Gray
```

建议保留引号，避免 `#` 被 Shell 解释为注释。

Quotation marks are recommended to prevent the shell from interpreting `#` as a comment.

#### `--hide_abnormal`

控制是否在输出图像或视频中隐藏异常实例：

Controls whether abnormal instances are hidden in annotated images or videos:

```text
0    显示异常实例，并使用异常颜色标记 / Display abnormal instances
1    不在可视化结果中绘制异常实例 / Do not draw abnormal instances
```

该参数只改变可视化结果。被隐藏的实例仍保留在 CSV 文件、统计结果和实例计数中。

This option affects visualization only. Hidden instances remain in the CSV files, statistical summaries, and instance counts.

---

### 2.11 异常判定逻辑 / Anomaly Decision Logic

启用异常检测后，脚本分别检查类别异常、形态异常和置信度异常。三类条件采用“或”逻辑，只要满足其中任意一项，该实例就被标记为异常：

When anomaly detection is enabled, the script independently evaluates label, morphological, and confidence anomalies. The three criteria are combined using OR logic. An instance is marked as abnormal if any criterion is met:

$$
\mathrm{Abnormal}=\mathrm{Label\ anomaly}\lor\mathrm{Morphology\ anomaly}\lor\mathrm{Confidence\ anomaly}
$$

---

### 2.12 输出文件 / Output Files

#### 图像模式 / Image Mode

```text
output_dir/
├── pred_images/
├── instance_metrics.csv
├── image_summary_metrics.csv
└── failed_images.txt
```

- `pred_images/`：带实例分割结果的图像 / Annotated prediction images;
- `instance_metrics.csv`：每个实例的类别、置信度、形态参数和异常信息 / Instance-level classes, confidence scores, morphology, and anomaly information;
- `image_summary_metrics.csv`：每幅图像的汇总统计 / Image-level summary statistics;
- `failed_images.txt`：读取或预测失败的图像，仅在发生失败时生成 / Failed-image records, generated only when failures occur.

#### 视频模式 / Video Mode

```text
output_dir/
├── predicted_*.mp4
├── instance_metrics_*.csv
├── frame_summary_metrics_*.csv
├── video_summary_metrics_*.csv
├── failed_frames_*.csv
└── predicted_frames/
```

- `predicted_*.mp4`：带预测结果的视频 / Annotated prediction video;
- `instance_metrics_*.csv`：视频中所有预测实例的数据 / Instance-level measurements;
- `frame_summary_metrics_*.csv`：每个抽样帧的汇总结果 / Frame-level summaries;
- `video_summary_metrics_*.csv`：视频整体汇总信息 / Video-level summary;
- `failed_frames_*.csv`：失败帧记录，仅在发生失败时生成 / Failed-frame records, generated only when failures occur;
- `predicted_frames/`：单独保存的预测帧 / Individually exported annotated frames.

视频均匀抽帧后，输出视频仅包含被预测的帧。脚本会调整输出帧率，使输出视频的播放时长尽量与原视频一致。

When uniform frame sampling is used, the output video contains only the processed frames. The output frame rate is adjusted so that the playback duration remains approximately equal to that of the input video.

---

### 2.13 最大预测实例数量 / Maximum Number of Predictions

Detectron2 默认每幅图像最多保留 100 个预测实例：

By default, Detectron2 retains at most 100 predicted instances per image:

```python
cfg.TEST.DETECTIONS_PER_IMAGE = 100
```

该设置不是当前脚本的命令行参数。在高密度图像中，如果每幅图像可能包含超过 100 个实例，可在 `build_predictor()` 中提高该值，例如：

This setting is not currently exposed as a command-line parameter. For high-density images containing more than 100 objects, increase the value in `build_predictor()`, for example:

```python
cfg.TEST.DETECTIONS_PER_IMAGE = 300
```

该值必须在创建 `DefaultPredictor` 之前完成设置。

The value must be set before creating `DefaultPredictor`.

---

### 2.14 使用建议 / Practical Recommendations

1. 首次分析新数据时，可先关闭异常检测并检查实例分割结果，再根据统计分布设定形态阈值。
2. 形态阈值应来源于当前实验体系中的正常样本，不建议直接沿用其他成像条件下的阈值。
3. 相对面积、相对直径和相对轴长均为比例值，不是百分数或像素值。
4. 所有下限必须小于或等于相应上限。
5. 对高密度图像进行预测前，应确认 `DETECTIONS_PER_IMAGE` 是否足够大。
6. `hide_abnormal=1` 只隐藏可视化中的异常实例，不会将其从统计数据中删除。

1. When analyzing a new dataset, first disable anomaly detection and inspect the instance-segmentation results before defining morphology thresholds.
2. Morphological thresholds should be derived from normal samples acquired under the relevant experimental and imaging conditions.
3. Relative area, relative diameter, and relative axis lengths are fractional values rather than percentages or pixel measurements.
4. Every lower threshold must be less than or equal to its corresponding upper threshold.
5. Before processing high-density images, confirm that `DETECTIONS_PER_IMAGE` is sufficiently large.
6. `hide_abnormal=1` hides abnormal instances from visualization only and does not remove them from the statistical outputs.
