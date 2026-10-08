# Multi-label Image Classification with PASCAL VOC 2012

An educational computer vision project comparing a custom convolutional neural network with EfficientNetB0 using transfer learning and fine-tuning. Each image can contain several of the 20 VOC object classes, so the models predict a vector of independent class probabilities rather than a single label.

> **Results status:** the retained notebook outputs are from the original course-project run. The data loader, learning-rate callbacks, and comparison code have since been corrected. The current notebook has not been trained again with these changes. Historical metrics below must not be presented as verified results of the current implementation or as official VOC2012 benchmark scores.

## Repository contents

| File | Description |
| --- | --- |
| `voc2012-multilabel-classification.ipynb` | Data preparation, exploratory analysis, model training, evaluation, and clearly marked historical outputs. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Excludes datasets, downloaded archives, model artifacts, local environments, and PDFs. |

The course report, presentation, dataset archives, and trained model weights are not included.

## Get the dataset

Use the official **VOC2012 training/validation archive**, not the test archive or a ZIP from another source. The [VOC2012 project page](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit) provides the training/validation download. An HTTPS download on the Oxford host is available as [VOCtrainval_11-May-2012.tar](https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).

The download is approximately **2 GB**; allow additional disk space for extraction. Follow the dataset's [database rights](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#rights), which refer to the corresponding Flickr image terms. Sample images in retained notebook outputs also come from the dataset.

Choose one of these setup methods:

1. **Manual download:** save the archive as `data/VOCtrainval_11-May-2012.tar` beside the notebook, then run the dataset cells. They verify its checksum and extract it automatically.
2. **Download from the notebook:** set `DOWNLOAD_DATASET = True` in the dataset configuration cell. The next cell downloads, checks, and extracts the archive.
3. **Existing extracted copy:** set `VOC_ROOT` in the notebook, or set the `VOC2012_DIR` environment variable, to the directory containing `Annotations`, `JPEGImages`, and `ImageSets/Main/trainval.txt`. No download is needed.

The default layout after extraction is:

```text
data/
├── VOCtrainval_11-May-2012.tar
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── JPEGImages/
        └── ImageSets/
            └── Main/
                └── trainval.txt
```

The archive checksum is MD5 `6cd6e144f989b92b3379bac3b3de84fd`, also used by [Torchvision's VOC dataset loader](https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py). A ZIP renamed to `.tar` is not a valid replacement. If a download fails, obtain the same archive through the official project page.

The current notebook indexes only image IDs in `ImageSets/Main/trainval.txt`. It creates its own seeded **70% training / 15% validation / 15% test** split from that labeled pool. The official test images exist, but their classification ground truth is not publicly released. This project's internal test split is therefore different from the official challenge test set. See the [VOC development kit documentation](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html).

## Requirements

Use **Python 3.12**. The original notebook metadata records Python 3.12.4, and the updated extraction code uses the safe TAR data filter.

A GPU is recommended for training, although the notebook can run on CPU. The code selects mixed precision when TensorFlow detects a GPU and float32 otherwise. Training the three stages can take substantial time; no fixed runtime is guaranteed. You need enough disk space for the approximately 2 GB archive and its extracted contents, plus memory for image batches and model training.

Internet access is needed to install dependencies, obtain the dataset, and download EfficientNetB0's ImageNet weights on the first run. No API keys, private accounts, or personal Google Drive paths are required by the code.

Exact original library versions were not recorded. The dependency file lists the required packages and the Seaborn minimum used by the plotting API, but does not recreate a verified historical environment. Results can vary with data selection, library versions, and hardware despite the fixed random seed.

## Run locally

### 1. Get the project files

Clone the repository, or select **Code → Download ZIP** on GitHub and extract the downloaded project archive. This project ZIP contains the notebook and documentation; it is separate from the VOC2012 dataset TAR archive.

Open a terminal in the directory containing `requirements.txt` and `voc2012-multilabel-classification.ipynb`. On macOS, you can type `cd `, drag that directory from Finder into the terminal, and press Enter. Subsequent paths such as `data/` are relative to this directory.

### 2. Create a Python environment

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, using Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the Python command is unavailable, install Python 3.12 and reopen the terminal. Keep this environment active for the remaining commands.

### 3. Open the notebook with the correct kernel

Register the environment as a named Jupyter kernel and launch JupyterLab:

```bash
python -m ipykernel install --user --name voc2012-ml --display-name "Python (VOC2012)"
python -m jupyter lab voc2012-multilabel-classification.ipynb
```

In the notebook, select **Python (VOC2012)** as its kernel. This ensures the notebook uses the packages installed in the environment rather than another Python installation.

### 4. Configure and prepare VOC2012

Follow one of the three dataset methods described above. For a first run with automatic download, locate the dataset configuration cell and change only:

```python
DOWNLOAD_DATASET = True
```

Leave `DATA_DIR`, `VOC_ROOT`, and `ARCHIVE_PATH` at their defaults unless you already have the dataset elsewhere. To use an existing extracted copy, replace the `VOC_ROOT` assignment with your own path, for example:

```python
VOC_ROOT = Path("/path/to/VOCdevkit/VOC2012")
```

On Windows, use a raw string for an absolute path, for example `Path(r"C:\datasets\VOCdevkit\VOC2012")`.

The preparation cell prints `Dataset ready:` when it finds or extracts the required directory structure. The subsequent index cell reports how many images were selected, missing images, and XML parse errors. Resolve missing or invalid files before training.

### 5. Execute the experiment

Restart the kernel and run all cells in their displayed order. Do not run the training or evaluation cells independently: they rely on functions, data splits, preprocessing, and predictions created earlier in the notebook.

The execution proceeds through:

1. Imports and precision configuration.
2. Dataset preparation and annotation indexing.
3. Exploratory analysis and multi-hot target encoding.
4. Seeded train/validation/test split and image pipelines.
5. CNN training and evaluation.
6. EfficientNetB0 training with a frozen backbone and evaluation.
7. EfficientNetB0 fine-tuning and evaluation.
8. Per-class comparisons and the final metrics table.

You can stop before the **Modello CNN** section to inspect the dataset and exploratory analysis without training. For a complete run, wait for all three training stages and the final results table to finish. Save the notebook to retain the new outputs. After closing the terminal, reactivate `.venv` before launching JupyterLab again.

## Run in Google Colab

1. Upload `voc2012-multilabel-classification.ipynb` to Colab and upload `requirements.txt` to the runtime's working directory.
2. Select a GPU runtime if available, before running the experiment.
3. Add and run a temporary setup cell containing `%pip install -r requirements.txt`. Restart the runtime if requested after package installation.
4. In the notebook's dataset configuration cell, set `DOWNLOAD_DATASET = True`. The default relative `data/` directory will be created in the runtime's working directory. Alternatively, point `VOC_ROOT` to an already extracted dataset accessible to that runtime.
5. Run the notebook from the beginning in order, then save or download the executed notebook to keep its new results.

Colab runtime files and trained models held in memory can be lost when a session ends. A fresh session may need the dataset again. The code does not mount Google Drive automatically. GPU availability and runtime limits depend on the Colab environment.

## Outputs and interpretation

Charts, training curves, class thresholds, example predictions, and metric tables are displayed inside the notebook. Training does not automatically export a reusable model file: model objects and fitted weights remain in the running kernel unless you explicitly save them.

The final table compares the CNN, frozen EfficientNetB0, and fine-tuned EfficientNetB0. mAP and F1 are reported on a 0–1 scale. Per-class thresholds are chosen on validation data and applied to test predictions. Ranking metrics evaluate the order of predicted class scores rather than a fixed probability threshold.

Existing saved outputs are historical. They are not evidence that the current code has just completed successfully. To report results for this revised version, complete a fresh run, inspect the final table, save the notebook, and update the historical-results section with the new dataset and environment details.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv`, install `requirements.txt`, and select the `Python (VOC2012)` kernel. In Colab, run the installation cell in the active runtime. |
| `VOC2012 data not found` | Set `DOWNLOAD_DATASET = True`, place the official archive at `ARCHIVE_PATH`, or point `VOC_ROOT` to the extracted `VOC2012` directory. |
| Archive checksum mismatch | Remove the incomplete or incorrect dataset TAR file and download the official training/validation archive again. Do not rename a ZIP to `.tar`. |
| Expected dataset directories not found | Check that `VOC_ROOT` directly contains `Annotations/`, `JPEGImages/`, and `ImageSets/Main/trainval.txt`. It must not point to their parent `VOCdevkit/`. |
| Download interrupted or unavailable | Check connectivity and disk space. Download through the official dataset page and use the manual method. |
| Missing XML files or images | Confirm that extraction completed and that the image list and annotations belong to the same official dataset. |
| Out of memory | Reduce `BATCH_SIZE` from 64 to 32 or 16 in the data configuration section, restart the kernel, and rerun the notebook. |
| Training is slow or no GPU is detected | CPU execution is supported but can be much slower. Check `tf.config.list_physical_devices("GPU")` in the active environment or use a GPU runtime. |
| Undefined variables or inconsistent predictions | Restart the kernel and execute all cells in order after changing the data or training settings. |

## Method

- Parse XML annotations and encode object presence as a 20-element multi-hot target.
- Explore image dimensions, object counts, class imbalance, co-occurrence, and example bounding boxes. Bounding boxes are used for exploration; the prediction task is image classification, not object detection.
- Resize images to 224 × 224 pixels and apply training-time augmentation.
- Train a custom CNN from scratch using a weighted binary cross-entropy loss calculated from training labels.
- Train an EfficientNetB0 classification head with the ImageNet backbone frozen, then fine-tune its final layers while keeping Batch Normalization layers frozen.
- Use validation performance for early stopping and per-class F1 threshold selection, then apply the chosen thresholds to the internal test set.
- Compare mean average precision (mAP), micro/macro F1, ranking metrics, and precision/recall at three predictions per image.

This is a custom educational evaluation protocol. It uses a random split rather than the official challenge splits, does not implement the challenge's special handling of difficult objects, and computes average precision with scikit-learn. Its scores should not be compared directly with official VOC leaderboards.

## Historical results

The original saved execution indexed **17,125 XML annotations from a locally packaged archive**. That archive is absent from this repository, and its equivalence to the official classification training/validation set has not been verified. The table below transcribes the notebook's original final summary; it describes that earlier experiment only.

| Model | Historical mAP | Historical micro F1 |
| --- | ---: | ---: |
| Custom CNN | 0.2968 | 0.4724 |
| EfficientNetB0, frozen backbone | 0.7696 | 0.7965 |
| EfficientNetB0, fine-tuned | 0.7788 | 0.8023 |

Before reporting new results, rerun the full notebook on the documented dataset. In particular, the corrected learning-rate callbacks now minimize `val_loss`, and the per-class comparison uses the saved predictions from the frozen-backbone stage instead of reusing the fine-tuned model. Outputs from the incorrectly labeled comparison have been removed.

## Validation and references

The publication preparation includes code syntax checks and isolated checks of dataset preparation and image-list selection. Full model training and hardware-specific TensorFlow behavior have not been verified after the changes.

Dataset reference: *The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results*, [project website](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/).

Architecture reference: *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019, [paper](https://proceedings.mlr.press/v97/tan19a.html).
