# Safety Helmet Detection using YOLO

A computer vision system that detects whether people are wearing safety helmets, built with Ultralytics YOLO.

**Detected classes**

| ID | Class | Meaning |
|----|-------|---------|
| 0 | `helmet` | A head wearing a safety helmet |
| 1 | `no_helmet` | A bare head (no helmet) |
| 2 | `person` | A whole person, with or without a helmet |

The goal is that every person in an image is detected as `person`, and their head is classified as `helmet` or `no_helmet`.

> **Status:** dataset preparation is complete. Model training and evaluation are in progress.

---

## 1. Dataset

- **Source:** [Hard Hat Detection (Kaggle, by andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- **Size:** 5,000 images (mostly about 416 px), all with matching annotation files
- **Original format:** Pascal VOC (XML)
- **Original classes:** `helmet`, `head`, `person`

### Class mapping

| Original label | Project class | ID |
|----------------|---------------|----|
| `helmet` | `helmet` | 0 |
| `head` | `no_helmet` | 1 |
| `person` | `person` | 2 |

### The `person` problem

In the original dataset, `person` is sparsely annotated: only **751** boxes, against **24,751** `helmet` + `head` boxes. Most workers have no `person` box, so training on it as-is would teach the model that unlabeled people are background.

**Solution:** I auto-labeled the missing `person` boxes with a COCO-pretrained YOLOv8m model (COCO class 0 = person). Any detection overlapping an existing `person` box (IoU > 0.5) was skipped to avoid duplicates. The original `helmet` and `no_helmet` labels were not touched.

---

## 2. Data preparation pipeline

```
Kaggle zip (images + VOC XML)
        │
        |   utils/prepare_dataset.py
        ▼  
YOLO labels + 70/20/10 split + data.yaml
        │
        |   utils/add_person_labels.py
        ▼  
Auto-labeled `person` boxes
        │
        |   utils/stats_final_check.py  +  utils/check_boxes.py
        ▼  
Statistics + visual verification
```

### Step 1: VOC to YOLO conversion and split

```
5000 pairs found, 0 XML files without an image
Total boxes per class: {'helmet': 18966, 'no_helmet': 5785, 'person': 751}
```

| Split | Images | helmet | no_helmet | person (original) |
|-------|-------:|-------:|----------:|------------------:|
| train | 3,500 | 13,422 | 4,055 | 475 |
| val | 1,000 | 3,712 | 1,150 | 217 |
| test | 500 | 1,832 | 580 | 59 |
| **Total** | **5,000** | **18,966** | **5,785** | **751** |

Split: 70% / 20% / 10%, random with seed 42.

### Step 2: Auto-labeling `person`

| Split | person before | person after |
|-------|--------------:|-------------:|
| train | 475 | 13,403 |
| val | 217 | 3,877 |
| test | 59 | 1,853 |
| **Total** | **751** | **19,133** |

**18,382 person boxes were added.**

### Final class distribution

| Split | Images | helmet | no_helmet | person |
|-------|-------:|-------:|----------:|-------:|
| train | 3,500 | 13,422 | 4,055 | 13,403 |
| val | 1,000 | 3,712 | 1,150 | 3,877 |
| test | 500 | 1,832 | 580 | 1,853 |
| **Total** | **5,000** | **18,966** | **5,785** | **19,133** |

### Known limitations

- **Class imbalance:** `helmet` is about 3.3x more frequent than `no_helmet`. Per-class precision and recall matter more than overall mAP, especially for `no_helmet`.
- **Auto-generated `person` labels:** they are good but not perfect, and were not fully reviewed by hand. This includes the test split.
- **65 images** have no `person` box at all.
- **2,546 images** contain a helmet/no_helmet box whose center is not inside any `person` box, which usually means the auto-labeler missed a small, blurry, or occluded person. The list is written to `person_review.txt`.
- **Possible near-duplicate images** across splits, which could inflate the metrics slightly.

---

## 3. Project structure

```
Safety-helmet-detection/
├── dataset/                   # generated, not tracked by Git
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── labels/{train,val,test}_backup/   # labels before person auto-labeling
│   └── data.yaml
├── utils/                     # helper scripts (not tracked by Git)
│   └── archive/
│       ├── images/            # downloaded from Kaggle
│       └── annotations/       # VOC XML files
│   └── prepare_dataset.py       # 
│   └── add_person_labels.py       # 
│   └── stats_final_check.py       # 
│   └── check_boxes.py       # 
├── reports/                     # label verification images
│   └── sample_pre_annotation/         # a few images before auto-labeling person
│   └── sample_post_annotation/        # a few images after auto-labeling person
│       ├── person_review.txt          # images needing manual review
└── README.md

```

`dataset/`, `utils/`, weights (`*.pt`), and `runs/` are excluded from Git because of their size.

---

`dataset/data.yaml`:

```yaml
path: <absolute path to dataset>
train: images/train
val: images/val
test: images/test
names:
  0: helmet
  1: no_helmet
  2: person
```

---

## 5. Training (coming next)

- Baseline: YOLOv8s / YOLO11s, trained on GPU (Google Colab)
- Evaluation: per-class precision, recall, mAP@50 and mAP@50-95, plus the confusion matrix
- Real-world test on frames extracted from videos

---


## Credits

- Dataset: [Hard Hat Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection) by andrewmvd on Kaggle (check the dataset page for its license terms).
- Models and training framework: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

Author: Youssef Ahmed