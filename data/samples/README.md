# Sample Data

This directory contains a small subset of the Duke Breast Cancer MRI dataset, converted to 16-bit PNG format, for testing and demonstration purposes.

## Structure

```
samples/duke/
├── Breast_MRI_001/          # Patient 1 (5 phases, 5 slices each)
│   ├── phase_0/             # Pre-contrast
│   ├── phase_1/             # Post-contrast 1
│   ├── phase_2/             # Post-contrast 2
│   ├── phase_3/             # Post-contrast 3
│   └── phase_4/             # Post-contrast 4
└── Breast_MRI_002/          # Patient 2 (4 phases, 5 slices each)
    ├── phase_0/
    ├── phase_1/
    ├── phase_2/
    └── phase_3/
```

## Full Datasets

For the complete datasets used in this study, please refer to:

- **I-SPY 2 Trial** (n=199): [TCIA](https://wiki.cancerimagingarchive.net/display/Public/ISPY2)
- **Duke Breast Cancer MRI** (n=922): [TCIA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903)
