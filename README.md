# FL-PHT

To test, you can put the example data under the path "./example_data/MICCAI_FeTS2022_TrainingData" and including a CSV file "partitioning_1.csv" and DICOM files under the folder with subject id, for example, "FeTS2022_00000/FeTS2022_00000_t1.nii.gz", "FeTS2022_00000/FeTS2022_00000_t1ce.nii.gz", "FeTS2022_00000/FeTS2022_00000_t2.nii.gz", "FeTS2022_00000/FeTS2022_00000_flair.nii.gz" and "FeTS2022_00000/FeTS2022_00000_seg.nii.gz".

```
├── isolated
    ├── requirements.txt
    ├── README.md
    ├── main.py
    ├── imageio.py
    ├── losses.py
    ├── ...
├── example_data
    └── MICCAI_FeTS2022_TrainingData
        ├── partitioning_1.csv
        ├── FeTS2022_00000
            ├── FeTS2022_00000_t1.nii.gz
            ├── FeTS2022_00000_t1ce.nii.gz
            ├── FeTS2022_00000_t2.nii.gz
            ├── FeTS2022_00000_flair.nii.gz
            ├── FeTS2022_00000_seg.nii.gz
        ├── FeTS2022_00001
            ├── FeTS2022_00001_t1.nii.gz
            ├── FeTS2022_00001_t1ce.nii.gz
            ├── FeTS2022_00001_t2.nii.gz
            ├── FeTS2022_00001_flair.nii.gz
            ├── FeTS2022_00001_seg.nii.gz
        ├── ...
├── federated
    ├── README.md
    ├── ...
└── pht-fl
    ├── learning_image
        ├── ...
    ├── learning_image
        ├── ...
    └── README.md
```
