# CPDNet: Controlled Point Deformation Network for Patient-Specific Postoperative Spine Shape Prediction

## Introduction
Adolescent Idiopathic Scoliosis (AIS) is a complex 3D spinal deformity that often requires surgical correction. Accurate prediction of the postoperative spine shape is valuable for both surgeons and patients during surgical planning. Existing biomechanical and machine-learning approaches either require detailed mechanical modeling or provide only coarse, non-personalized predictions. Predicting a patient-specific 3D postoperative spine shape remains a challenging problem.

This project introduces **CPDNet (Controlled Point Deformation Network)**, a novel deep learning architecture that frames the problem as a 3D point-set deformation task. Each spine is represented by anatomical landmarks, and CPDNet explicitly learns global spine shape properties through inter-vertebral rigid transformations, rather than treating points independently.

## Problem Definition
- **Goal:** Predict a patient’s postoperative 3D spine shape given the preoperative 3D configuration and clinical metadata.
- **Input:** Preoperative spine landmarks (ordered 3D point set)
- **Output:** Postoperative spine landmarks (ordered 3D point set)
- **Key innovation:** CPDNet enforces anatomical consistency by learning rigid transformations (rotation + translation) between consecutive vertebrae, preserving both local and global spine structure.

## Dataset
- **Source:** 99 AIS patients (Lenke 1A) who underwent posterior spinal fusion surgery (2009–2018, Sainte-Justine University Hospital, Montreal, Canada)
- **Imaging:** Pre- and postoperative biplanar X-ray scans
- **3D Representation:**
  - 18 vertebral levels (C7 to L5)
  - 6 anatomical landmarks per vertebra (108 3D points per spine)
- **Clinical metadata:** Gender, Age, Height, Weight, BMI, Operating surgeon
- **Split:** 80% training, 10% validation, 10% testing

## Preprocessing Steps
- **3D Reconstruction:** Landmarks reconstructed from biplanar X-rays
- **Landmark Extraction:** Six landmarks per vertebra
- **Global Registration:** All spines registered to a common coordinate system anchored at L5
- **Inter-vertebral Transform Computation:** Rigid transformations (rotation + translation) between vertebrae computed for supervision

## Model Architecture: CPDNet
- **Inputs per point:**
  - 3D coordinates (x, y, z)
  - Patient metadata (broadcast to all points)
  - Gaussian noise (for flexibility)
- **Feature Extraction:**
  - Fully Convolutional Neural Networks (F-CNNs)
  - Three 1D convolution layers: 128 → 64 → 3
- **Output:** 3D displacement vector per point
- **Key design choices:**
  - Same displacement for all landmarks of a vertebra (preserves topology)
  - Noise enables local flexibility
  - Metadata enables personalized predictions
- **Novelty:** Rigid Transformation Loss enforces learning of anatomically plausible translations and rotations between vertebrae

## Evaluation Metrics
- **Coronal Balance (°):** Global spine alignment (frontal plane)
- **Cobb Angle – Frontal (PA, °):** Scoliosis severity (coronal)
- **Cobb Angle – Sagittal (LAT, °):** Curvature (sagittal)
- **Reported as:** RMSE between predicted and ground-truth clinical measurements

## Results
- **Compared methods:**
  - P2P-NET (state-of-the-art point displacement network)
  - PDNet (CPDNet without rigid transformation loss)
- **CPDNet achieves the lowest error across all metrics:**
  - Coronal Balance: 1.5°
  - Cobb LAT: 7.6°
  - Cobb PA: 4.9°
- **Ablation findings:**
  - Noise improves local flexibility
  - Metadata improves personalization
  - Rigid transformation loss provides the largest performance gain
- **Qualitative:**
  - Preserves spinal topology
  - Models thoracic/thoracolumbar corrections
  - Produces anatomically plausible shapes even with limited training data

## Conclusion
CPDNet is a supervised deep learning framework for patient-specific prediction of postoperative 3D spine shape in scoliosis surgery. By integrating point-based deformation, clinical metadata, and a novel rigid transformation loss, CPDNet learns both local deformations and global spinal structure, outperforming existing methods and demonstrating strong clinical relevance.

Clinically, CPDNet can serve as the foundation for a surgical planning and visualization tool, helping surgeons and patients better understand expected postoperative outcomes for different surgical strategies.

---

## Installation & Requirements

### Required Libraries
- Python 3.6+
- TensorFlow 1.x (tested with 1.15)
- NumPy
- h5py
- OpenCV (cv2)
- autograd

### (Optional, for visualization)
- matplotlib

### Installation
Install the required libraries using pip and the provided requirements file:

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 1.x is required for this codebase. If you have TensorFlow 2.x installed, consider using a virtual environment.

## Usage
1. Prepare your dataset in HDF5 format as described above.
2. Edit the configuration and paths in `run.py` as needed.
3. Run the training script:
   ```bash
   python run.py --mode train
   ```
4. For evaluation/testing:
   ```bash
   python run.py --mode test
   ```

## Citation
If you use this code, please cite the original manuscript.
Please find the details of the project through the following papers:

Maryam Khani, Philippe Debanné, Hubert Labelle, Stefan Parent, Farida Cheriet, ”Prediction of Postoperative 3D Spine Shape using Controlled Point Deformation Network”, conference of SPIE Medial Imaging, April 2023. Link:https://doi.org/10.1117/12.2654224.
