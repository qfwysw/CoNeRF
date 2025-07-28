# CoNeRF Project

This project provides a pipeline for contour-guided neural radiance field (CoNeRF) training, denoising, and visualization, with a focus on 3D reconstruction of rice plant structures.

## Setup Instructions

### 1. Environment Setup

Follow the [Nerfstudio official installation guide](https://github.com/nerfstudio-project/nerfstudio) to set up the environment. Make sure all dependencies are correctly installed.

### 2. Download Datasets and Models

- **Dataset**  
  Download `rice.zip` from [Google Drive](https://drive.google.com/file/d/1n5VeyFv8ZsOgOLphgaBMxIyuiSVEu5Pr/view?usp=sharing), and extract it to the project root directory as:

  ```
  ./rice/
  ```

- **Model Checkpoints**  
  Download `outputs.zip` from [Baidu Netdisk](https://pan.baidu.com/s/19U2nz4kn3oL6Y0T3viYaPA?pwd=2bmx) (extraction code: `2bmx`) and extract it to:

  ```
  ./outputs/
  ```

- **Rendered Results (optional)**  
  Download `render1.zip` from [Google Drive](https://drive.google.com/file/d/1RIXCiBRu87XQRWSaqrQOKmWIRn5h_Dh_/view?usp=sharing), and extract it to:
  ```
  ./render1/
  ```

### 3. Install NAFNet (for Image Denoising)

Clone and install [NAFNet](https://github.com/megvii-research/NAFNet.git), which is used for image denoising:

```bash
git clone https://github.com/megvii-research/NAFNet.git
cd NAFNet
# Follow the NAFNet README to install dependencies and set up
```

## Pipeline Scripts

- **Generate Contour Masks**

  ```bash
  python pytools/mask_json.py
  ```

  This script generates binary contour masks based on the dataset metadata.

- **Train CoNeRF**

  ```bash
  bash mytools/train_demo1_reg.sh
  ```

  This shell script launches training using Nerfstudio with contour regularization.

- **Denoise Point Cloud via Projection**
  ```bash
  python pytools/projection.py
  ```
  Projects the 3D points onto denoised image space to remove outliers and enhance reconstruction quality.

## Notes

- Ensure all paths in the scripts are correctly set relative to the root directory.
- The dataset is focused on rice panicles; other plant species may require different configurations or annotations.

## Citation

If you use this code or dataset, please cite the corresponding paper (to be updated).

---

For questions or contributions, please open an issue or submit a pull request.
