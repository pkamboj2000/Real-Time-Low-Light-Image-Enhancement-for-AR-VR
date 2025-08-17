# Dataset Information

This directory contains the datasets used for training and evaluation.

## Directory Structure

- `raw/`: Original datasets in their native format
- `processed/`: Preprocessed and augmented data ready for training
- `samples/`: Sample images for quick testing and demos

## Supported Datasets

### 1. LOL Dataset (Low-Light Dataset)
- **Description**: Paired low-light and normal-light images
- **Download**: [LOL Dataset](https://daooshee.github.io/BMVC2018website/)
- **Structure**: 
  ```
  raw/LOL/
  ├── train/
  │   ├── low/     # Low-light images
  │   └── high/    # Normal-light reference images
  └── test/
      ├── low/
      └── high/
  ```

### 2. SID Dataset (See-in-the-Dark)
- **Description**: Raw sensor data for extreme low-light conditions
- **Download**: [SID Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)
- **Structure**:
  ```
  raw/SID/
  ├── Sony/
  │   ├── short/   # Short exposure (low-light)
  │   └── long/    # Long exposure (reference)
  └── Fuji/
      ├── short/
      └── long/
  ```

### 3. AEA Dataset (Aria Everyday Activities) - Optional
- **Description**: Meta's egocentric video dataset for AR/VR research
- **Download**: [AEA Dataset](https://www.projectaria.com/datasets/aea/)
- **Note**: Requires application and approval from Meta
- **Structure**:
  ```
  raw/AEA/
  ├── recordings/
  └── annotations/
  ```

### 4. Custom Webcam Data
- **Description**: Custom captured low-light images/videos
- **Structure**:
  ```
  raw/custom/
  ├── images/
  └── videos/
  ```

## Data Preprocessing

Run the preprocessing scripts to prepare data for training:

```bash
# Process LOL dataset
python src/data/preprocess_lol.py --input_dir data/raw/LOL --output_dir data/processed/LOL

# Process SID dataset  
python src/data/preprocess_sid.py --input_dir data/raw/SID --output_dir data/processed/SID

# Generate sample data for testing
python src/data/generate_samples.py --output_dir data/samples
```

## Dataset Statistics

After preprocessing, statistics will be generated:
- Image count and resolution distribution
- Lighting condition analysis
- Data split information

## Usage Notes

1. **Storage Requirements**: 
   - LOL: ~2GB
   - SID: ~25GB
   - AEA: Variable (depends on selected recordings)

2. **Preprocessing Time**: 
   - LOL: ~5 minutes
   - SID: ~30 minutes
   - AEA: Variable

3. **Format Compatibility**: All datasets are converted to a unified format for training compatibility.
