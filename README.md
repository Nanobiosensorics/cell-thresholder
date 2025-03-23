# Cell Thresholder App

## Main Features

### 1. **Import Image**
   - **How to Use**:
     1. Drag and drop your image (JPEG, PNG) file into the application window.
     2. The image will load and be displayed on the screen for processing.

### 2. **Import Mask from NPZ File**
   - **How to Use**:
     1. Drag and drop your NPZ file containing the mask data onto the application window.
     2. The NPZ file should contain the `im_markers` key with the mask data.
     3. Optionally, you can adjust the mask offset if required in the `segment` panel.
     4. The mask will be overlaid on the image to visualize the segmentation.

### 3. **Preprocess Image with Rectangular or Circular Masks**
   - **How to Use**:
     - **Rectangular (Label) Mask**:
       1. Right-click anywhere on the image.
       2. Select **"Label cut"** from the context menu.
       3. Draw a rectangular area on the image by selecting two points to remove unwanted regions by excluding them from the analysis.
     - **Circular Mask**:
       1. Right-click anywhere on the image.
       2. Select **"Circle mask"** from the context menu.
       3. Draw a circular mask by selecting three points to eliminate regions outside the circle for cleaner analysis.

### 4. **Segment the Image**
   - **How to Use**:
     1. Right-click anywhere on the image.
     2. Select **"Segment"** from the context menu.
     3. Set a **Gaussian Blur kernel size** to apply a Gaussian filter to the image before thresholding, improving segmentation results by reducing noise
     4. The **Triangle Method** will automatically segment the image by analyzing pixel intensity distributions.
     5. The resulting segmentation will be applied to the image.

### 5. **Threshold the Image**
   - **How to Use**:
     1. Select a threshold value on the lower **"Threshold"** panel.
     2. Click **"Apply"** to execute the thresholding.
     3. Cells with an integral intensity lower than the selected threshold value will be excluded from the mask.

### 6. **Export Processed Data to CSV**
   - **How to Use**:
     1. After completing the thresholding and masking steps, right-click on the image.
     2. Select **"Export"** from the context menu.
     3. Choose the location and filename for the CSV file.
     4. The app will export the following information:
        - **Total Cell Count**
        - **Cell ID**
        - **Cell Area**
        - **Integral Intensity**
     5. The CSV file will be saved with this data for further analysis.

## Installation

1. Clone or download the repository to your local machine.
2. Install dependencies via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

Alternatively, you can download the precompiled executable for windows from [here](https://drive.google.com/file/d/1mQ9FRDyqS9TUO4o6hqwR0C4Ky5FDN37o/view?usp=drive_link) and run it directly without needing to install Python or dependencies.

## Requirements

- Python 3.6 or higher
- PyQt5
- OpenCV
- NumPy
- Scikit-image
