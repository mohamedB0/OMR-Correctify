# OMR-Correctify
# Image Processing Project

## Project Structure

### Directories

#### `answer_sheets/`
Contains reference image files:
- `Boxes.png`
- `bubbles.png`

#### `output/`
Stores intermediate and processed image files:
- `blur.png`: Blurred image version
- `cropped_mcq.png`: Cropped multiple-choice question image
- `grayscale.png`: Grayscale converted image
- `original.png`: Original input image
- `preprocessed.png`: Preprocessed image
- `threshold.png`: Thresholded image

#### `output_bubbles/`
Detailed image processing steps for bubble-related processing:
- `01_original.png`: Original bubble image
- `02_gray.png`: Grayscale conversion
- `03_blurred.png`: Blurred bubble image
- `04_edges.png`: Edge detection result
- `05_all_contours.png`: All detected contours
- `06_largest_contour.png`: Largest contour identified
- `07_warped.png`: Warped image transformation
- `marked_bubbles.png`: Bubbles marked or processed

### Files
- `bubbles.py`: Main Python script for image processing
- `requirements.txt`: Project dependencies

## Processing Workflow
The project appears to implement an image processing pipeline, likely for tasks such as:
- Image preprocessing
- Bubble or form detection
- Contour analysis
- Image transformation and marking
