# line-and-circle-detection
This project shows how to detect lines and circles in binary images using morphological operations in OpenCV. It uses custom-designed kernels for shape detection,applies erosion,dilation,opening,and closing techniques,visualizes results using Matplotlib. it includes a method for counting detected components using connected component analysis.

# Line and Circle Detection with OpenCV

This project focuses on detecting geometric shapes—specifically lines and circles—in binary images using morphological operations and custom kernels in OpenCV. It includes visualization, shape enhancement, and component counting techniques.

## 📂 Contents

- Circle Detection using Morphological Opening
- Line Detection using 12 Custom Kernels
- Visualization of Binary, Eroded, Dilated, and Opened Images
- Component Counting with Connected Components
- Kernel Design and Comparison
- Result Saving and Analysis

## 🛠 Technologies Used

- Python
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## 🔍 Key Features

- Converts grayscale images to binary using Otsu's thresholding
- Detects circles using a 15×15 pixel circular kernel
- Detects lines using 12 directional line kernels
- Applies morphological operations: erosion, dilation, opening, and closing
- Combines results using logical OR for comprehensive line detection
- Counts connected components to estimate number of shapes
- Visualizes each step using Matplotlib

## 📊 Methodology

### Circle Detection
- Binary conversion of input image and kernel
- Morphological opening to isolate circular shapes
- Comparison with erosion + dilation for validation

### Line Detection
- 12 directional kernels applied via opening + closing
- Logical OR used to combine all kernel results
- Dilation applied selectively to improve component separation

### Component Counting
- Uses `cv.connectedComponents()` to count shapes
- Adjusts for background and artifacts
- Handles overlapping lines by counting per kernel

## 👨‍🏫 Supervised By

Professor Khosravi  
University of Birjand

## 👤 Author

Armita Ashabyamin

## 💾 Output Files

- `line.bmp`: Final image with detected lines
- `circle.bmp`: Final image with detected circles

## 📎 How to Run

1. Place input image and kernels in the specified directory.
2. Run the Python script to process and visualize results.
3. Output images and component counts will be saved and printed.

## 📌 Notes

This project is ideal for understanding morphological shape detection and component analysis in image processing. It can be extended to detect other geometric patterns or used in document analysis, medical imaging, and industrial inspection.

