# Smart Vehicle - Traffic Sign Recognition System

A smart vehicle control system that uses computer vision and machine learning to recognize traffic signs and autonomously control a vehicle based on the detected signs.

## 🚗 Project Overview

This project implements an intelligent vehicle control system that:
- Captures real-time video from an IP camera mounted on the vehicle
- Detects and recognizes traffic signs using HOG (Histogram of Oriented Gradients) features and SVM (Support Vector Machine) classifier
- Sends control commands to the vehicle based on recognized traffic signs via socket communication
- Supports multiple traffic sign types including speed limits, turn directions, and stop signs

## 🎯 Features

- **Real-time Traffic Sign Recognition**: Detects blue and red traffic signs using HSV color space filtering
- **Machine Learning Classification**: Uses pre-trained SVM model with HOG features for accurate sign classification
- **Autonomous Vehicle Control**: Automatically adjusts vehicle speed and direction based on recognized signs
- **Multi-threading Support**: Implements threaded camera capture for smooth video processing
- **Socket Communication**: Controls vehicle hardware through TCP/IP socket commands

## 📋 Supported Traffic Signs

The system can recognize the following traffic signs:

### Speed Limits
- Speed Limit 15 km/h
- Speed Limit 30 km/h
- Speed Limit 60 km/h
- Speed Limit 80 km/h

### Directional Signs
- Turn Left
- Turn Right
- No Straight / Stop

## 🛠️ Technologies Used

- **Python 3.x**
- **OpenCV** - Computer vision and image processing
- **scikit-learn** - Machine learning (SVM classifier)
- **scikit-image** - HOG feature extraction
- **NumPy** - Numerical computations
- **Threading** - Concurrent video capture and processing

## 📦 Installation

### Prerequisites

- Python 3.6 or higher
- Smart vehicle hardware with camera and network connectivity
- Pre-trained SVM model file (`svm_model.pkl`)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jdalwaysbrave/Smart_vehicle.git
cd Smart_vehicle
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the SVM model file:
   - Place `svm_model.pkl` in the project root directory
   - This model should be trained on traffic sign images with HOG features

## 🚀 Usage

### Running the Main Program

The project includes three main Python files:

#### 1. final_edition.py (Recommended)
The most recent and stable version with improved threading and vehicle control.

```bash
python final_edition.py
```

#### 2. Recognition_rect.py
Enhanced version with threaded camera capture for better performance.

```bash
python Recognition_rect.py
```

#### 3. CODE3.py
Earlier version with basic functionality.

```bash
python CODE3.py
```

### Configuration

Before running, ensure the following:

1. **Network Configuration**:
   - Vehicle IP address: `192.168.1.1`
   - Socket port: `2001`
   - Video stream URL: `http://192.168.1.1:8080/?action=stream`

2. **Model File**:
   - Ensure `svm_model.pkl` exists in the project directory

3. **Camera Access**:
   - Verify the IP camera stream is accessible
   - Test the stream URL in a browser first

### Vehicle Control Commands

The system sends hexadecimal commands via socket:

- **Speed Control**: `\xff\x02\x01\xXX\xff` (right track) and `\xff\x02\x02\xXX\xff` (left track)
- **Direction Control**: `\xff\x00\xXX\x00\xff`
  - `0x01`: Forward
  - `0x03`: Turn right
  - `0x04`: Turn left
  - `0x00`: Stop

## 🔧 How It Works

1. **Video Capture**: Captures real-time video from the IP camera stream
2. **Preprocessing**: Converts RGB to HSV color space and applies color thresholding to isolate blue/red signs
3. **Contour Detection**: Finds contours in the binary image and extracts bounding rectangles
4. **Feature Extraction**: Computes HOG features from the detected sign region
5. **Classification**: Uses SVM classifier to identify the sign type
6. **Vehicle Control**: Sends appropriate control commands to the vehicle based on the recognized sign

### Image Processing Pipeline

```
Camera Stream → HSV Conversion → Color Filtering → Contour Detection 
→ ROI Extraction → HOG Features → SVM Classification → Vehicle Control
```

## 📂 Project Structure

```
Smart_vehicle/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── final_edition.py         # Main program (latest version)
├── Recognition_rect.py      # Version with threaded camera
├── CODE3.py                 # Basic version
└── svm_model.pkl           # Pre-trained SVM model (not included)
```

## 🔬 Technical Details

### Color Detection Thresholds

**Blue Signs** (HSV):
- Hue: 110-124
- Saturation: 43-255
- Value: 46-255

**Red Signs** (HSV):
- Hue: 165-180
- Saturation: 43-255
- Value: 46-255

### HOG Parameters
- Orientations: 9 bins
- Pixels per cell: 8x8
- Cells per block: 2x2
- Block normalization: L2
- Feature transform: Square root

## ⚠️ Important Notes

1. **Network Connection**: Ensure stable network connection to the vehicle
2. **Model Dependency**: The system requires a pre-trained `svm_model.pkl` file
3. **Real-time Processing**: Performance depends on camera frame rate and processing power
4. **Safety**: Test in a controlled environment before deploying
5. **Deprecated Warning**: The code uses `sklearn.externals.joblib` which is deprecated. Consider updating to `import joblib` directly

## 🐛 Troubleshooting

### Common Issues

**Connection refused error**:
- Check if the vehicle is powered on and connected to the network
- Verify the IP address (192.168.1.1) is correct
- Ensure port 2001 and 8080 are accessible

**Model loading error**:
- Verify `svm_model.pkl` exists in the correct location
- Check if the model file is compatible with your scikit-learn version

**Camera stream issues**:
- Test the stream URL in a browser: `http://192.168.1.1:8080/?action=stream`
- Check camera power and network connection

**No signs detected**:
- Verify lighting conditions
- Adjust HSV thresholds if needed
- Check if signs are in proper color range (blue/red)

## 🔄 Future Improvements

- [ ] Update to use `joblib` directly instead of `sklearn.externals.joblib`
- [ ] Add configuration file for easy parameter adjustment
- [ ] Implement logging system for debugging
- [ ] Add model training scripts
- [ ] Support for additional traffic sign types
- [ ] Add unit tests for core functions
- [ ] Implement error handling and recovery mechanisms
- [ ] Add GUI for monitoring and control

## 📝 License

This project is open source. Please add an appropriate license file if you plan to distribute it.

## 👥 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This project is designed for educational and research purposes. Ensure proper safety measures when testing with actual vehicle hardware.
