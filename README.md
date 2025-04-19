# AI-Based Facial Recognition for Security and Surveillance on ESP32-S3

![ESP32](https://img.shields.io/badge/Platform-ESP32S3-blue.svg)
![AI](https://img.shields.io/badge/AI-FaceRecognition-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is part of my Bachelor's Thesis at the German University in Cairo (in partnership with GIU Berlin). It implements a real-time, AI-based facial recognition system optimized for **security and surveillance**, using **Principal Component Analysis (PCA)** and **Eigenfaces** on an **ESP32-S3 microcontroller** with a camera module.

---

## 📚 Thesis Context

> _"This work proposes a lightweight, efficient AI model for facial recognition based on eigenfaces, targeting surveillance systems in resource-constrained environments. It aims to deploy PCA on the ESP32-S3 to enable real-time, offline recognition with support for SD card storage, mobile API integration, and LED-based feedback."_

> — *Adham Allam, GIU Berlin Bachelor Thesis 2024*

Read the full thesis [here](./GIU_Berlin_Bachelor'sThesis_AdhamAllam.pdf)

---

## 🚀 Project Highlights

- ✅ **Two-Stage Face Detection** using MSR01 and MNP01 models  
- ✅ **PCA-Based Dimensionality Reduction** (18 components from 4096-pixel input)  
- ✅ **Eigenfaces Projection** and **Class Centroid Matching**  
- ✅ **Dual-Core Optimization** using FreeRTOS Tasks on ESP32-S3  
- ✅ **On-Device Face Recognition** (No cloud inference)  
- ✅ **Secure Results Transmission** to Mobile/Web via HTTP API  
- ✅ **SD Card Logging** and JPEG storage of full/cropped/gray/resized images  
- ✅ **LED Feedback** for security access control  
- ✅ **Edge AI & IoT Ready** – real-time inference at the edge  

---

## 🧠 System Architecture

```
+-------------+       +------------------+      +------------------+
|  Camera     |  -->  | Face Detection   | -->  | Image Preprocess |
+-------------+       | MSR01 + MNP01    |      | Crop + Gray + 64x64
                      +------------------+      +------------------+
                                                        |
                                                +--------------------+
                                                | Normalize + Center |
                                                +--------------------+
                                                        |
                                                +---------------------+
                                                | PCA + Eigenfaces    |
                                                +---------------------+
                                                        |
                                                +---------------------+
                                                | Euclidean Distance  |
                                                | to Class Centroids  |
                                                +---------------------+
                                                        |
                                       +----------------------------------+
                                       | Result → LED / Server / SD Card |
                                       +----------------------------------+
```

---

## 🔧 Hardware Requirements

- 🧠 ESP32-S3 XIAO Dev Board (with PSRAM)  
- 📷 OV2640/OV5640 Camera Module (RGB565 supported)  
- 💾 Micro SD Card (for image storage)  
- 🔌 Power supply via USB-C  
- 📡 WiFi Access Point (for sending results)  

---

## 📦 Software & Libraries

- [ESP-IDF or Arduino-ESP32](https://github.com/espressif/arduino-esp32)  
- `esp_camera.h` (ESP32 camera driver)  
- `esp_http_client.h` (WiFi HTTP post)  
- `freertos` (task management, semaphores)  
- `human_face_detect_msr01.hpp` & `mnp01.hpp`  
- `Base64.h`, `img_converters.h`, etc.  
- Custom headers:  
  - `pca_transformation_matrix.h`  
  - `pca_mean_vector.h`  
  - `pca_eigenvalues.h`  
  - `class_centroids.h`  

---

## 📁 Folder Structure

```bash
├── include/
│   ├── pca_mean_vector.h
│   ├── pca_transformation_matrix.h
│   ├── class_centroids.h
│   └── pca_eigenvalues.h
├── src/
│   ├── main.cpp            # Main face recognition logic
│   └── allam_CAM.h         # Camera + image utilities
├── captured/
│   └── image0.jpg          # Stored camera frames (SD)
├── README.md
└── GIU_Berlin_Bachelor'sThesis_AdhamAllam.pdf
```

---

## 🔄 Workflow

1. Boot device and connect to Wi-Fi  
2. Capture frame from camera  
3. Detect faces using MSR01 + MNP01  
4. Crop, grayscale, resize face image to 64×64  
5. Normalize pixel values and subtract PCA mean  
6. Run PCA on 18 components using dual-core tasks  
7. Project result onto eigenfaces  
8. Calculate Euclidean distance to class centroids  
9. Transmit recognition result to HTTP server  
10. Log image and result on SD card  
11. Blink LED (once = unrecognized, twice = recognized)  

---

## 📲 Mobile / Server Integration

Set up a local or cloud API to receive POST data at:

```http
POST http://<SERVER_IP>:3500/api/face_recognition
Content-Type: application/json

{
  "class_label": 0,
  "min_distance": 0.82
}
```

---

## 📏 Performance

| Metric                    | Value          |
|--------------------------|----------------|
| Frame Size               | 240 x 240      |
| PCA Input Vector         | 4096           |
| PCA Components           | 18             |
| Processing Time (avg.)   | ~140ms         |
| On-device Classification | Yes            |
| Offline Support          | Yes (w/o WiFi) |
| Power Efficiency         | High (ESP32-S3)|

---

## 📚 Academic Contributions

- Bachelor’s Thesis submitted to GIU Berlin (April 2024)  
- Supervisor: Dr. Mohamed Karam  
- Grade: **A+ (Excellent)**  
- Final Title: *AI-Based Facial Recognition for Security & Surveillance Using Edge Microcontrollers*  
- Research Focus:  
  - Lightweight AI inference on low-power hardware  
  - Edge IoT system for real-time surveillance  
  - Optimization using parallelism (dual-core)  

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🙋 Acknowledgments

- **German University in Cairo** and **German International University Berlin** for academic guidance  
- **Dr. Mohamed Karam** for supervision and technical mentorship  
- **OpenMMLab & ESP32 Teams** for toolkits and model backbones  

---

## ✨ Future Work

- Add support for more advanced classifiers (e.g., SVM or k-NN)  
- Improve robustness under varying lighting and angles  
- Enhance GUI feedback via OLED or Bluetooth BLE  
- Extend for multiple face tracking and database training  

---

> This project showcases how AI, when integrated with IoT and Edge Computing, can revolutionize security systems through embedded intelligence.
