# HydroCare: Sip-Level Hydration Monitoring via Wearable ToF Sensor 💧

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![iOS](https://img.shields.io/badge/iOS-Swift-orange.svg)
![Hardware](https://img.shields.io/badge/Hardware-VL53L8CX%20ToF-brightgreen.svg)

## Overview
HydroCare is a privacy-preserving, wearable healthcare system designed to autonomously monitor fluid intake and eating habits. Unlike traditional camera-based systems that compromise user privacy, HydroCare utilizes a chest-mounted **8x8 Time-of-Flight (ToF) sensor (VL53L8CX)** to capture chest-to-hand/container spatial movements and gestures. 

The system uses a custom two-stage Deep Learning pipeline to first classify the user's current action and then quantitatively estimate the volume of liquid consumed. All insights are seamlessly transmitted to a companion iOS application for real-time tracking.

## Key Features
* **Privacy-Preserving Hardware:** Chest-mounted 8x8 ToF sensor that captures depth matrices rather than optical images.
* **Comprehensive Dataset:** Custom-collected dataset featuring 45 distinct human participants, encompassing various container types, drinking durations, and eating behaviors.
* **Two-Stage Machine Learning Pipeline:** 
  * **Stage 1 (Classification):** Accurately classifies user activity into three states: `Drinking`, `Eating`, or `None` (Idle).
  * **Stage 2 (Regression/Estimation):** If `Drinking` is detected, the second model calculates an estimation of the exact liquid volume consumed per sip.
* **iOS Integration:** Real-time data syncing to a custom Swift-based iOS app for user-facing health analytics.

## Machine Learning Architecture
The core intelligence of HydroCare relies on a **Temporal Convolutional Network (TCN)** pipeline designed for time-series depth data:

1. **Activity Classifier:** Analyzes the temporal sequence of the 8x8 ToF depth matrices to recognize hand-to-mouth gestures. Achieved **94% classification accuracy** across the 3-class system (Eating/Drinking/None).
2. **Volume Estimator:** Extracts spatial features (container size/angle) and temporal features (duration of sip) to predict intake volume, achieving an **$R^2$ score of 0.89**.

## Data Collection
The dataset used to train this model was rigorously collected in-house:
* **Participants:** 45 individuals.
* **Variables Captured:** Gesture duration, hand movement trajectories, container variance (mugs, bottles, cups), and distinct eating motions vs. drinking motions.

## System Architecture Flow
`Chest-Mounted ToF Sensor` ➡️ `Depth Matrix (8x8) Extraction` ➡️ `TCN Classification (Eat/Drink/Null)` ➡️ `Volume Estimation (If Drink)` ➡️ `Bluetooth/WiFi Transmission` ➡️ `iOS Application Dashboard`

## Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* Scikit-Learn, Pandas, NumPy
* iOS Development Environment (Xcode/Swift) for the mobile app

### Installation
1. Clone the repository:
   ```bash  
   git clone https://github.com/alirezamxi/hydrocare.git  
   cd hydrocare  
