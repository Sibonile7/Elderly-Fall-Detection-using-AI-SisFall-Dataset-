Elderly Fall Detection using AI (SisFall Dataset)
📖 Overview

This project detects falls in elderly people using the SisFall dataset and a deep learning model.
When a fall is detected in real-time from wearable IMU data (e.g., MPU6050 on Arduino/ESP32), the system sends an SMS alert to caregivers or family members.

1) Uses SisFall dataset (200 Hz accelerometer & gyroscope data)
2) Trains a 1D-CNN + GRU model to classify fall vs. daily activity
3) Works with real-time IMU streaming via serial
4) SMS notifications powered by Twilio
