# HW3 Anomaly Detection
- student id: 310551062
- name: 唐宇謙

## Intro
- [kaggle link](https://www.kaggle.com/competitions/ccbda2022spring-hw3/)
- HW2 is from a kaggle competition about anomaly detection. For the time-series data of a sensor, we need to predict which part of the data is normal and which part is anomaly.
- My model ranked **second** out of 43 in private leaderboard, **third** in public leaderboard.

## Model architecture
- LSTM Fully Connected (FC) Autoencoder
- Encoder uses LSTM and a FC to encode the sensor data
- Decoder is quite the same as the encoder to reconstruct the data
- Activation function uses leaky relu to avoid too many info losses
- For anomaly detection, I add a parameter LAMBDA to control the scalar of the public anomaly part, for the anomaly part should have larger reconstruction loss

## Code execution
```bash
python3 main.py
```
