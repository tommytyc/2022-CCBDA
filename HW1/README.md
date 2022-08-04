# HW1 Music Memorability Prediction
- student id: 310551062
- name: 唐宇謙

## Intro
- [kaggle link](https://www.kaggle.com/competitions/music-regression)
- HW1 is from a kaggle competition about music memorability prediction. Given a piece of music signal(about 5 seconds), we need to predict the memorability score.
- My model ranked **sixth** out of 45 in the public leaderboard.
- Due to the data distribution being quite imbalance between public and private leaderboard, TA didn't take the private one into consideration when they were grading the HW1.

## Model architecture
- Audio data preprocessing
    - Refer to `dataset.py`
- 4 convolution layers
    Learning the temporal relationship of the mel spectrogram of audio
- 3 linear layers
    - Learning the regression score

## Code execution
- You need to have two directories `result` and `model` for code execution
    ```bash
    mkdir model
    mkdir result
    ```
- Train the model
    ```bash
    python3 main.py --train_path <train_csv> --audio_dir <audio_dir>
    ```
- Test the model
    ```bash
    python3 test.py --filename [list of audio file path to predict]
    ```
