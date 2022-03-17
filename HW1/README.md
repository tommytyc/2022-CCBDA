# HW1 Music Memorability Prediction
- student id: 310551062
- name: 唐宇謙

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