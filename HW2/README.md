# HW2 Self-supervised Learning
- student id: 310551062
- name: 唐宇謙

## Environment creation
- Install the following package:
    - torch
    - torchvision
    - numpy
    - tqdm
- You can use `python3 venv` to create a virtual environment.

## Code execution
- Supposed that you put the data directory at `data/`, and the subdirectory of `data/` is `unlabeled/` and `test/`.
```bash
python3 byol.py --train_imgdir data/unlabeled/ --test_imgdir data/test/
```
