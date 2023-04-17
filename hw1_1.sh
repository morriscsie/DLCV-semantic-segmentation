#!/bin/bash

# TODO - run your inference Python3 code
python3.8 test_P1B.py --test_dir "${1}" --pred_file "${2}" --ckpt_path "./model/best_model_P1B.pt"
