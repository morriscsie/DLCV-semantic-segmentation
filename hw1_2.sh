#!/bin/bash

# TODO - run your inference Python3 code
python3.8 test_P2B.py --test_dir "${1}" --pred_dir "${2}" --ckpt_path "./model/best_model_P2B.pt"