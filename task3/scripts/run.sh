#!/bin/bash

source activate nlp

# python -u run.py --learning_rate=1e-5 --batch_size=64 --early_stop=True >> log_1e-5.txt 2>&1
python -u run.py --learning_rate=3e-5 --batch_size=64 --early_stop=True >> log_3e-5.txt 2>&1
python -u run.py --learning_rate=7e-5 --batch_size=64 --early_stop=True >> log_4e-5.txt 2>&1
python -u run.py --learning_rate=1e-4 --batch_size=64 --early_stop=True >> log_1e-4.txt 2>&1
python -u run.py --learning_rate=3e-4 --batch_size=64 --early_stop=True >> log_3e-4.txt 2>&1
python -u run.py --learning_rate=7e-4 --batch_size=64 --early_stop=True >> log_7e-4.txt 2>&1
