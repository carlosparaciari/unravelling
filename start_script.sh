#!/bin/bash

nohup python main.py > runtime.log 2>&1 &
echo $! > process_PID.txt