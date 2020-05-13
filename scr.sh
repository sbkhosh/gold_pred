#!/bin/bash

file_exec="exec.py"

ipython3 ./$file_exec
rm -Rf __pycache__
rm -Rf *_info
