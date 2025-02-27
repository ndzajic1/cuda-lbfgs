#!/bin/bash
nvcc -arch=sm_75 $1 functions.cpp line_search.cpp vector_utils.cpp -o lbfgs_cuda -lcublas
