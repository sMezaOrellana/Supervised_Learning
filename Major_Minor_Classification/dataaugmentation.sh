#!/bin/bash

ls -l $1 | awk -F" " '{print $9}' | tail -n +2 | ./pitchshift.py