#!/bin/bash

GIT_REPO_URL=https://github.com/josiahcoad/ActionRecognition
REPO=ActionRecognition/demo
VIDEO=videos/pos1.mp4
UINJSON=524006234.json
UINJPG=524006234.jpg
RESULTSPATH=./results/

git clone $GIT_REPO_URL
cd $REPO
echo $VIDEO
echo "This has been tested with Python3.7"
echo "Please make sure you have 'pip install -r requirments.txt'"
python demo.py --vidpath $VIDEO

#rename the generated timeLabel.json and figure with your UIN.
cp $VIDEO.json $UIN_JSON
cp $VIDEO.jpg $UIN_JPG
