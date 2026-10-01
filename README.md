# Real-time Object Detection and Tracking

YOLOv8 detection + SORT tracking (Kalman filter + Hungarian matching) on webcam or video.

## Setup
    pip install -r requirements.txt

## Run
    python detect_track.py                      # webcam
    python detect_track.py --source video.mp4   # video file
    python detect_track.py --save out.mp4       # record output

Options: --model, --conf, --line (tripwire height 0-1), --save

## Keys
Q quit | H heatmap | T trails | SPACE pause
