import cv2
import numpy as np
from ultralytics import YOLO
import os

# Input video path
input_path = "videos/1070679382-preview.mp4"
assert os.path.exists(input_path), "Video file not found"

# Extract video name
video_name = os.path.splitext(os.path.basename(input_path))[0]
output_video_path = f"output/{video_name}_result.avi"
output_report_path = f"output/{video_name}_report.txt"

# Open video
cap = cv2.VideoCapture(input_path)
assert cap.isOpened(), "Error opening video"

# Video properties
w, h, fps = (int(cap.get(x)) for x in (
    cv2.CAP_PROP_FRAME_WIDTH, 
    cv2.CAP_PROP_FRAME_HEIGHT, 
    cv2.CAP_PROP_FPS
))


line_in = [(375, 100), (535, 100)]
line_out = [(70, 100), (230, 100)] 


# Create output folder if it does not exist
os.makedirs("output", exist_ok=True)

# Video writer
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# YOLO model with tracking
model = YOLO("runs/detect/train2/weights/best.pt")

# Track history dictionary (track_id -> (cx, cy))
track_history = {}

# Counters
count_in = 0
count_out = 0

# Sets to avoid double counting
counted_in_ids = set()
counted_out_ids = set()

frame_index = 0
in_times = []
out_times = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.25, iou=0.5)
    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Draw bounding box around the object (cyan border)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if track_id in track_history:
                prev_cx, prev_cy = track_history[track_id]

                # IN: crossing IN line from below to above
                if prev_cy > line_in[0][1] >= cy:
                    if track_id not in counted_in_ids and (line_in[0][0] <= cx <= line_in[1][0]):
                        count_in += 1
                        counted_in_ids.add(track_id)
                        in_times.append(frame_index / fps)

                # OUT: crossing OUT line from above to below
                if prev_cy < line_out[0][1] <= cy:
                    if track_id not in counted_out_ids and (line_out[0][0] <= cx <= line_out[1][0]):
                        count_out += 1
                        counted_out_ids.add(track_id)
                        out_times.append(frame_index / fps)

            track_history[track_id] = (cx, cy)

    # Draw lines
    cv2.line(frame, line_in[0], line_in[1], (0, 255, 0), 2)   # green IN
    cv2.line(frame, line_out[0], line_out[1], (0, 0, 255), 2) # red OUT 

    # Save frame
    video_writer.write(frame)

    # Show frame
    cv2.imshow("Object Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Function to calculate average time between events
def calc_avg_time(times):
    if len(times) < 2:
        return 0
    diffs = np.diff(times)
    return np.mean(diffs)

# Calculate stats
avg_time_in = calc_avg_time(in_times)
avg_time_out = calc_avg_time(out_times)
percent_claimed = ((count_in - count_out) / count_in) * 100 if count_in > 0 else 0

# Prepare report
report_text = (
    f"--- REPORT for {video_name} ---\n"
    f"Total objects IN: {count_in}\n"
    f"Total objects OUT: {count_out}\n"
    f"Average time between entries (IN): {avg_time_in:.2f} s\n"
    f"Average time between exits (OUT): {avg_time_out:.2f} s\n"
    f"Percentage of claimed bags: {percent_claimed:.2f} %\n"
)

# Save report
with open(output_report_path, "w") as f:
    f.write(report_text)

print("Processing completed.")
print(f"Report saved at {output_report_path}")
print(f"Final video saved at {output_video_path}")
