import cv2

cap = cv2.VideoCapture("volley_2_video.avi")
if not cap.isOpened():
    raise RuntimeError("OpenCV could not open the input AVI file.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback if FPS not detected

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("v2.mp4", fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    writer.write(frame)
    frame_count += 1

cap.release()
writer.release()
print(f"✅ Wrote {frame_count} frames to converted.mp4")