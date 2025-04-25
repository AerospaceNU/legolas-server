import cv2

from legolas_tracking.tracked_object import TrackerObject


def draw_tracked_object(frame, obj: TrackerObject, color=(0, 255, 0)):
    """
    Draws the bounding box and label of a tracked object onto a cv2 frame.

    Parameters:
        frame: The OpenCV image to draw on.
        obj: The TrackerObject to draw.
        color: Optional RGB color for the rectangle and text.
    """
    # Bounding box coordinates
    top_left = tuple(map(int, (obj.bbox.top_left.x, obj.bbox.top_left.y)))
    bottom_right = tuple(map(int, (obj.bbox.bottom_right.x, obj.bbox.bottom_right.y)))

    # Draw rectangle
    cv2.rectangle(frame, top_left, bottom_right, color, thickness=2)

    # Label: "class_name (ID)"
    label = f"{obj.class_name}({obj.persistent_id}) ({obj.confidence:0.4f})"
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size
    label_bg_bottom = top_left[1]

    # Draw label text
    cv2.putText(
        frame,
        label,
        (top_left[0] + 2, label_bg_bottom - 2),
        font,
        font_scale,
        (0, 0, 0),  # text color: black for contrast
        thickness=2,
    )
