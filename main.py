import cv2
from ultralytics import YOLO

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("yolo11m.pt")

# -------------------------------
# VIDEO
# -------------------------------
cap = cv2.VideoCapture("video/ha.mp4")

# -------------------------------
# VEHICLE COUNTS
# -------------------------------
total_count = 0

car_count = 0
bike_count = 0
bus_count = 0
truck_count = 0

counted_ids = set()

# -------------------------------
# COUNTING LINE
# -------------------------------
line_y = 350
offset = 10

# -------------------------------
# FRAME SKIP
# -------------------------------
frame_skip = 2
frame_count = 0

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames for speed
    if frame_count % frame_skip != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (960, 540))

    # -------------------------------
    # DETECTION + TRACKING
    # -------------------------------
    results = model.track(
        frame,
        persist=True,
        conf=0.15,
        iou=0.5
    )

    # -------------------------------
    # DRAW COUNTING LINE
    # -------------------------------
    cv2.line(
        frame,
        (0, line_y),
        (960, line_y),
        (255, 0, 0),
        3
    )

    # -------------------------------
    # PROCESS DETECTIONS
    # -------------------------------
    if results[0].boxes is not None and results[0].boxes.id is not None:

        for box in results[0].boxes:

            cls = int(box.cls[0])
            track_id = int(box.id[0])

            # Vehicle Classes
            # 2 = Car
            # 3 = Motorcycle
            # 5 = Bus
            # 7 = Truck

            if cls in [2, 3, 5, 7]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Center Point
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # -------------------------------
                # CLASS NAMES + COLORS
                # -------------------------------
                if cls == 2:
                    label_name = "Car"
                    color = (0, 255, 0)

                elif cls == 3:
                    label_name = "Bike"
                    color = (0, 255, 255)

                elif cls == 5:
                    label_name = "Bus"
                    color = (255, 0, 0)

                elif cls == 7:
                    label_name = "Truck"
                    color = (0, 0, 255)

                # -------------------------------
                # DRAW BOX
                # -------------------------------
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Draw Center Point
                cv2.circle(
                    frame,
                    (cx, cy),
                    4,
                    (255, 255, 255),
                    -1
                )

                # Label
                label = f"{label_name} ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                # -------------------------------
                # COUNTING LOGIC
                # -------------------------------
                bottom_y = y2

                if (line_y - offset) < bottom_y < (line_y + offset):

                    if track_id not in counted_ids:

                        counted_ids.add(track_id)

                        total_count += 1

                        # Separate counts
                        if cls == 2:
                            car_count += 1

                        elif cls == 3:
                            bike_count += 1

                        elif cls == 5:
                            bus_count += 1

                        elif cls == 7:
                            truck_count += 1

    # -------------------------------
    # UI PANEL
    # -------------------------------
    cv2.rectangle(frame, (10, 10), (260, 180), (0, 0, 0), -1)

    cv2.putText(frame,
                f"Total Vehicles: {total_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"Cars: {car_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Bikes: {bike_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"Buses: {bus_count}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    cv2.putText(frame,
                f"Trucks: {truck_count}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    # -------------------------------
    # SHOW WINDOW
    # -------------------------------
    cv2.imshow("AI Traffic Analysis System", frame)

    # ESC TO EXIT
    if cv2.waitKey(1) == 27:
        break

# -------------------------------
# RELEASE
# -------------------------------
cap.release()
cv2.destroyAllWindows()