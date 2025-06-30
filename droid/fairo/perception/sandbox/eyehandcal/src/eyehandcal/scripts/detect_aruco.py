import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import numpy as np


# ------------------- Config -------------------
ARUCO_DICT = {
    "4x4_50": aruco.DICT_4X4_50,
    "5x5_100": aruco.DICT_5X5_100,
    "6x6_250": aruco.DICT_6X6_250,
    "7x7_1000": aruco.DICT_7X7_1000,
    "original": aruco.DICT_ARUCO_ORIGINAL,
    "apriltag": aruco.DICT_APRILTAG_36h11
}
USE_REALSENSE = True  # Set False to use webcam
SELECTED_DICT = "6x6_250"  # Change this to your desired dictionary
# ---------------------------------------------

def initialize_camera():
    if USE_REALSENSE:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        return pipeline
    else:
        cap = cv2.VideoCapture(0)
        return cap

def get_frame(camera):
    if USE_REALSENSE:
        frames = camera.wait_for_frames()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data())
    else:
        ret, frame = camera.read()
        return frame if ret else None

def detect_aruco(frame, dict_type="6x6_250"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    parameters = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids

def draw_aruco(frame, corners, ids):
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in enumerate(corners):
            c = corner[0]
            for j in range(4):
                pt1 = tuple(c[j].astype(int))
                pt2 = tuple(c[(j+1)%4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            center = tuple(c.mean(axis=0).astype(int))
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {ids[i][0]}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

def main():
    print(f"[INFO] Using ArUco Dictionary: {SELECTED_DICT}")
    camera = initialize_camera()

    while True:
        frame = get_frame(camera)
        if frame is None:
            continue

        corners, ids = detect_aruco(frame, SELECTED_DICT)
        output = draw_aruco(frame, corners, ids)

        cv2.imshow("ArUco Marker Detection", output)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    if USE_REALSENSE:
        camera.stop()
    else:
        camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
