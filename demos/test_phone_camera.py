"""
Quick test: connect to phone via IP Webcam app and display stream.

Usage:
    python demos/test_phone_camera.py
    python demos/test_phone_camera.py --url http://192.168.29.163:8080
"""

import cv2
import time
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://192.168.29.163:8080",
                   help="Base URL from IP Webcam app (no trailing slash)")
    args = p.parse_args()

    # IP Webcam exposes the video stream at /video
    stream_url = f"{args.url}/video"
    print(f"Connecting to: {stream_url}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("FAILED to open stream. Check:")
        print("  - Phone and laptop on same WiFi")
        print("  - IP Webcam app server is running")
        print("  - URL/IP is correct (look at the app screen)")
        return

    # Read one frame to get resolution
    ret, frame = cap.read()
    if not ret:
        print("Connected but no frames received. Try restarting the app server.")
        return

    h, w = frame.shape[:2]
    print(f"Connected. Resolution: {w}x{h}")
    print("Press 'q' to quit. Watch the FPS counter.\n")

    frame_count = 0
    start = time.time()
    last_print = start

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream dropped, reconnecting...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            continue

        frame_count += 1
        now = time.time()
        elapsed = now - start
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Print FPS every second
        if now - last_print >= 1.0:
            print(f"FPS: {fps:.1f}  |  Frames: {frame_count}  |  Resolution: {w}x{h}")
            last_print = now

        # Overlay info on the frame
        cv2.putText(frame, f"Phone Camera | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"{w}x{h} | Press 'q' to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Phone Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal stats: {frame_count} frames in {elapsed:.1f}s = {fps:.1f} FPS")


if __name__ == "__main__":
    main()
