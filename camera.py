import cv2
import mediapipe as mp
import numpy as np
import math
import socket
from mediapipe.framework.formats import landmark_pb2

# ---------------- CONFIG ----------------
ALPHA = 0.3           # smoother for Pi
MAX_DELTA = 40        # max angle change per frame
PI_IP = "172.20.10.9"
PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

prev_angles = {'rs': 90, 're': 90, 'ls': 90, 'le': 90}
epsilon = 0.1  # small offset to prevent zero-length vectors

# ---------------- HELPERS ----------------
def smooth(name, val):
    out = ALPHA * val + (1 - ALPHA) * prev_angles[name]
    prev_angles[name] = out
    return out

def rate_limit(prev, new):
    d = new - prev
    if abs(d) > MAX_DELTA:
        new = prev + math.copysign(MAX_DELTA, d)
    return new

def is_visible(lm, t=0.6):
    return lm.visibility > t

def angle_yz(a, b, c):
    """Compute angle between three landmarks in Y-Z plane"""
    ba = np.array([a.y - b.y, a.x - b.x])
    bc = np.array([c.y - b.y, c.x - b.x])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# ---------------- MAIN ----------------
def main():
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("FATAL: Camera not found!")
        return

    print("Pi Camera Active. Press Ctrl+C to stop.")

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5,
                      model_complexity=0) as pose:

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # --- Ensure proper frame shape ---
                if frame.ndim != 3 or frame.shape[2] != 3:
                    try:
                        frame = frame.reshape(480, 640, 3)
                    except:
                        continue
                if frame.shape[:2] != (480, 640):
                    frame = cv2.resize(frame, (640, 480))

                frame = cv2.flip(frame, 1)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                res = pose.process(img)

                if not res.pose_landmarks:
                    cv2.imshow('Pi Robot Arm Control', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                lm = res.pose_landmarks.landmark

                # ---------------- RIGHT ARM ----------------
                r_sh, r_el, r_wr, r_hp = lm[12], lm[14], lm[16], lm[24]
                a = landmark_pb2.NormalizedLandmark()
                a.x, a.y, a.z, a.visibility = r_sh.x, r_sh.y + epsilon, r_sh.z, r_sh.visibility

                if all(is_visible(j) for j in [r_sh, r_el, r_wr, r_hp]):
                    re = angle_yz(r_sh, r_el, r_wr)
                    rs = angle_yz(a, r_sh, r_el)

                    # Apply rate limiting & smoothing
                    rs = smooth('rs', rate_limit(prev_angles['rs'], rs))
                    re = smooth('re', rate_limit(prev_angles['re'], re))

                    # Optional inversion if angles look mirrored
                    # rs = 180 - rs
                    # re = 180 - re

                    # Sanity check: only send angles 0-180
                    rs = np.clip(rs, 0, 180)
                    re = np.clip(re, 0, 180)

                    # Send each joint separately
                    sock.sendto(f"R_SHOULDER:{rs:.2f}".encode(), (PI_IP, PORT))
                    sock.sendto(f"R_ELBOW:{re:.2f}".encode(), (PI_IP, PORT))

                    # Debug
                    print(f"R | Shoulder: {rs:.1f} | Elbow: {re:.1f}", end="\r")

                # ---------------- LEFT ARM ----------------
                l_sh, l_el, l_wr, l_hp = lm[11], lm[13], lm[15], lm[23]
                b = landmark_pb2.NormalizedLandmark()
                b.x, b.y, b.z, b.visibility = l_sh.x, l_sh.y + epsilon, l_sh.z, l_sh.visibility

                if all(is_visible(j) for j in [l_sh, l_el, l_wr, l_hp]):
                    le = angle_yz(l_sh, l_el, l_wr)
                    ls = angle_yz(b, l_sh, l_el)

                    ls = smooth('ls', rate_limit(prev_angles['ls'], ls))
                    le = smooth('le', rate_limit(prev_angles['le'], le))

                    ls = np.clip(ls, 0, 180)
                    le = np.clip(le, 0, 180)

                    sock.sendto(f"L_SHOULDER:{ls:.2f}".encode(), (PI_IP, PORT))
                    sock.sendto(f"L_ELBOW:{le:.2f}".encode(), (PI_IP, PORT))

                    print(f"L | Shoulder: {ls:.1f} | Elbow: {le:.1f}", end="\r")

                # Show camera frame
                cv2.imshow('Pi Robot Arm Control', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n--- Stopped by User ---")

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
