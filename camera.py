import cv2
import mediapipe as mp
import numpy as np
import math
import socket
from mediapipe.framework.formats import landmark_pb2

# ---------------- CONFIG ----------------
ALPHA = 0.3
MAX_DELTA = 40
PI_IP = "172.20.10.9"
PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

prev_angles = {'rs': 90, 're': 90, 'ls': 90, 'le': 90}
epsilon = 0.1

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
    ba = np.array([a.y - b.y, a.x - b.x])
    bc = np.array([c.y - b.y, c.x - b.x])
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.dot(ba, bc) / (n1 * n2)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def draw_arm(frame, sh, el, wr, color):
    h, w, _ = frame.shape
    pts = []
    for lm in [sh, el, wr]:
        pts.append((int(lm.x * w), int(lm.y * h)))

    cv2.line(frame, pts[0], pts[1], color, 3)
    cv2.line(frame, pts[1], pts[2], color, 3)

    for p in pts:
        cv2.circle(frame, p, 6, color, -1)

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

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    ) as pose:

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                res = pose.process(img)

                if not res.pose_landmarks:
                    cv2.imshow("Pi Robot Arm Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                lm = res.pose_landmarks.landmark

                # ---------------- RIGHT ARM ----------------
                r_sh, r_el, r_wr = lm[12], lm[14], lm[16]
                a = landmark_pb2.NormalizedLandmark()
                a.x, a.y, a.z, a.visibility = r_sh.x, r_sh.y + epsilon, r_sh.z, r_sh.visibility

                if all(is_visible(j) for j in [r_sh, r_el, r_wr]):
                    re = angle_yz(r_sh, r_el, r_wr)
                    rs = angle_yz(a, r_sh, r_el)

                    rs = smooth('rs', rate_limit(prev_angles['rs'], rs))
                    re = smooth('re', rate_limit(prev_angles['re'], re))

                    rs = np.clip(rs, 0, 180)
                    re = np.clip(re, 0, 180)

                    sock.sendto(f"R_SHOULDER:{rs:.2f}".encode(), (PI_IP, PORT))
                    sock.sendto(f"R_ELBOW:{re:.2f}".encode(), (PI_IP, PORT))

                    draw_arm(frame, r_sh, r_el, r_wr, (0, 255, 0))

                    cv2.putText(frame,
                                f"R Shoulder: {rs:.1f}  R Elbow: {re:.1f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

                # ---------------- LEFT ARM ----------------
                l_sh, l_el, l_wr = lm[11], lm[13], lm[15]
                b = landmark_pb2.NormalizedLandmark()
                b.x, b.y, b.z, b.visibility = l_sh.x, l_sh.y + epsilon, l_sh.z, l_sh.visibility

                if all(is_visible(j) for j in [l_sh, l_el, l_wr]):
                    le = angle_yz(l_sh, l_el, l_wr)
                    ls = angle_yz(b, l_sh, l_el)

                    ls = smooth('ls', rate_limit(prev_angles['ls'], ls))
                    le = smooth('le', rate_limit(prev_angles['le'], le))

                    ls = np.clip(ls, 0, 180)
                    le = np.clip(le, 0, 180)

                    sock.sendto(f"L_SHOULDER:{ls:.2f}".encode(), (PI_IP, PORT))
                    sock.sendto(f"L_ELBOW:{le:.2f}".encode(), (PI_IP, PORT))

                    draw_arm(frame, l_sh, l_el, l_wr, (255, 0, 0))

                    cv2.putText(frame,
                                f"L Shoulder: {ls:.1f}  L Elbow: {le:.1f}",
                                (320, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 0), 2)

                cv2.imshow("Pi Robot Arm Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopped")

        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
