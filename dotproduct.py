import cv2
import mediapipe as mp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- CONFIG ----------------

ALPHA = 0.5
MAX_DELTA = 5

prev_angles = {
    'rs': 90, 're': 90,
    'ls': 90, 'le': 90
}

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
epsilon = 0.1    

def angle_yz(a, b, c):
    ba = np.array([a.y - b.y, a.z - b.z])
    bc = np.array([c.y - b.y, c.z - b.z])
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# ---------- CV2 DRAWING FUNCTION ----------
def draw_arm_lines(frame, hip, shoulder, elbow, wrist):
    h, w, _ = frame.shape
    
    # Convert normalized coordinates to pixels
    hip_xy = (int(hip.x * w), int(hip.y * h))
    shoulder_xy = (int(shoulder.x * w), int(shoulder.y * h))
    elbow_xy = (int(elbow.x * w), int(elbow.y * h))
    wrist_xy = (int(wrist.x * w), int(wrist.y * h))
    
    # Draw lines
    cv2.line(frame, hip_xy, shoulder_xy, (0, 255, 0), 2)
    cv2.line(frame, shoulder_xy, elbow_xy, (0, 255, 0), 2)
    cv2.line(frame, elbow_xy, wrist_xy, (0, 255, 0), 2)
    
    # Draw circles at joints with colors
    cv2.circle(frame, hip_xy, 5, (0, 0, 255), -1)       # red
    cv2.circle(frame, shoulder_xy, 5, (0, 255, 255), -1) # yellow
    cv2.circle(frame, elbow_xy, 5, (255, 0, 0), -1)     # blue
    cv2.circle(frame, wrist_xy, 5, (0, 255, 0), -1)     # green

# ---------------- MAIN ----------------

def main():
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    # Matplotlib setup
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    with mp_pose.Pose() as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img)

            if not res.pose_landmarks:
                cv2.imshow('Robot Control System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            lm = res.pose_landmarks.landmark

            # ---------- RIGHT ARM ----------
            r_sh = lm[12]
            r_el = lm[14]
            r_wr = lm[16]
            r_hp = lm[24]
            a  = mp.solutions.pose.landmark_pb2.NormalizedLandmark()
            a.x = r_sh.x
            a.y = r_sh.y + epsilon
            a.z = r_sh.z
            a.visibility = r_sh.visibility            

            if all(is_visible(p) for p in [r_sh, r_el, r_wr, r_hp]):
                re = angle_yz(r_sh, r_el, r_wr)
                rs = angle_yz(a, r_sh, r_el)

                rs = smooth('rs', rate_limit(prev_angles['rs'], rs))
                re = smooth('re', rate_limit(prev_angles['re'], re))

                # CV2 draw
                draw_arm_lines(frame, r_hp, r_sh, r_el, r_wr)

                # Matplotlib plot
                ax.clear()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(0, 1)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                ax.scatter(r_hp.x, r_hp.y, r_hp.z, c='red', s=50, label='Hip')
                ax.scatter(r_sh.x, r_sh.y, r_sh.z, c='yellow', s=50, label='Shoulder')
                ax.scatter(r_el.x, r_el.y, r_el.z, c='blue', s=50, label='Elbow')
                ax.scatter(r_wr.x, r_wr.y, r_wr.z, c='green', s=50, label='Wrist')

                ax.plot([r_hp.x, r_sh.x, r_el.x, r_wr.x],
                        [r_hp.y, r_sh.y, r_el.y, r_wr.y],
                        [r_hp.z, r_sh.z, r_el.z, r_wr.z],
                        c='black')

            # ---------- LEFT ARM ----------
            l_sh = lm[11]
            l_el = lm[13]
            l_wr = lm[15]
            l_hp = lm[23]
            b = mp.solutions.pose.landmark_pb2.NormalizedLandmark()
            b.x = l_sh.x
            b.y = l_sh.y + 0.1
            b.z = l_sh.z 
            b.visibility = l_sh.visibility            

            if all(is_visible(p) for p in [l_sh, l_el, l_wr, l_hp]):
                le = angle_yz(l_sh, l_el, l_wr)
                ls = angle_yz(b, l_sh, l_el)

                ls = smooth('ls', rate_limit(prev_angles['ls'], ls))
                le = smooth('le', rate_limit(prev_angles['le'], le))

                # CV2 draw
                draw_arm_lines(frame, l_hp, l_sh, l_el, l_wr)
                
                cv2.putText(frame, "TRACKING ACTIVE", (50, 50), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Shoulder: {int(rs )}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (0,255,0), 2)

                cv2.putText(frame, f"Elbow: {int(re)}",
                     (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                     1, (0,255,0), 2)
                

            # Show CV2 frame
            cv2.imshow('Robot Control System', frame)
            plt.draw()
            plt.pause(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

