import cv2
import mediapipe as mp
import numpy as np
# import pigpio
import math
import time

# ---------------- CONFIGURATION ----------------
PIN_R_SHOULDER, PIN_R_ELBOW = 18, 17
PIN_L_SHOULDER, PIN_L_ELBOW = 27, 22

LIMITS = {
    'shoulder': (20, 160),
    'elbow':    (5, 175)
}

# ROBOT PHYSICAL DIMENSIONS (Meters)
ROBOT_L1 = 0.15 
ROBOT_L2 = 0.12
ROBOT_TOTAL = ROBOT_L1 + ROBOT_L2

ALPHA_Y, ALPHA_Z = 0.3, 0.15
MIN_PULSE, MAX_PULSE = 500, 2500

# ---------------- STATE & FILTERS ----------------
class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None
    def apply(self, value):
        if self.state is None: self.state = value
        else: self.state = (self.alpha * value) + (1 - self.alpha) * self.state
        return self.state

filter_ry = EMAFilter(ALPHA_Y); filter_rz = EMAFilter(ALPHA_Z)
filter_ly = EMAFilter(ALPHA_Y); filter_lz = EMAFilter(ALPHA_Z)

# Global scale factor (Calculated during calibration)
reach_ratio = 1.0 

# ---------------- PIGPIO & HELPERS ----------------
# pi = pigpio.pi()

def set_servo_angle(pin, angle, joint_type):
    min_limit, max_limit = LIMITS[joint_type]
    safe_angle = max(min_limit, min(max_limit, angle))
    pulse = MIN_PULSE + (safe_angle / 180.0) * (MAX_PULSE - MIN_PULSE)
    # pi.set_servo_pulsewidth(pin, pulse)

def solve_ik(dy, dr):
    # Use ROBOT lengths for IK math
    h = math.sqrt(dy**2 + dr**2)
    h = min(h, ROBOT_TOTAL - 1e-6)

    phi = math.atan2(dy, dr)
    cos_psi = (ROBOT_L1**2 + h**2 - ROBOT_L2**2) / (2 * ROBOT_L1 * h)
    psi = math.acos(np.clip(cos_psi, -1, 1))

    shoulder_deg = math.degrees(phi - psi)
    cos_el = (ROBOT_L1**2 + ROBOT_L2**2 - h**2) / (2 * ROBOT_L1 * ROBOT_L2)
    elbow_deg = math.degrees(math.acos(np.clip(cos_el, -1, 1)))

    return shoulder_deg, elbow_deg

# ---------------- MAIN ----------------
def main():
    global reach_ratio
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    
    calibration_end_time = time.time() + 5 # 5 seconds for calibration
    calibrated = False

    with mp_pose.Pose(model_complexity=1) as pose:
        print("PHASE 1: CALIBRATION. Stand still with arms at your sides...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if res.pose_world_landmarks:
                wlm = res.pose_world_landmarks.landmark
                
                # --- CALIBRATION LOGIC ---
                if not calibrated:
                    # Measure Human L1 (Shoulder to Elbow) and L2 (Elbow to Wrist)
                    # We use Euclidean distance in 3D space
                    def dist3d(p1, p2):
                        return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)
                    
                    h_l1 = dist3d(wlm[12], wlm[14]) # Right shoulder to elbow
                    h_l2 = dist3d(wlm[14], wlm[16]) # Right elbow to wrist
                    human_total = h_l1 + h_l2

                    if human_total > 0.1: # Ensure we see a person
                        reach_ratio = ROBOT_TOTAL / human_total
                    
                    remaining = int(calibration_end_time - time.time())
                    cv2.putText(frame, f"CALIBRATING: {remaining}s", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if remaining <= 0:
                        calibrated = True
                        print(f"CALIBRATION DONE. Ratio: {reach_ratio:.2f}")

                # --- TRACKING LOGIC ---
                else:
                    # Apply reach_ratio to the raw camera deltas
                    dy_r = filter_ry.apply((wlm[16].y - wlm[12].y) * reach_ratio)
                    dr_r = filter_rz.apply((wlm[16].z - wlm[12].z) * reach_ratio)
                    
                    rs, re = solve_ik(dy_r, dr_r)
                    set_servo_angle(PIN_R_SHOULDER, rs + 90, 'shoulder')
                    set_servo_angle(PIN_R_ELBOW, 180 - re, 'elbow')

                    # Left Arm
                    dy_l = filter_ly.apply((wlm[15].y - wlm[11].y) * reach_ratio)
                    dr_l = filter_lz.apply((wlm[15].z - wlm[11].z) * reach_ratio)
                    
                    ls, le = solve_ik(dy_l, dr_l)
                    set_servo_angle(PIN_L_SHOULDER, ls + 90, 'shoulder')
                    set_servo_angle(PIN_L_ELBOW, 180 - le, 'elbow')

                    cv2.putText(frame, "TRACKING ACTIVE", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Shoulder: {int(rs + 90)}",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

                    cv2.putText(frame, f"Elbow: {int(180 - re)}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

            cv2.imshow('Robot Control System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    # pi.stop()

if __name__ == "__main__":
    main()