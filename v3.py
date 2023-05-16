import subprocess
import time
import cv2
import mediapipe as mp
import pyautogui

# Using mediapipe to set up hand recognition
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)


def distance(a, b):
    return abs(a - b)


class Point3D(object):
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z


def mirror(landmark):
    res = []
    for l in landmark:
        res.append(Point3D(-l.x, l.y, l.z))
    return res


state = "unknown"
start_time = time.time()


def position(p):
    global state
    global start_time

    if state == "unknown":
        start_time = time.time()
        state = p
        print("unknown -> ", p)
    elif state != "hold" and (time.time() - start_time) > 0.3:
        action(p)
        state = "hold"
        print(time.time(), " : known -> hold")


def action(p):
    if p == "thumb_up":
        pyautogui.hotkey('space')
    elif p == "open_hand":
        pyautogui.hotkey('s')
    elif p == "zero_hand":
        pyautogui.hotkey('m')
    elif p == "victory_hand":
        pyautogui.hotkey('right')
    elif p == "fist_hand":
        pyautogui.hotkey('ctrl', 'q')
    elif p == "fists_hands":
        pyautogui.hotkey('ctrl', 'q')
        exit(0)
    elif p == "victories_hands":
        pyautogui.hotkey('left')
    elif p == "square_hands":
        pyautogui.hotkey('f')


def thumb_up(handLandmarks):
    return handLandmarks[4].y < handLandmarks[3].y \
        and handLandmarks[8].x > handLandmarks[6].x \
        and handLandmarks[12].x > handLandmarks[10].x \
        and handLandmarks[16].x > handLandmarks[14].x \
        and handLandmarks[20].x > handLandmarks[18].x \
        and handLandmarks[17].x < handLandmarks[0].x


def open_hand(handLandmarks):
    return handLandmarks[4].x < handLandmarks[3].x \
        and handLandmarks[8].y < handLandmarks[6].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y < handLandmarks[14].y \
        and handLandmarks[20].y < handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def zero_hand(handLandmarks):
    return distance(handLandmarks[4].x, handLandmarks[8].x) < 0.2 \
        and distance(handLandmarks[4].y, handLandmarks[8].y) < 0.2 \
        and handLandmarks[7].y < handLandmarks[8].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y < handLandmarks[14].y \
        and handLandmarks[20].y < handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def victory_hand(handLandmarks):
    return handLandmarks[4].x > handLandmarks[3].x \
        and handLandmarks[8].y < handLandmarks[6].y \
        and handLandmarks[12].y < handLandmarks[10].y \
        and handLandmarks[16].y > handLandmarks[14].y \
        and handLandmarks[20].y > handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def fist_hand(handLandmarks):
    return distance(handLandmarks[4].x, handLandmarks[10].x) < 0.03 \
        and distance(handLandmarks[4].y, handLandmarks[10].y) < 0.03 \
        and handLandmarks[8].y > handLandmarks[6].y \
        and handLandmarks[12].y > handLandmarks[10].y \
        and handLandmarks[16].y > handLandmarks[14].y \
        and handLandmarks[20].y > handLandmarks[18].y \
        and handLandmarks[17].x > handLandmarks[0].x


def square_hands(rightLandmarks, leftLandmarks):
    return rightLandmarks[4].y < rightLandmarks[3].y \
        and rightLandmarks[8].x < rightLandmarks[6].x \
        and rightLandmarks[12].x > rightLandmarks[10].x \
        and rightLandmarks[16].x > rightLandmarks[14].x \
        and rightLandmarks[20].x > rightLandmarks[18].x \
        and rightLandmarks[17].x < rightLandmarks[0].x \
        and leftLandmarks[4].y > leftLandmarks[3].y \
        and leftLandmarks[8].x > leftLandmarks[6].x \
        and leftLandmarks[12].x < leftLandmarks[10].x \
        and leftLandmarks[16].x < leftLandmarks[14].x \
        and leftLandmarks[20].x < leftLandmarks[18].x \
        and leftLandmarks[17].x > leftLandmarks[0].x \
        and distance(rightLandmarks[4].y, leftLandmarks[8].y) < 0.03 \
        and distance(rightLandmarks[4].x, leftLandmarks[8].x) < 0.03 \
        and distance(leftLandmarks[4].y, rightLandmarks[8].y) < 0.03 \
        and distance(leftLandmarks[4].x, rightLandmarks[8].x) < 0.03


def event_loop():
    global state

    prev_time = 0
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:

            # print(len(results.multi_hand_landmarks))

            if len(results.multi_hand_landmarks) == 2:

                left_landmarks = results.multi_hand_landmarks[0]
                right_landmarks = results.multi_hand_landmarks[1]

                # print(left_landmarks.landmark)
                # print(mirror(left_landmarks.landmark))

                if fist_hand(right_landmarks.landmark) and fist_hand(mirror(left_landmarks.landmark)):
                    position("fists_hands")
                elif victory_hand(right_landmarks.landmark) and victory_hand(mirror(left_landmarks.landmark)):
                    position("victories_hands")
                elif square_hands(right_landmarks.landmark, left_landmarks.landmark):
                    position("square_hands")
                else:
                    state = "unknown"
                    print(" -> unknown")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, left_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, right_landmarks, mp_hands.HAND_CONNECTIONS)

            elif len(results.multi_hand_landmarks) == 1:

                hand_landmarks = results.multi_hand_landmarks[0]

                # ranked from most to least restrictive
                if zero_hand(hand_landmarks.landmark):
                    position("zero_hand")
                elif thumb_up(hand_landmarks.landmark):
                    position("thumb_up")
                elif fist_hand(hand_landmarks.landmark):
                    position("fist_hand")
                elif victory_hand(hand_landmarks.landmark):
                    position("victory_hand")
                elif open_hand(hand_landmarks.landmark):
                    position("open_hand")
                else:
                    state = "unknown"
                    print(" -> unknown")

                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        else:
            state = "unknown"
            print(" no hand -> unknown")

        # calcul fps
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time))
        prev_time = cur_time
        text = "fps : " + str(fps)

        cv2.putText(image, text, (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        # print(fps)

        # Display image
        cv2.imshow('MediaPipe Hands', image)

        # Echap
        if cv2.waitKey(1) == 27:
            break


# For webcam input:
cap = cv2.VideoCapture(0)

help_image = cv2.imread('sign_dictionnary.png', cv2.IMREAD_UNCHANGED)

scale_percent = 40  # percent of original size
width = int(help_image.shape[1] * scale_percent / 100)
height = int(help_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(help_image, dim, interpolation=cv2.INTER_AREA)

cv2.imshow("Help", resized)

# Using the subprocess module to open VLC
vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
media_file = r'C:\Users\Albane\Pictures\Perso\debussy_arabesque_num_1.mp4'
subprocess.Popen([vlc_path, media_file])

event_loop()
cap.release()
cv2.destroyAllWindows()
