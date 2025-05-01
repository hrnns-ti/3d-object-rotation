import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import threading
import queue
import time

# Queue untuk komunikasi data dari OpenCV ke OpenGL
data_queue = queue.Queue()

# Variabel global untuk OpenGL (nilai default)
scale = 1.0
rot_x = 0.0
rot_y = 0.0

running = True  # Flag untuk menghentikan program

# Faktor sensitivitas rotasi (bisa disesuaikan)
rotation_sensitivity = 0.10  # Kurangi sensitivitas

# Ambang batas jarak untuk deteksi kepalan tangan
clenched_threshold = 50

# Jumlah frame untuk kalibrasi awal
calibration_frames = 30
calibrated = False
initial_index_x = 0
initial_index_y = 0
calibration_count = 0

# Faktor smoothing (0 = tanpa smoothing, 1 = smoothing penuh)
smoothing_factor = 0.2  # Tambah smoothing

# Dead zone untuk rotasi (dalam pixel)
dead_zone = 20

# Kecepatan rotasi maksimum per frame
max_rotation_speed = 6

# Ambang batas pergerakan (dalam pixel)
movement_threshold = 3

# Kecepatan rotasi saat ini
current_rot_x_speed = 0
current_rot_y_speed = 0
speed_smoothing_factor = 0.07

# Parameter Garis Koordinat (OpenCV)
panjang_garis = 100
warna_x = (0, 0, 255)  # Merah
warna_y = (0, 255, 0)  # Hijau
tebal_garis = 1  # Dikurangi jadi 10%
tebal_garis_dasar = 3  # Ketebalan garis dasar

# Setup pencahayaan OpenGL
def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 10, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.7, 0.7, 0.7, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

# Fungsi menggambar kubus 3D
def draw_cube():
    glBegin(GL_QUADS)

    # Atas (merah)
    glNormal3f(0, 1, 0)
    glColor3f(1, 0, 0)
    glVertex3f(1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)

    # Bawah (hijau)
    glNormal3f(0, -1, 0)
    glColor3f(0, 1, 0)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)

    # Depan (biru)
    glNormal3f(0, 0, 1)
    glColor3f(0, 0, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)

    # Belakang (kuning)
    glNormal3f(0, 0, -1)
    glColor3f(1, 1, 0)
    glVertex3f(1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)

    # Kiri (magenta)
    glNormal3f(-1, 0, 0)
    glColor3f(1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)

    # Kanan (cyan)
    glNormal3f(1, 0, 0)
    glColor3f(0, 1, 1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, -1, -1)

    glEnd()

# Fungsi display OpenGL
def display():
    global scale, rot_x, rot_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 7, 0, 0, 0, 0, 1, 0)

    # Aplikasikan transformasi (rotasi dan skala)
    glRotatef(rot_x, 1, 0, 0)
    glRotatef(rot_y, 0, 1, 0)
    glScalef(scale, scale, scale)

    # Gambar kubus
    draw_cube()

    glutSwapBuffers()

# Fungsi reshape OpenGL
def reshape(w, h):
    if h == 0:
        h = 1
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w / h, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

# Inisialisasi OpenGL
def init_gl():
    glClearColor(0.1, 0.1, 0.1, 1)
    glEnable(GL_DEPTH_TEST)
    setup_lighting()

# Fungsi idle OpenGL
def idle():
    global scale, rot_x, rot_y
    try:
        new_scale, new_rot_x, new_rot_y = data_queue.get_nowait()
        scale = new_scale
        rot_x = new_rot_x
        rot_y = new_rot_y
    except queue.Empty:
        pass
    glutPostRedisplay()

# Thread OpenGL
def opengl_thread():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutCreateWindow(b"Objek 3D")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutIdleFunc(idle)
    init_gl()
    glutMainLoop()

# Fungsi untuk mendeteksi apakah tangan terkepal
def is_hand_clenched(hand, threshold=clenched_threshold):
    thumb_tip = hand['lmList'][4]
    distances = [np.linalg.norm(np.array(thumb_tip) - np.array(hand['lmList'][i])) for i in range(8, 21, 4)]
    return all(d < threshold for d in distances)

# Fungsi untuk melakukan Lerp (Linear Interpolation)
def lerp(a, b, t):
    return a + (b - a) * t

# Fungsi untuk menggambar sistem koordinat di OpenCV
def gambar_koordinat(img, origin, panjang, tebal=1):
    x, y = origin
    # Gambar sumbu X (merah)
    cv2.line(img, (x, y), (x + panjang, y), warna_x, tebal)
    cv2.putText(img, 'X', (x + panjang, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_x, 2)

    # Gambar sumbu Y (hijau)
    cv2.line(img, (x, y), (x, y - panjang), warna_y, tebal)
    cv2.putText(img, 'Y', (x + 10, y - panjang), cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna_y, 2)

# Thread utama OpenCV
def main():
    global running, rotation_sensitivity, rot_x, rot_y
    global calibrated, initial_index_x, initial_index_y, calibration_count
    global smoothing_factor, dead_zone, max_rotation_speed, movement_threshold
    global current_rot_x_speed, current_rot_y_speed, speed_smoothing_factor
    global tebal_garis

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    detector = HandDetector(detectionCon=0.8, maxHands=2)

    # Mulai thread OpenGL
    thread_gl = threading.Thread(target=opengl_thread, daemon=True)
    thread_gl.start()

    prev_index_x_right = None
    prev_index_y_right = None

    update_interval = 2.5
    last_update_time = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame, flipType=False)

        rot_x_val = 0
        rot_y_val = 0
        scale_val = scale

        if hands:
            hands = sorted(hands, key=lambda x: x['center'][0])

            # Proses tangan kiri (skala)
            if len(hands) >= 1:
                hand_left = hands[0]
                lmList_left = hand_left['lmList']
                thumb_tip_left = lmList_left[4]
                index_tip_left = lmList_left[8]

                thumb_pos_left = (int(thumb_tip_left[0]), int(thumb_tip_left[1]))
                index_pos_left = (int(index_tip_left[0]), int(index_tip_left[1]))

                dist_left = np.linalg.norm(np.array(thumb_pos_left) - np.array(index_pos_left))
                scale_val = np.interp(dist_left, [30, 200], [0.5, 2.5])

                cv2.line(img, thumb_pos_left, index_pos_left, (0, 0, 255), tebal_garis_dasar)
                cv2.putText(img, f'Scale: {scale_val:.2f}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), tebal_garis_dasar)

            # Proses tangan kanan (rotasi)
            if len(hands) >= 2:
                hand_right = hands[1]
                lmList_right = hand_right['lmList']
                if lmList_right and len(lmList_right) > 8:
                    index_tip_right = lmList_right[8]
                    index_pos_right = (int(index_tip_right[0]), int(index_tip_right[1]))

                    # Kalibrasi Awal
                    if not calibrated and calibration_count < calibration_frames:
                        initial_index_x += index_pos_right[0]
                        initial_index_y += index_pos_right[1]
                        calibration_count += 1
                        cv2.putText(img, f'Calibrating: {calibration_count}/{calibration_frames}', (10, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), tebal_garis_dasar)
                        if calibration_count == calibration_frames:
                            initial_index_x /= calibration_count
                            initial_index_y /= calibration_count
                            calibrated = True
                            print("Kalibrasi Selesai")

                    # Update koordinat awal setiap 3 detik jika sudah kalibrasi dan tangan tidak terkepal
                    if calibrated and not is_hand_clenched(hand_right):
                        current_time = time.time()
                        if current_time - last_update_time > update_interval:
                            initial_index_x = index_pos_right[0]
                            initial_index_y = index_pos_right[1]
                            last_update_time = current_time
                            print(f"Koordinat awal diupdate: ({initial_index_x:.2f}, {initial_index_y:.2f})")

                        if prev_index_x_right is not None and prev_index_y_right is not None:
                            delta_x = index_pos_right[0] - initial_index_x
                            delta_y = index_pos_right[1] - initial_index_y
                            delta_x = np.clip(delta_x, -50, 50)
                            delta_y = np.clip(delta_y, -50, 50)

                            if abs(delta_x) < dead_zone:
                                delta_x = 0
                            if abs(delta_y) < dead_zone:
                                delta_y = 0

                            if abs(delta_x) > movement_threshold or abs(delta_y) > movement_threshold:
                                rot_x_val = delta_y * rotation_sensitivity
                                rot_y_val = delta_x * rotation_sensitivity
                        else:
                            rot_x_val = 0
                            rot_y_val = 0

                        rot_x_val = np.clip(rot_x_val, -max_rotation_speed, max_rotation_speed)
                        rot_y_val = np.clip(rot_y_val, -max_rotation_speed, max_rotation_speed)

                        current_rot_x_speed = lerp(current_rot_x_speed, rot_x_val, speed_smoothing_factor)
                        current_rot_y_speed = lerp(current_rot_y_speed, rot_y_val, speed_smoothing_factor)

                        prev_index_x_right = index_pos_right[0]
                        prev_index_y_right = index_pos_right[1]

                        cv2.putText(img, f'RotX: {rot_x:.1f}', (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), tebal_garis_dasar)
                        cv2.putText(img, f'RotY: {rot_y:.1f}', (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), tebal_garis_dasar)
                    else:
                        current_rot_x_speed = lerp(current_rot_x_speed, 0, speed_smoothing_factor)
                        current_rot_y_speed = lerp(current_rot_y_speed, 0, speed_smoothing_factor)
                        prev_index_x_right = None
                        prev_index_y_right = None
                else:
                    prev_index_x_right = None
                    prev_index_y_right = None
            else:
                prev_index_x_right = None
                prev_index_y_right = None

        rot_x += current_rot_x_speed
        rot_y += current_rot_y_speed

        if data_queue.empty():
            data_queue.put((scale_val, rot_x, rot_y))

        if calibrated and 'lmList_right' in locals():
            origin_x = int(initial_index_x)
            origin_y = int(initial_index_y)
            gambar_koordinat(img, (origin_x, origin_y), panjang_garis, tebal_garis)

        cv2.imshow("Kontrol gerakan", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
        
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
