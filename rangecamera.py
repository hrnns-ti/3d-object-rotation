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
rotation_sensitivity = 0.15  # Kurangi sensitivitas

# Ambang batas jarak untuk deteksi kepalan tangan
clenched_threshold = 50

# Jumlah frame untuk kalibrasi awal
calibration_frames = 30
calibrated = False
initial_index_x = 0
initial_index_y = 0
calibration_count = 0

# Faktor smoothing (0 = tanpa smoothing, 1 = smoothing penuh)
smoothing_factor = 0.2 #Tambah smoothing

# Dead zone untuk rotasi (dalam pixel)
dead_zone = 20

# Kecepatan rotasi maksimum per frame
max_rotation_speed = 5

# Ambang batas pergerakan (dalam pixel)
movement_threshold = 5

# Kecepatan rotasi saat ini
current_rot_x_speed = 0
current_rot_y_speed = 0
speed_smoothing_factor = 0.07

# Setup pencahayaan OpenGL
def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION,  (0, 0, 10, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT,   (0.2, 0.2, 0.2, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,   (0.7, 0.7, 0.7, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR,  (1.0, 1.0, 1.0, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

# Fungsi menggambar kubus 3D
def draw_cube():
    glBegin(GL_QUADS)

    # Atas (merah)
    glNormal3f(0, 1, 0)
    glColor3f(1, 0, 0)
    glVertex3f( 1, 1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f( 1, 1, 1)

    # Bawah (hijau)
    glNormal3f(0, -1, 0)
    glColor3f(0, 1, 0)
    glVertex3f( 1, -1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, -1, -1)
    glVertex3f( 1, -1, -1)

    # Depan (biru)
    glNormal3f(0, 0, 1)
    glColor3f(0, 0, 1)
    glVertex3f( 1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, -1, 1)
    glVertex3f( 1, -1, 1)

    # Belakang
    glNormal3f(0, 0, -1)
    glColor3f(1, 1, 0)
    glVertex3f( 1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f( 1, 1, -1)

    # Kiri
    glNormal3f(-1, 0, 0)
    glColor3f(1, 0, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)

    # Kanan
    glNormal3f(1, 0, 0)
    glColor3f(0, 1, 1)
    glVertex3f( 1, 1, -1)
    glVertex3f( 1, 1, 1)
    glVertex3f( 1, -1, 1)
    glVertex3f( 1, -1, -1)

    glEnd()

# Fungsi display OpenGL
def display():
    global scale, rot_x, rot_y
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Atur posisi kamera
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
    # Cek apakah ada data baru dari queue
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
    # Hitung jarak antara ibu jari dan setiap jari lainnya
    distances = [np.linalg.norm(np.array(thumb_tip) - np.array(hand['lmList'][i])) for i in range(8, 21, 4)]
    # Jika semua jarak kurang dari ambang batas, anggap tangan terkepal
    return all(d < threshold for d in distances)

# Fungsi untuk melakukan Lerp (Linear Interpolation)
def lerp(a, b, t):
    return a + (b - a) * t

# Thread utama OpenCV
def main():
    global running, rotation_sensitivity, rot_x, rot_y
    global calibrated, initial_index_x, initial_index_y, calibration_count
    global smoothing_factor, dead_zone, max_rotation_speed, movement_threshold
    global current_rot_x_speed, current_rot_y_speed, speed_smoothing_factor

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = HandDetector(detectionCon=0.8, maxHands=2)  # Deteksi 2 tangan

    # Mulai thread OpenGL
    thread_gl = threading.Thread(target=opengl_thread, daemon=True)
    thread_gl.start()

    # Variabel untuk menyimpan posisi jari telunjuk sebelumnya
    prev_index_x_right = None
    prev_index_y_right = None

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hands, img = detector.findHands(frame, flipType=False)

        # Inisialisasi rot_x_val dan rot_y_val
        rot_x_val = 0
        rot_y_val = 0

        scale_val = scale  # Pertahankan nilai skala sebelumnya

        if hands:
            # Urutkan tangan berdasarkan posisi X (tangan kiri lebih kecil)
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

                # Hitung skala berdasarkan jarak ibu jari dan jari telunjuk
                scale_val = np.interp(dist_left, [30, 200], [0.5, 2.5])

                cv2.line(img, thumb_pos_left, index_pos_left, (255, 0, 0), 3)
                cv2.putText(img, f'Scale: {scale_val:.2f}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Proses tangan kanan (rotasi)
            if len(hands) >= 2:
                hand_right = hands[1]
                lmList_right = hand_right['lmList']
                # Periksa apakah lmList_right dan index_tip_right valid
                if lmList_right and len(lmList_right) > 8:

                    index_tip_right = lmList_right[8]
                    index_pos_right = (int(index_tip_right[0]), int(index_tip_right[1]))

                    # Kalibrasi Awal
                    if not calibrated and calibration_count < calibration_frames:
                        initial_index_x += index_pos_right[0]
                        initial_index_y += index_pos_right[1]
                        calibration_count += 1
                        cv2.putText(img, f'Calibrating: {calibration_count}/{calibration_frames}', (10, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if calibration_count == calibration_frames:
                            initial_index_x /= calibration_count
                            initial_index_y /= calibration_count
                            calibrated = True
                            print("Kalibrasi Selesai")

                    # Periksa apakah tangan kanan tidak terkepal dan kalibrasi sudah selesai
                    if calibrated and not is_hand_clenched(hand_right):
                        # Hitung jarak antara posisi saat ini dan posisi sebelumnya
                        if prev_index_x_right is not None and prev_index_y_right is not None:
                            delta_x = index_pos_right[0] - initial_index_x
                            delta_y = index_pos_right[1] - initial_index_y
                            # Batasi Delta
                            delta_x = np.clip(delta_x, -50, 50)
                            delta_y = np.clip(delta_y, -50, 50)

                            # Aplikasikan dead zone
                            if abs(delta_x) < dead_zone:
                                delta_x = 0
                            if abs(delta_y) < dead_zone:
                                delta_y = 0

                            # Lakukan rotasi hanya jika ada pergerakan
                            if abs(delta_x)> movement_threshold or abs(delta_y) > movement_threshold :
                            # Hitung jarak relatif dari posisi netral
                                # Kurangi sensitivitas rotasi
                                rot_x_val = delta_y * rotation_sensitivity
                                rot_y_val = delta_x * rotation_sensitivity

                        else:
                            rot_x_val = 0
                            rot_y_val = 0

                        # Batasi kecepatan rotasi
                        rot_x_val = np.clip(rot_x_val, -max_rotation_speed, max_rotation_speed)
                        rot_y_val = np.clip(rot_y_val, -max_rotation_speed, max_rotation_speed)

                        # Smoothing kecepatan
                        current_rot_x_speed = lerp(current_rot_x_speed, rot_x_val, speed_smoothing_factor)
                        current_rot_y_speed = lerp(current_rot_y_speed, rot_y_val, speed_smoothing_factor)

                        # Update posisi jari telunjuk sebelumnya
                        prev_index_x_right = index_pos_right[0]
                        prev_index_y_right = index_pos_right[1]

                        cv2.putText(img, f'RotX: {rot_x:.1f}', (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(img, f'RotY: {rot_y:.1f}', (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Perlambat kecepatan rotasi saat tangan terkepal atau tidak ada pergerakan
                        current_rot_x_speed = lerp(current_rot_x_speed, 0, speed_smoothing_factor)
                        current_rot_y_speed = lerp(current_rot_y_speed, 0, speed_smoothing_factor)
                        # Reset posisi jari telunjuk sebelumnya jika tangan terkepal
                        prev_index_x_right = None
                        prev_index_y_right = None
                else:
                    # Reset posisi jari telunjuk sebelumnya jika deteksi gagal
                    prev_index_x_right = None
                    prev_index_y_right = None

            else:
                # Reset posisi jari telunjuk sebelumnya jika tidak ada tangan kanan
                prev_index_x_right = None
                prev_index_y_right = None

        # Aplikasikan kecepatan rotasi saat ini
        rot_x += current_rot_x_speed
        rot_y += current_rot_y_speed

        # Kirim data ke queue (non-blocking)
        if data_queue.empty():
            data_queue.put((scale_val, rot_x, rot_y))

        cv2.imshow("Kontrol gerakan", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        # Batasi frame rate OpenCV agar tidak terlalu berat
        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
