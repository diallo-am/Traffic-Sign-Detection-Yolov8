import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import pygame
import math
import sys
import time
from threading import Lock
import random

# ==================== CONF ====================
IP_WEBCAM_URL = "http://192.168.1.--:8080/video"   
LATENCE = 0.5
SPEED_STEP = 2

# Fréquence de mise à jour
SIM_DT = 0.2

# Seuils
OVERSPEED_MARGIN = 3  # km/h
OVERSPEED_BEEP_COOLDOWN = 1.5  # entre deux bips


shared_state = {
    'limit': None,         
    'vehicle_speed': 30,    # vitesse véhicule initiale simulé
    'rpm': 0.0,
    'gear': 'N',
    'mileage': 14356,
    'overspeed': False,
}
state_lock = Lock()

TARGET_SPEED = 30         
CURRENT_SPEED = 30        # vitesse actuelle simulée
LIMIT_TIMEOUT = 5 # temps affichage du panneau
last_limit_time = 0  # dernière détection
last_beep_time = 0        
beep_sound = None         

# ==================== YOLO ====================
Valid_model = YOLO("best.pt")  # Chemin du modèle

def normalize_image(image):
    return image / 255.0

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# ==================== AUDIO  ====================
def init_beep():
    global beep_sound
    try:
        pygame.mixer.init()
        #bip 700 Hz de 120 ms
        sr = 22050
        dur = 0.12
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        wave = (0.5*np.sin(2*np.pi*700*t)).astype(np.float32)
        beep_sound = pygame.mixer.Sound(wave)
    except Exception as e:
        print("[AUDIO] Initialisation impossible :", e)
        beep_sound = None

def play_beep():
    global last_beep_time, beep_sound
    if beep_sound is None:
        return
    now = time.time()
    if now - last_beep_time >= OVERSPEED_BEEP_COOLDOWN:
        last_beep_time = now
        try:
            beep_sound.play()
        except:
            pass


# ==================== Mise à jour vitesse ====================

def update_vehicle_speed():
    global CURRENT_SPEED, TARGET_SPEED, last_limit_time

    while True:
        time.sleep(LATENCE)

        with state_lock:
            limit = shared_state.get('limit', None)
            if limit is not None:
                last_limit_time = time.time()

            time_since_limit = time.time() - last_limit_time
            panneau_actif = (limit is not None) and (time_since_limit <= LIMIT_TIMEOUT)

            if not panneau_actif:
                change = random.uniform(-4, 8)
                TARGET_SPEED = max(0, min(140, CURRENT_SPEED + change))
            else:
                if CURRENT_SPEED > limit + OVERSPEED_MARGIN:
                    TARGET_SPEED = limit - 1  
                else:
                    change = random.uniform(-4, 8)
                    TARGET_SPEED = max(0, min(limit, CURRENT_SPEED + change))

            if CURRENT_SPEED < TARGET_SPEED:
                CURRENT_SPEED = min(CURRENT_SPEED + SPEED_STEP, TARGET_SPEED)
            elif CURRENT_SPEED > TARGET_SPEED:
                CURRENT_SPEED = max(CURRENT_SPEED - SPEED_STEP, TARGET_SPEED)

            shared_state['vehicle_speed'] = int(round(CURRENT_SPEED))
            shared_state['overspeed'] = (panneau_actif and CURRENT_SPEED > limit + OVERSPEED_MARGIN if limit else False)

            if not panneau_actif:
                shared_state['limit'] = None

        time.sleep(SIM_DT)

# ==================== Tkinter (Détection et affichage) ====================
def launch_tkinter_gui():
    window = tk.Tk()
    window.title("Détection(image| vidéo | webcam/IP)")
    window.geometry("700x400")
    window.configure(bg="#f8f8f8")

    header = tk.Label(window, text="Détection de panneaux",
                      bg="#f8f8f8", fg="#333", font=("Arial", 16, "bold"))
    header.pack(pady=10)

    info_frame = tk.Frame(window, bg="#f0f0f0")
    info_frame.pack(pady=5, fill="x")

    
    video_label = tk.Label(window, bg="#222")
    video_label.pack(pady=10)

    def set_limit_from_label(label: str):
        global TARGET_SPEED
        if "Speed Limit" in label:
            try:
                value = int(label.split()[-1])
                with state_lock:
                    shared_state['limit'] = value
                TARGET_SPEED = value
            except:
                pass

    def show_frame_on_label(img_bgr):
        annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(annotated_rgb).resize((300, 200))
        imgtk = ImageTk.PhotoImage(pil_img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    def detect_image():
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        image = cv2.imread(file_path)
        if image is None:
            print("Erreur : image invalide")
            return

        resized = resize_image(image)
        normalized = normalize_image(resized)
        img_u8 = (normalized * 255).astype(np.uint8)

        results = Valid_model.predict(source=img_u8, imgsz=640, conf=0.5, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = Valid_model.names[cls_id]
            set_limit_from_label(label)

        annotated = results[0].plot(line_width=1)
        show_frame_on_label(annotated)

    def detect_video():
        file_path = filedialog.askopenfilename(filetypes=[("Vidéos", "*.mp4 *.avi *.mov")])
        if not file_path:
            return

        def run_video():
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print("Erreur : vidéo non lisible")
                return

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized = resize_image(frame)
                normalized = normalize_image(resized)
                img_u8 = (normalized * 255).astype(np.uint8)

                results = Valid_model.predict(source=img_u8, imgsz=640, conf=0.5, verbose=False)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = Valid_model.names[cls_id]
                    set_limit_from_label(label)

                annotated = results[0].plot(line_width=1)
                show_frame_on_label(annotated)
                window.update()

            cap.release()

        threading.Thread(target=run_video, daemon=True).start()

    def detect_webcam():
      def run_webcam():
        # IP Webcam
        src = IP_WEBCAM_URL if IP_WEBCAM_URL else 0
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print("Erreur : webcam/flux non disponible")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while True:
            for _ in range(3):
                cap.grab()

            ret, frame = cap.read()
            if not ret:
                break

            resized = resize_image(frame)
            normalized = normalize_image(resized)
            img_u8 = (normalized * 255).astype(np.uint8)

            results = Valid_model.predict(
                source=img_u8,
                imgsz=640,
                conf=0.5,
                verbose=False
            )

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = Valid_model.names[cls_id]
                set_limit_from_label(label)

            annotated = results[0].plot(line_width=1)
            show_frame_on_label(annotated)
            window.update()

        cap.release()

      threading.Thread(target=run_webcam, daemon=True).start()


    btn_frame = tk.Frame(window, bg="#f8f8f8")
    btn_frame.pack()

    icon_camera = ImageTk.PhotoImage(Image.open("camera.png").resize((24, 24)))
    icon_video = ImageTk.PhotoImage(Image.open("video.png").resize((24, 24)))
    icon_webcam = ImageTk.PhotoImage(Image.open("webcam.png").resize((24, 24)))

    btn_image = tk.Button(btn_frame, image=icon_camera, command=detect_image,
                           fg="white", padx=12, pady=8)
    btn_image.grid(row=0, column=0, padx=12, pady=12)

    btn_video = tk.Button(btn_frame, image=icon_video, command=detect_video,
                           fg="white", padx=12, pady=8)
    btn_video.grid(row=0, column=2, padx=12, pady=12)

    btn_webcam = tk.Button(btn_frame, image=icon_webcam, command=detect_webcam,
                           fg="white", padx=12, pady=8)
    btn_webcam.grid(row=0, column=4, padx=12, pady=12)

    btn_image.image = icon_camera
    btn_video.image = icon_video
    btn_webcam.image = icon_webcam

    window.mainloop()

# ==================== Dashboard Pygame ====================
def launch_dashboard():
    pygame.init()
    init_beep()

    width, height = 700, 440
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ADAS (Simulation)")

    # Couleurs
    BLACK = (10, 10, 12)
    WHITE = (245, 245, 245)
    GRAY = (90, 90, 90)
    RED = (255, 60, 60)
    ORANGE = (255, 170, 60)
    BLUE = (60, 140, 255)
    DARK_GRAY = (35, 35, 40)
    LIGHT_BLUE = (160, 200, 255)
    GREEN = (60, 200, 120)
    YELLOW = (240, 220, 70)

    # Polices
    font_xs = pygame.font.SysFont('Arial', 16, bold=True)
    font_small = pygame.font.SysFont('Arial', 18, bold=True)
    font_medium = pygame.font.SysFont('Arial', 24, bold=True)
    font_large = pygame.font.SysFont('Arial', 36, bold=True)
    font_rpm = pygame.font.SysFont('Arial', 28, bold=True)
    font_big = pygame.font.SysFont('Arial', 64, bold=True)

    def draw_card(rect, title=None):
        pygame.draw.rect(screen, DARK_GRAY, rect, border_radius=16)
        pygame.draw.rect(screen, (60, 60, 65), rect, width=2, border_radius=16)
        if title:
            t = font_small.render(title, True, WHITE)
            screen.blit(t, (rect[0]+14, rect[1]+8))

    def speed_to_angle(speed, min_speed=0, max_speed=140, min_angle=30, max_angle=330):
        sp = max(min_speed, min(speed, max_speed))
        return min_angle + (sp - min_speed) / (max_speed - min_speed) * (max_angle - min_angle)

    def draw_speedometer(speed, limit):
        # Carte
        rect = (30, 40, 300, 320)
        draw_card(rect, "")
        center = (rect[0]+rect[2]//2, rect[1]+rect[3]//2+20)
        radius = 150

        # Graduation
        step = 10
        min_speed, max_speed = 0, 140
        angle_range, start_angle = 300, 30
        nb_ticks = ((max_speed - min_speed) // step) + 1
        for i in range(nb_ticks):
            v = min_speed + i*step
            ang = start_angle + i * (angle_range/(nb_ticks-1))
            rad = math.radians(ang)
            length = 16 if v % 20 == 0 else 10
            col = WHITE if v % 20 == 0 else GRAY
            start_pos = (center[0] + (radius-24) * math.cos(rad), center[1] + (radius-24) * math.sin(rad))
            end_pos   = (center[0] + (radius-length) * math.cos(rad), center[1] + (radius-length) * math.sin(rad))
            pygame.draw.line(screen, col, start_pos, end_pos, 2)
            if v % 20 == 0:
                txt = font_xs.render(str(v), True, WHITE)
                text_pos = (center[0] + (radius-44) * math.cos(rad), center[1] + (radius-44) * math.sin(rad))
                screen.blit(txt, (text_pos[0]-txt.get_width()/2, text_pos[1]-txt.get_height()/2))

        # Aiguille
        ang = speed_to_angle(speed)
        rad = math.radians(ang)
        needle_color = RED if speed > (limit + OVERSPEED_MARGIN) else WHITE
        needle_end = (center[0] + (radius-38) * math.cos(rad), center[1] + (radius-38) * math.sin(rad))
        pygame.draw.line(screen, needle_color, center, needle_end, 5)
        pygame.draw.circle(screen, (80, 80, 80), center, 7)

        # Valeur numérique
        speed_text = font_big.render(str(int(round(speed))), True, WHITE)
        kmh_text = font_medium.render("km/h", True, ORANGE)
        screen.blit(speed_text, (center[0] - speed_text.get_width()/2, center[1]- speed_text.get_height()/2 - 14))
        screen.blit(kmh_text, (center[0] - kmh_text.get_width()/2, center[1] + 28))


    def draw_speed_limit_sign(limit_value, pos):
        # Panneau 
        x, y = pos
        pygame.draw.circle(screen, WHITE, (x, y), 40)
        pygame.draw.circle(screen, RED, (x, y), 40, 10)
        txt = font_medium.render(str(int(limit_value)), True, (0,0,0))
        screen.blit(txt, (x - txt.get_width()//2, y - txt.get_height()//2))

    def draw_tachometer(rpm):
        rect = (370, 40, 300, 320)
        draw_card(rect, "")
        center = (rect[0]+rect[2]//2, rect[1]+rect[3]//2+10)
        radius = 90

        # Graduation
        for i in range(0, 9):  
            angle = 30 + i * (300/8)
            rad = math.radians(angle)
            length = 16 if i % 2 == 0 else 10
            start_pos = (center[0] + (radius-10)*math.cos(rad), center[1] + (radius-10)*math.sin(rad))
            end_pos   = (center[0] + (radius-length)*math.cos(rad), center[1] + (radius-length)*math.sin(rad))
            pygame.draw.line(screen, WHITE, start_pos, end_pos, 2)

            if i % 1 == 0:  
                tv = font_xs.render(str(i), True, WHITE)
                text_pos = (center[0] + (radius-28)*math.cos(rad), center[1] + (radius-28)*math.sin(rad))
                screen.blit(tv, (text_pos[0]-tv.get_width()/2, text_pos[1]-tv.get_height()/2))

        # Aiguille
        min_rpm, max_rpm = 0, 8
        min_angle, max_angle = 30, 330
        angle = min_angle + (rpm - min_rpm) / (max_rpm - min_rpm) * (max_angle - min_angle)
        rad = math.radians(angle)
        needle_end = (center[0] + (radius-26)*math.cos(rad), center[1] + (radius-26)*math.sin(rad))
        pygame.draw.line(screen, BLUE, center, needle_end, 4)
        pygame.draw.circle(screen, (80, 80, 80), center, 6)

        # Texte numérique
        rpm_text = font_rpm.render(f"{rpm:.1f}", True, LIGHT_BLUE)
        x1000_text = font_small.render("x1000", True, LIGHT_BLUE)
        screen.blit(rpm_text, (center[0]-rpm_text.get_width()/2, center[1]-rpm_text.get_height()/2-8))
        screen.blit(x1000_text, (center[0]-x1000_text.get_width()/2, center[1]+18))


    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with state_lock:
            v = shared_state['vehicle_speed']
            limit = shared_state['limit']
            mileage = shared_state['mileage']
            if v < 10:
                rpm = 1.0; gear = "1"
            elif v < 25:
                rpm = 1.8; gear = "2"
            elif v < 45:
                rpm = 2.5; gear = "3"
            elif v < 80:
                rpm = 3.2; gear = "4"
            elif v <120:
                rpm = 4.6; gear = "5"
            else:
                rpm= 5; gear="5"
            shared_state['rpm'] = rpm
            shared_state['gear'] = gear

            overspeed = shared_state['overspeed']

        screen.fill(BLACK)

        draw_speedometer(v, 30)
        draw_tachometer(rpm)
    

        # si excès, alerte
        if overspeed:
            draw_speed_limit_sign(limit, (350,400))
            play_beep()

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

    sys.exit()

# ==================== Lancement ====================
if __name__ == "__main__":
    threading.Thread(target=update_vehicle_speed, daemon=True).start()
    # Dashboard
    threading.Thread(target=launch_dashboard, daemon=True).start()
    # UI 
    launch_tkinter_gui()
