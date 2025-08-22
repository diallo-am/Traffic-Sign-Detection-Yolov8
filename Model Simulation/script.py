import os
import cv2
import numpy as np
import threading
import pygame
import math
import sys
import time
from threading import Lock
import random
from ultralytics import YOLO

# ==================== CONFIG GÉNÉRALE ====================
# Adresse IP Webcam (ex: appli "IP Webcam" sur Android)
# Exemple : "http://192.168.1.101:8080/video"   ou laisser "" pour /dev/video0
IP_WEBCAM_URL = "http://192.168.1.101:8080/video"

# Latence (s) entre la nouvelle limite détectée et l'ajustement de la vitesse véhicule
LATENCE = 0.5

# Pas d'accélération/décélération (km/h) à chaque update de la vitesse
SPEED_STEP = 2

# Fréquence de mise à jour de la simulation (s)
SIM_DT = 0.2

# Seuil d’alerte (si vitesse véhicule > limite + marge)
OVERSPEED_MARGIN = 3  # km/h
OVERSPEED_BEEP_COOLDOWN = 1.5  # secondes entre deux bips

# Timeout (s) après disparition du panneau pour garder la limite affichée/active
LIMIT_TIMEOUT = 5

# ==================== ÉTAT PARTAGÉ ====================
shared_state = {
    'limit': None,           # limite détectée (km/h)
    'vehicle_speed': 30,     # vitesse véhicule simulée (km/h)
    'rpm': 0.0,
    'gear': 'N',
    'mileage': 14356,
    'overspeed': False,
}
state_lock = Lock()

# Variables internes pour la simulation vitesse
TARGET_SPEED = 30         # cible vers laquelle le véhicule va converger (suivant limite)
CURRENT_SPEED = 30        # vitesse actuelle simulée
last_limit_time = 0       # dernière détection
last_beep_time = 0        # anti-spam pour le bip
beep_sound = None         # son d'alerte (pygame)

# ==================== YOLO ====================
Valid_model = YOLO("best.pt")  # Chemin du modèle

def normalize_image(image):
    return image / 255.0

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# ==================== AUDIO (bip) ====================
def init_beep():
    """Initialise pygame.mixer et génère un petit bip (sinus) en mémoire."""
    global beep_sound
    try:
        pygame.mixer.init()
        sr = 22050
        dur = 0.12
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        wave = (0.5*np.sin(2*np.pi*700*t)).astype(np.float32)
        beep_sound = pygame.mixer.Sound(wave)
    except Exception as e:
        print("[AUDIO] Impossible d'initialiser le son (pas bloquant) :", e)
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

# ==================== SIMULATION VITESSE ====================
def update_vehicle_speed():
    """
    Simule la vitesse véhicule :
    - Sans panneau récent → variations aléatoires
    - Avec panneau → ralentir si overspeed
    """
    global CURRENT_SPEED, TARGET_SPEED, last_limit_time

    while True:
        time.sleep(LATENCE)

        with state_lock:
            limit = shared_state.get('limit', None)

            # Si on a une limite active, last_limit_time est mis à jour par le thread cam
            # Temps écoulé depuis le dernier panneau
            time_since_limit = time.time() - last_limit_time
            panneau_actif = (limit is not None) and (time_since_limit <= LIMIT_TIMEOUT)

            if not panneau_actif:
                # Aucun panneau actif → vitesse libre (petites variations)
                change = random.uniform(-4, 8)
                TARGET_SPEED = max(0, min(140, CURRENT_SPEED + change))
            else:
                # Panneau actif → contrôle vitesse
                if CURRENT_SPEED > limit + OVERSPEED_MARGIN:
                    TARGET_SPEED = max(0, limit - 1)  # descendre sous la limite
                else:
                    change = random.uniform(-4, 8)
                    TARGET_SPEED = max(0, min(limit, CURRENT_SPEED + change))

            # Convergence progressive
            if CURRENT_SPEED < TARGET_SPEED:
                CURRENT_SPEED = min(CURRENT_SPEED + SPEED_STEP, TARGET_SPEED)
            elif CURRENT_SPEED > TARGET_SPEED:
                CURRENT_SPEED = max(CURRENT_SPEED - SPEED_STEP, TARGET_SPEED)

            # Publication
            shared_state['vehicle_speed'] = int(round(CURRENT_SPEED))
            shared_state['overspeed'] = (panneau_actif and (CURRENT_SPEED > limit + OVERSPEED_MARGIN) if limit else False)

            # Après timeout, on efface la limite pour l'affichage
            if not panneau_actif:
                shared_state['limit'] = None

        time.sleep(SIM_DT)

# ==================== DÉTECTION CAM (IP Webcam / USB) ====================
def set_limit_from_label(label: str):
    """Extrait la limite depuis un label YOLO 'Speed Limit XX' et met à jour TARGET_SPEED + shared_state['limit']."""
    global TARGET_SPEED, last_limit_time
    if "Speed" in label and "Limit" in label:
        try:
            value = int(label.split()[-1])
            with state_lock:
                shared_state['limit'] = value
                last_limit_time = time.time()
            TARGET_SPEED = value
        except:
            pass

def camera_loop():
    """
    Thread de capture + détection.
    Se connecte à IP_WEBCAM_URL si défini, sinon /dev/video0.
    Réduit le buffering, skippe quelques frames pour diminuer la latence.
    """
    src = IP_WEBCAM_URL if IP_WEBCAM_URL else 0
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print("[CAM] Impossible d'ouvrir la webcam/flux :", src)
        # Tentative fallback sur /dev/video0 si URL invalide
        if src != 0:
            print("[CAM] Tentative fallback sur /dev/video0 ...")
            cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[CAM] Échec d'ouverture de toute source vidéo. Sortie du thread cam.")
        return

    # Réduit le buffering pour avoir moins de décalage (effet selon drivers/codec)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except:
        pass

    while True:
        # Jette quelques frames pour garder la plus récente (utile en IP)
        for _ in range(3):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Prétraitement léger
        resized = resize_image(frame, (640, 640))
        normalized = normalize_image(resized)
        img_u8 = (normalized * 255).astype(np.uint8)

        # Inference YOLO
        try:
            results = Valid_model.predict(
                source=img_u8,
                imgsz=640,
                conf=0.5,
                verbose=False
            )
            if len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = Valid_model.names[cls_id]
                    set_limit_from_label(label)
        except Exception as e:
            # En cas d'erreur d'inférence, on continue
            print("[YOLO] Erreur inference :", e)

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

    def draw_speedometer(speed, limit_display):
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
        needle_color = RED if (limit_display is not None and speed > (limit_display + OVERSPEED_MARGIN)) else WHITE
        needle_end = (center[0] + (radius-38) * math.cos(rad), center[1] + (radius-38) * math.sin(rad))
        pygame.draw.line(screen, needle_color, center, needle_end, 5)
        pygame.draw.circle(screen, (80, 80, 80), center, 7)

        # Valeur numérique
        speed_text = font_big.render(str(int(round(speed))), True, WHITE)
        kmh_text = font_medium.render("km/h", True, ORANGE)
        screen.blit(speed_text, (center[0] - speed_text.get_width()/2, center[1]- speed_text.get_height()/2 - 14))
        screen.blit(kmh_text, (center[0] - kmh_text.get_width()/2, center[1] + 28))

    def draw_speed_limit_sign(limit_value, pos):
        # Panneau rond type EU : cerclage rouge + fond blanc + nombre noir
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
        for i in range(0, 9):  # de 0 à 8
            angle = 30 + i * (300/8)
            rad = math.radians(angle)
            length = 16 if i % 2 == 0 else 10
            start_pos = (center[0] + (radius-10)*math.cos(rad), center[1] + (radius-10)*math.sin(rad))
            end_pos   = (center[0] + (radius-length)*math.cos(rad), center[1] + (radius-length)*math.sin(rad))
            pygame.draw.line(screen, WHITE, start_pos, end_pos, 2)

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

    def draw_gear_and_mileage(gear, mileage):
        rect = (470, 270, 400, 150)
        draw_card(rect, "Boîte & Odomètre")
        pygame.draw.circle(screen, (70, 70, 75), (rect[0]+80, rect[1]+90), 46, 0)
        pygame.draw.circle(screen, (110, 110, 120), (rect[0]+80, rect[1]+90), 46, 3)
        gtxt = font_large.render(gear, True, WHITE)
        screen.blit(gtxt, (rect[0]+80-gtxt.get_width()/2, rect[1]+90-gtxt.get_height()/2))
        mtxt = font_medium.render(f"{mileage} km", True, YELLOW)
        screen.blit(mtxt, (rect[0]+170, rect[1]+80 - mtxt.get_height()/2))

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        with state_lock:
            v = shared_state['vehicle_speed']
            limit = shared_state['limit']
            mileage = shared_state['mileage']
            # RPM/gear simulés en fonction de la vitesse
            if v < 10:
                rpm = 1.0; gear = "1"
            elif v < 25:
                rpm = 1.8; gear = "2"
            elif v < 45:
                rpm = 2.5; gear = "3"
            elif v < 80:
                rpm = 3.2; gear = "4"
            elif v < 120:
                rpm = 4.6; gear = "5"
            else:
                rpm = 5.0; gear = "5"
            shared_state['rpm'] = rpm
            shared_state['gear'] = gear
            overspeed = shared_state['overspeed']

        screen.fill(BLACK)

        # Dessins
        draw_speedometer(v, limit if limit is not None else 30)
        draw_tachometer(rpm)
        # draw_gear_and_mileage(gear, mileage)  # à activer si besoin

        # Bandeau d'alerte si excès
        if overspeed and (limit is not None):
            draw_speed_limit_sign(limit, (350, 400))
            play_beep()

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

# ==================== Lancement ====================
if __name__ == "__main__":
    # Thread simulation vitesse (converge vers la limite)
    threading.Thread(target=update_vehicle_speed, daemon=True).start()
    # Thread caméra + YOLO (IP Webcam/USB)
    threading.Thread(target=camera_loop, daemon=True).start()
    # Dashboard en thread principal
    launch_dashboard()
