import os
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

# ==================== Variables Partagées ====================
from threading import Lock

shared_state = {
    'speed': 0,
    'rpm': 0.0,
    'gear': 'N',
    'mileage': 14356
}
state_lock = Lock()

# ==================== Chargement du modèle YOLO ====================
Valid_model = YOLO("best.pt")  # Remplace par ton chemin exact si besoin

def normalize_image(image):
    return image / 255.0

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

# ==================== Tkinter pour détection ====================
def launch_tkinter_gui():
    window = tk.Tk()
    window.title("Détection avec YOLOv8 (image ou vidéo)")
    window.geometry("800x600")
    window.configure(bg="#f8f8f8")

    video_label = tk.Label(window, bg="#f8f8f8")
    video_label.pack()

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
        normalized_uint8 = (normalized * 255).astype(np.uint8)

        results = Valid_model.predict(source=normalized_uint8, imgsz=640, conf=0.5)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = Valid_model.names[cls_id]
            print("Label détecté :", label)

            if "Speed Limit" in label:
                try:
                    value = int(label.split()[-1])
                    with state_lock:
                        shared_state['speed'] = value
                except:
                    pass

        annotated = results[0].plot(line_width=1)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize((600, 400))
        imgtk = ImageTk.PhotoImage(pil_img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    def detect_video():
        file_path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov")])
        if not file_path:
            return

        def run_video():
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized = resize_image(frame)
                normalized = normalize_image(resized)
                normalized_uint8 = (normalized * 255).astype(np.uint8)

                results = Valid_model.predict(source=normalized_uint8, imgsz=640, conf=0.5)
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = Valid_model.names[cls_id]
                    if "Speed Limit" in label:
                        try:
                            value = int(label.split()[-1])
                            with state_lock:
                                shared_state['speed'] = value
                        except:
                            pass

                annotated = results[0].plot(line_width=1)
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((600, 400))
                imgtk = ImageTk.PhotoImage(pil_img)
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)
                window.update()

            cap.release()

        threading.Thread(target=run_video).start()


    def detect_webcam():
      
      def run_webcam():
        cap = cv2.VideoCapture(0)  # 0 = caméra par défaut
        if not cap.isOpened():
            print("Erreur : webcam non disponible")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized = resize_image(frame)
            normalized = normalize_image(resized)
            normalized_uint8 = (normalized * 255).astype(np.uint8)

            results = Valid_model.predict(source=normalized_uint8, imgsz=640, conf=0.5)

            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = Valid_model.names[cls_id]
                if "Speed Limit" in label:
                    try:
                        value = int(label.split()[-1])
                        with state_lock:
                            shared_state['speed'] = value
                        print("Détecté sur webcam :", label)
                    except:
                        pass

            annotated = results[0].plot(line_width=1)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((600, 400))
            imgtk = ImageTk.PhotoImage(pil_img)

            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            window.update()

        cap.release()

      threading.Thread(target=run_webcam).start()


    btn_image = tk.Button(window, text="📷 Image", command=detect_image,
                          bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
    btn_image.pack(pady=10)

    btn_video = tk.Button(window, text="🎥 Vidéo", command=detect_video,
                          bg="#2196F3", fg="white", font=("Arial", 12), padx=10, pady=5)
    btn_video.pack(pady=10)

    btn_webcam = tk.Button(window, text="📷 Webcam", command=detect_webcam,
                       bg="#FF5722", fg="white", font=("Arial", 12), padx=10, pady=5)
    btn_webcam.pack(pady=10)


    window.mainloop()

# ==================== Pygame : Dashboard ====================
def launch_dashboard():
    pygame.init()
    width, height = 800, 480
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Tableau de Bord Automobile")

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (80, 80, 80)
    RED = (255, 50, 50)
    ORANGE = (255, 150, 50)
    BLUE = (50, 50, 255)
    DARK_GRAY = (30, 30, 30)
    LIGHT_BLUE = (100, 100, 255)

    font_small = pygame.font.SysFont('Arial', 18, bold=True)
    font_medium = pygame.font.SysFont('Arial', 24, bold=True)
    font_large = pygame.font.SysFont('Arial', 36, bold=True)
    font_rpm = pygame.font.SysFont('Arial', 28, bold=True)



    def speed_to_angle(speed):
       min_speed = 10
       max_speed = 120
       min_angle = 30
       max_angle = 330
       speed = max(min_speed, min(speed, max_speed))  # clamp
       angle_range = max_angle - min_angle
       return min_angle + ((speed - min_speed) / (max_speed - min_speed)) * angle_range

    def draw_speedometer(speed):
        center = (width//3, height//2)
        radius = 120
        pygame.draw.circle(screen, GRAY, center, radius, 3)
        pygame.draw.circle(screen, DARK_GRAY, center, radius-3, 2)

        step = 10
        min_speed = 10
        max_speed = 120
        angle_range = 300
        start_angle = 30

        nb_ticks = ((max_speed - min_speed) // step) + 1

        for i in range(nb_ticks):
            speed_value = min_speed + i * step
            angle = start_angle + (i * (angle_range / (nb_ticks - 1)))
            rad = math.radians(angle)

            if speed_value % 20 == 0:
                length = 15
                text = font_small.render(str(speed_value), True, WHITE)
                text_pos = (center[0] + (radius-25) * math.cos(rad),
                    center[1] + (radius-25) * math.sin(rad))
                screen.blit(text, (text_pos[0] - text.get_width()/2,
                           text_pos[1] - text.get_height()/2))
            else:
                length = 10

            start_pos = (center[0] + (radius-10) * math.cos(rad),
                 center[1] + (radius-10) * math.sin(rad))
            end_pos = (center[0] + (radius-length) * math.cos(rad),
               center[1] + (radius-length) * math.sin(rad))
            pygame.draw.line(screen, WHITE, start_pos, end_pos, 2)

        
        
        angle = speed_to_angle(speed)
        rad = math.radians(angle)
        needle_end = (center[0] + (radius-30) * math.cos(rad),
                      center[1] + (radius-30) * math.sin(rad))
        pygame.draw.line(screen, RED, center, needle_end, 4)
        pygame.draw.circle(screen, DARK_GRAY, center, 5)

        speed_text = font_large.render(str(speed), True, WHITE)
        kmh_text = font_medium.render("km/h", True, ORANGE)
        screen.blit(speed_text, (center[0] - speed_text.get_width()/2,
                                 center[1] - speed_text.get_height()/2 - 10))
        screen.blit(kmh_text, (center[0] - kmh_text.get_width()/2,
                               center[1] + 20))
   



    def draw_tachometer(rpm):
        center = (2*width//3, height//2)
        radius = 120
        pygame.draw.circle(screen, GRAY, center, radius, 3)
        pygame.draw.circle(screen, DARK_GRAY, center, radius-3, 2)

        for i, angle in enumerate(range(30, 331, 30)):
            rad = math.radians(angle)
            if i % 2 == 0:
                length = 15
                value = str(i // 2)
                text = font_small.render(value, True, WHITE)
                text_pos = (center[0] + (radius-25) * math.cos(rad),
                            center[1] + (radius-25) * math.sin(rad))
                screen.blit(text, (text_pos[0] - text.get_width()/2,
                                   text_pos[1] - text.get_height()/2))
            else:
                length = 10
            start_pos = (center[0] + (radius-10) * math.cos(rad),
                         center[1] + (radius-10) * math.sin(rad))
            end_pos = (center[0] + (radius-length) * math.cos(rad),
                       center[1] + (radius-length) * math.sin(rad))
            pygame.draw.line(screen, WHITE, start_pos, end_pos, 2)

        angle = 30 + (rpm / 8.0) * 300
        rad = math.radians(angle)
        needle_end = (center[0] + (radius-30) * math.cos(rad),
                      center[1] + (radius-30) * math.sin(rad))
        pygame.draw.line(screen, RED, center, needle_end, 4)
        pygame.draw.circle(screen, DARK_GRAY, center, 5)

        rpm_text = font_rpm.render(f"{rpm:.1f}", True, LIGHT_BLUE)
        x1000_text = font_medium.render("x1000", True, LIGHT_BLUE)
        rpm_label = font_medium.render("RPM", True, LIGHT_BLUE)

        screen.blit(rpm_text, (center[0] - rpm_text.get_width()/2,
                               center[1] - rpm_text.get_height()/2 - 15))
        screen.blit(x1000_text, (center[0] - x1000_text.get_width()/2,
                                 center[1] + 5))
        screen.blit(rpm_label, (center[0] - rpm_label.get_width()/2,
                                center[1] + 35))

    def draw_gear_display(gear):
        gear_text = font_large.render(gear, True, WHITE)
        pygame.draw.circle(screen, GRAY, (width//2, height//2), 40, 3)
        screen.blit(gear_text, (width//2 - gear_text.get_width()/2,
                                height//2 - gear_text.get_height()/2))

    def draw_mileage(mileage):
        mileage_text = font_medium.render(f"{mileage} km", True, WHITE)
        screen.blit(mileage_text, (width//2 - mileage_text.get_width()/2,
                                   height - 50))

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with state_lock:
            speed = shared_state['speed']
            mileage = shared_state['mileage']

            # Mise à jour automatique rpm/gear selon vitesse
            if speed < 20:
                rpm = 1.2
                gear = "1"
            elif speed < 40:
                rpm = 2.0
                gear = "2"
            elif speed < 60:
                rpm = 2.8
                gear = "3"
            elif speed < 100:
                rpm = 3.5
                gear = "4"
            else:
                rpm = 4.2
                gear = "5"
            shared_state['rpm'] = rpm
            shared_state['gear'] = gear

        screen.fill(BLACK)
        draw_speedometer(speed)
        draw_tachometer(rpm)
        draw_gear_display(gear)
        draw_mileage(mileage)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

# ==================== Lancer les deux interfaces ====================
if __name__ == "__main__":
    threading.Thread(target=launch_dashboard, daemon=True).start()
    launch_tkinter_gui()
