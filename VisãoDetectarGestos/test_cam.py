import cv2
import time
import numpy as np
import mediapipe as mp
import frl_core  # Importa o SEU arquivo

mp_pose = mp.solutions.pose
# model_complexity=0 é o modelo mais leve/rápido (ideal para Raspberry Pi depois)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
mp_drawing = mp.solutions.drawing_utils

# Mapeamento: Índices do MediaPipe para o padrão COCO-17 usado no frl_core.py
MAP_MP_TO_COCO = {11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10}

def main():
    # Mude o número da porta aqui se necessário (ex: 0 para a padrão, 1 ou 2 para a outra webcam)
    CAM_PORT = 1 
    
    cap = cv2.VideoCapture(CAM_PORT)
    
    # Reduz a resolução para simular a câmera ruim do Rasp e ganhar FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Inicializa as classes do seu núcleo
    debouncer = frl_core.Debouncer()
    mover = frl_core.Mover()

    last_time = time.time()

    # Controle de FPS alvo (15 FPS = ~0.066 segundos por frame)
    TARGET_FPS = 15
    FRAME_DURATION = 1.0 / TARGET_FPS

    print(f"=== INICIANDO SISTEMA DE GESTOS (Porta: {CAM_PORT}, Alvo: {TARGET_FPS} FPS) ===")
    print("Fique a ~1.5m da câmera. Pressione 'q' na janela do vídeo para sair.")

    while cap.isOpened():
        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame da câmera. Verifique a porta.")
            break
        
        # Espelha o frame para não bugar o cérebro humano (como um espelho)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Converte BGR (OpenCV) para RGB (MediaPipe)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a pose
        results = pose.process(image_rgb)

        # Tempo delta (dt) para o Mover calcular a física
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        gesto_bruto = "NENHUM"
        comando_ativo = "HOVER"
        pwm = (1500, 1500, 1500, 1500) # (roll, pitch, thr, yawp)

        # Cria arrays vazios no formato (17,2) e (17,) que o seu código espera
        kpts = np.zeros((17, 2))
        confs = np.zeros(17)

        if results.pose_landmarks:
            # Desenha os keypoints do MediaPipe na imagem para você visualizar
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extrai e converte os keypoints do braço
            for mp_idx, coco_idx in MAP_MP_TO_COCO.items():
                landmark = results.pose_landmarks.landmark[mp_idx]
                kpts[coco_idx] = [landmark.x * w, landmark.y * h]
                confs[coco_idx] = landmark.visibility

            # 1. Classifica os braços no seu código
            gesto_bruto, estados_bracos, angulos = frl_core.classify(kpts, confs)
            
            # 2. Passa pelo filtro de segurança (Debouncer)
            comando_ativo, mudou = debouncer.update(gesto_bruto, current_time)
            
            # 3. Calcula movimento (Mover)
            cmd_vetor = frl_core.COMMANDS.get(comando_ativo, (0, 0, 0))
            pwm = mover.step(cmd_vetor, dt)

            if max(confs) > 0.3: 
                print(f"Gesto Lido: {gesto_bruto:10} | Comandando: {comando_ativo:10} | PWM (R, P, T, Y): {pwm}")
        else:
            comando_ativo, _ = debouncer.update("NENHUM", current_time)
            pwm = mover.step((0,0,0), dt)

        cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Detec: {gesto_bruto}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, f"COMANDO: {comando_ativo}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if comando_ativo != "HOVER" else (0, 165, 255), 3)
        
        cv2.putText(frame, f"PWM (Roll, Pitch, Thr, Yaw): {pwm}", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Mostra a janela
        cv2.imshow('Drone Gesture Test', frame)

        # Calcula quanto tempo gastou processando este frame e dorme o excedente para travar em 15 FPS
        elapsed = time.time() - loop_start
        if elapsed < FRAME_DURATION:
            time.sleep(FRAME_DURATION - elapsed)

        # Sai se apertar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
