import cv2
from deepface import DeepFace
import time
from statistics import mode

#Incepe captura video
cap = cv2.VideoCapture(0)

#Variabile de optimizare si afisare
frame_counter = 0
ANALYSIS_INTERVAL = 10 #Analizeaza odata 10 cadre

TARGET_WIDTH = 640 #vom analiza pe o imagine de 640px

#variabile pentru stabilizarea emotiei
EMOTION_HISTORY_SIZE = 15
emotion_history = ["neutral"] * EMOTION_HISTORY_SIZE
stable_emotion = "neutral"

#variabila de a calcula fps
prev_time = time.time()


current_faces = [] #Lista cu fetele detectate

#Dictionar de culori
emotion_colors = {
    "angry" : (0,0,225), #rosu
    "happy" : (0,255,255), #galben
    "sad" : (255,0,0), #albastru
    "surprize" : (0,165,255), #portocaliu
    "neutral" : (255,255,255) #alb
}

while True:
    ret, frame = cap.read() #captura cadru cu cadru
    if not ret:
        break

    #calculam si afisam fps
    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time
    fps = 1/delta_time if delta_time>0 else 0

    frame_counter +=1

    #blocul de analiza

    if frame_counter % ANALYSIS_INTERVAL == 0:
        try:
            #redimensionam cadrul pentru analiza(ca sa fie mai rapida)
            scale = TARGET_WIDTH / frame.shape[1]
            dim = (TARGET_WIDTH, int(frame.shape[0] * scale))
            analysis_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) #inter_area pentru cea mai buna calitate la micsorare

            #convertim cadrul micsorat in RGB
            analysis_frame_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)

            results = DeepFace.analyze(
                analysis_frame_rgb,
                actions = ['emotion'],
                detector_backend = 'ssd',
                enforce_detection = False #nu se opreste programul daca nu se gaseste fata
            )
            #results contine coordonatele pentru cadrul mic, trebuie marit la loc pentru a fi afisat in cadrul mare
            scale_up = frame.shape[1] / TARGET_WIDTH

            current_faces = [] #golim lista veche si o umplemcu noile rezultate

            #stabilizarea emotiei
            if results:
                results.sort(key=lambda x: x['region']['w'] * x['region']['h'], reverse=True)
                main_face = results[0]

                emotion = main_face['dominant_emotion']

                #actualizam istoricul
                emotion_history.append(emotion)
                emotion_history.pop(0) #scoatem cea mai veche emotie

                #gasim emotia stabila
                try:
                    stable_emotion = mode(emotion_history)
                except Exception as e:
                    stable_emotion = emotion_history[-1]

            #mapam toate fetele
            for face_info in results:
                region = face_info['region']
                score = face_info['emotion'][face_info['dominant_emotion']]

                #scalam coordonatele la dimensiunea din original
                x = int(region['x'] * scale_up) 
                y = int(region['y'] * scale_up)
                w = int(region['w'] * scale_up)
                h = int(region['h'] * scale_up)

                if face_info == main_face: #daca e fata principala folosim emotia stabila
                    current_faces.append(   ((x, y, w, h), stable_emotion, score)   )
                    #daca nu, folosim emotia detectata
                else:
                    current_faces.append(   ((x, y, w, h), face_info['dominant_emotion'], score)   )

        except Exception as e:
            print(f"Eroare la analiza DeepFace: {e}")
            pass
    
    #afisam fps urile pe ecran
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    for (region_coords, emotion, score) in current_faces:
        #extragem coordonatele(i format x,y,w,h)
        x, y, w, h = region_coords

        #preluam culoarea
        color = emotion_colors.get(emotion , (0,0,225)) #default rosu

        #formatam textul
        text_to_display = f"{emotion} ({score:.0f}%)"

        #dreptunghiul din jurul fetei
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)

        #scriem textul cu emotia
        cv2.putText(frame, text_to_display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    #afisam rezultatele
    cv2.imshow('Real-time Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()      


