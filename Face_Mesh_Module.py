import cv2
import mediapipe as mp
import time


class Face_Mesh_Detector:
    def __init__(self, static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        # If there is more than 1 face, then change the argument of FaceMesh class.
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode, max_num_faces, refine_landmarks,
                                                    min_detection_confidence, min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []

        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACEMESH_TESSELATION,
                                                self.draw_spec, self.draw_spec)
                face = []

                for l_id, lm in enumerate(face_lms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.putText(img, str(l_id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                                0.8, (0, 255, 0), 1)
                    # print(l_id, (x, y))
                    face.append([x, y])

                faces.append(face)

        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = Face_Mesh_Detector()

    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img, draw=False)

        while not success:
            print('Video ended.')
            break

        if len(faces) != 0:
            print(faces[0])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (200, 200, 100), 2)

        cv2.imshow("Video", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

