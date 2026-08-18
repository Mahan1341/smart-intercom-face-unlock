import time

import cv2

from config import (
    AUTHORIZED_FRAMES_REQUIRED,
    BUTTON_TEMPLATE_PATH,
    BUTTON_THRESHOLD,
    FACE_THRESHOLD,
    GLOBAL_COOLDOWN_SECONDS,
    MODEL_PATH,
    REFERENCE_IMAGE_PATH,
    WINDOW_NAME,
)
from face_recognition import FaceRecognizer
from intercom_ui import IntercomUI


def draw_face_status(frame, bbox, similarity, authorized):
    if bbox is None or similarity is None:
        return

    x, y, w, h = bbox
    color = (0, 255, 0) if authorized else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        f"similarity: {similarity:.2f}",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def main() -> None:
    recognizer = FaceRecognizer(
        model_path=str(MODEL_PATH),
        reference_image_path=str(REFERENCE_IMAGE_PATH),
        threshold=FACE_THRESHOLD,
    )
    intercom = IntercomUI(
        window_name=WINDOW_NAME,
        button_template_path=str(BUTTON_TEMPLATE_PATH),
        button_threshold=BUTTON_THRESHOLD,
    )

    last_action_time = 0.0
    consecutive_authorized_frames = 0

    while True:
        window = intercom.get_window()
        if window is None:
            print(f"Window containing '{WINDOW_NAME}' was not found")
            time.sleep(1)
            continue

        frame, window_bbox = intercom.capture(window)
        authorized, similarity, face_bbox = recognizer.compare(frame)

        if authorized:
            consecutive_authorized_frames += 1
        else:
            consecutive_authorized_frames = 0

        draw_face_status(frame, face_bbox, similarity, authorized)

        button_position, button_score = intercom.find_button(frame)
        if button_position is not None:
            cv2.circle(frame, button_position, 8, (255, 0, 0), -1)

        authorization_confirmed = (
            consecutive_authorized_frames >= AUTHORIZED_FRAMES_REQUIRED
        )
        cooldown_finished = (
            time.time() - last_action_time >= GLOBAL_COOLDOWN_SECONDS
        )

        if authorization_confirmed and cooldown_finished and button_position is not None:
            intercom.click_button(button_position, window_bbox)
            last_action_time = time.time()
            consecutive_authorized_frames = 0
            print(
                "Door action triggered "
                f"(face={similarity:.2f}, button={button_score:.2f})"
            )

        cv2.imshow("Smart Intercom Face Unlock", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
