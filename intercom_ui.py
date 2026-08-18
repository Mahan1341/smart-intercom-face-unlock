import time

import cv2
import mss
import numpy as np
import pyautogui
import pygetwindow as gw


class IntercomUI:
    def __init__(self, window_name: str, button_template_path: str, button_threshold: float) -> None:
        self.window_name = window_name
        self.button_threshold = button_threshold
        self.screen_capture = mss.mss()

        template = cv2.imread(button_template_path)
        if template is None:
            raise FileNotFoundError(f"Button template not found: {button_template_path}")

        self.button_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.template_height, self.template_width = self.button_template.shape

    def get_window(self):
        for title in gw.getAllTitles():
            if self.window_name.lower() in title.lower():
                windows = gw.getWindowsWithTitle(title)
                if windows:
                    return windows[0]
        return None

    def capture(self, window):
        bbox = {
            "top": window.top,
            "left": window.left,
            "width": window.width,
            "height": window.height,
        }
        image = self.screen_capture.grab(bbox)
        frame = np.array(image)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), bbox

    def find_button(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = gray.shape

        if self.template_height > frame_height or self.template_width > frame_width:
            return None, 0.0

        result = cv2.matchTemplate(gray, self.button_template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(result)

        if score < self.button_threshold:
            return None, float(score)

        x, y = location
        center = (
            x + self.template_width // 2,
            y + self.template_height // 2,
        )
        return center, float(score)

    @staticmethod
    def click_button(position, window_bbox) -> None:
        x = window_bbox["left"] + position[0]
        y = window_bbox["top"] + position[1]

        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()
        time.sleep(0.05)
        pyautogui.mouseUp()
