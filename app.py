from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.metrics import dp
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
from ultralytics import YOLO
# Placeholder function for license plate detection using YOLOv3 model
def detect_license_plate(frame, model):
    results = model.track(frame, persist=True)
    annotated_frame = results[0].plot()

    return annotated_frame


class LicensePlateScreen(BoxLayout):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.model = model_path
        # Header
        header = Label(text='Scan License Plate', size_hint_y=None, height=dp(50))
        self.add_widget(header)

        # License Plate Image
        self.license_plate_image = Image(size_hint=(1, 0.4))
        self.add_widget(self.license_plate_image)

        # Information Section
        info_layout = GridLayout(cols=2, size_hint_y=None, height=dp(80), padding=dp(10))
        info_layout.add_widget(Label(text='License Plate'))
        info_layout.add_widget(Label(text='GLN 8988', halign='right'))
        self.add_widget(info_layout)

        # Confirmation Button
        confirm_button = Button(text='CONFIRM', size_hint_y=None, height=dp(50))
        self.add_widget(confirm_button)

        # Open the camera
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

    def update(self, dt):
        # Read a frame from the camera
        ret, frame = self.capture.read()
        if ret:
            # Perform license plate detection using YOLOv3
            detected_frame = detect_license_plate(frame, self.model)

            # Flip the frame vertically
            detected_frame = cv2.flip(detected_frame, 0)

            # Convert the frame from BGR to RGB
            detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to texture
            texture = Texture.create(size=(detected_frame.shape[1], detected_frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(detected_frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

            # Assign the texture to the Kivy Image widget
            self.license_plate_image.texture = texture

    def on_stop(self):
        # Release the camera
        self.capture.release()

class LicensePlateApp(App):
    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        super().__init__(**kwargs)

    def build(self):
        self.title = "SnapPlate"
        return LicensePlateScreen(self.model_path)

if __name__ == '__main__':
    # Provide the path to your trained YOLOv3 model
    model_path = "license_plate_detector.pt"
    model = YOLO(model_path)
    LicensePlateApp(model).run()
