import arcade

from core.core import CameraFirstPerson


class PlayerFirstPerson:
    def __init__(self, camera: CameraFirstPerson, shaders) -> None:
        self.camera = camera
        self.shaders = shaders
        self.speed = 15
        self.look_sensitivity = 0.15
        self.keys_down = set()

    def on_key_press(self, symbol):
        self.keys_down.add(symbol)

    def on_key_release(self, symbol):
        self.keys_down.discard(symbol)

    def on_mouse_motion(self, dx, dy):
        horizontal = -dx * self.look_sensitivity
        vertical = dy * self.look_sensitivity
        self.rotate(horizontal, vertical)

    def update(self, delta_time):
        if arcade.key.W in self.keys_down:
            self.move(0, self.speed * delta_time)
        if arcade.key.A in self.keys_down:
            self.move(90, self.speed * delta_time)
        if arcade.key.S in self.keys_down:
            self.move(180, self.speed * delta_time)
        if arcade.key.D in self.keys_down:
            self.move(-90, self.speed * delta_time)

        self.camera.update(self.shaders)

    def move(self, direction, amount):
        self.camera.move(direction, amount)

    def rotate(self, horizontal, vertical):
        self.camera.increment_direction(horizontal, vertical)
