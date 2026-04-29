import arcade
import numpy

from core.core import CameraFirstPerson


class PlayerFirstPerson:
    def __init__(self, camera: CameraFirstPerson, shaders, colliders=None, terrain_bounds=None, ground_height_fn=None) -> None:
        self.camera = camera
        self.shaders = shaders
        self.speed = 6
        self.look_sensitivity = 0.15
        self.keys_down = set()
        self.colliders = colliders or []
        self.terrain_bounds = terrain_bounds
        self.ground_height_fn = ground_height_fn
        self.radius = 0.35
        self.eye_height = 1.7
        self.ground_height = 0.0
        self.camera.position[2] = self.ground_height + self.eye_height

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

        self.ground_height = self._sample_ground_height(self.camera.position[0], self.camera.position[1])
        self.camera.position[2] = self.ground_height + self.eye_height
        self.camera.update(self.shaders)

    def move(self, direction, amount):
        walk_direction = numpy.radians((direction + self.camera.theta) % 360)
        delta_x = amount * self.camera.move_speed * numpy.cos(walk_direction, dtype=numpy.float32)
        delta_y = amount * self.camera.move_speed * numpy.sin(walk_direction, dtype=numpy.float32)

        current_position = self.camera.position.copy()

        next_position = current_position.copy()
        next_position[0] += delta_x
        if not self._is_blocked(next_position):
            current_position[0] = next_position[0]

        next_position = current_position.copy()
        next_position[1] += delta_y
        if not self._is_blocked(next_position):
            current_position[1] = next_position[1]

        self.camera.position = current_position

    def rotate(self, horizontal, vertical):
        self.camera.increment_direction(horizontal, vertical)

    def _sample_ground_height(self, x, y):
        if self.ground_height_fn is None:
            return 0.0
        return float(self.ground_height_fn(x, y))

    def _is_blocked(self, position):
        if self.terrain_bounds is not None:
            terrain_min, terrain_max = self.terrain_bounds
            if position[0] - self.radius < terrain_min[0] or position[0] + self.radius > terrain_max[0]:
                return True
            if position[1] - self.radius < terrain_min[1] or position[1] + self.radius > terrain_max[1]:
                return True

        for collider in self.colliders:
            bounds_min, bounds_max = collider.get_world_bounds()

            if position[2] < bounds_min[2] or position[2] > bounds_max[2] + 2.0:
                continue

            closest_x = min(max(position[0], bounds_min[0]), bounds_max[0])
            closest_y = min(max(position[1], bounds_min[1]), bounds_max[1])
            delta_x = position[0] - closest_x
            delta_y = position[1] - closest_y

            if delta_x * delta_x + delta_y * delta_y < self.radius * self.radius:
                return True

        return False
