import arcade
import numpy

from core.core import CameraFirstPerson, CameraThirdPerson


class PlayerController:
    def __init__(
        self,
        camera,
        shaders,
        colliders=None,
        terrain_bounds=None,
        ground_height_fn=None,
        terrain_contains_fn=None,
        always_play_walk=False,
    ) -> None:
        self.camera = camera
        self.shaders = shaders
        self.speed = 5.2
        self.look_sensitivity = 0.15
        self.keys_down = set()
        self.colliders = colliders or []
        self.terrain_bounds = terrain_bounds
        self.ground_height_fn = ground_height_fn
        self.terrain_contains_fn = terrain_contains_fn
        self.radius = 0.4
        self.eye_height = 1.7
        self.ground_height = 0.0

    def on_key_press(self, symbol):
        self.keys_down.add(symbol)

    def on_key_release(self, symbol):
        self.keys_down.discard(symbol)

    def on_mouse_motion(self, dx, dy):
        horizontal = -dx * self.look_sensitivity
        vertical = dy * self.look_sensitivity
        self.rotate(horizontal, vertical)

    def rotate(self, horizontal, vertical):
        self.camera.increment_direction(horizontal, vertical)

    def _sample_ground_height(self, x, y):
        if self.ground_height_fn is None:
            return 0.0
        return float(self.ground_height_fn(x, y))

    def _is_blocked(self, position, probe_height):
        if self.terrain_contains_fn is not None and not self.terrain_contains_fn(position[0], position[1], self.radius):
            return True

        if self.terrain_bounds is not None:
            terrain_min, terrain_max = self.terrain_bounds
            if position[0] - self.radius < terrain_min[0] or position[0] + self.radius > terrain_max[0]:
                return True
            if position[1] - self.radius < terrain_min[1] or position[1] + self.radius > terrain_max[1]:
                return True

        probe_z = self._sample_ground_height(position[0], position[1]) + probe_height
        for collider in self.colliders:
            if collider.collides_with_circle(position, self.radius, probe_z):
                return True

        return False


class PlayerFirstPerson(PlayerController):
    def __init__(
        self,
        camera: CameraFirstPerson,
        shaders,
        colliders=None,
        terrain_bounds=None,
        ground_height_fn=None,
        terrain_contains_fn=None,
        always_play_walk=False,
    ) -> None:
        super().__init__(
            camera,
            shaders,
            colliders=colliders,
            terrain_bounds=terrain_bounds,
            ground_height_fn=ground_height_fn,
            terrain_contains_fn=terrain_contains_fn,
        )
        self.speed = 6.0
        self.radius = 0.4
        self.eye_height = 1.7
        self.camera.position[2] = self.ground_height + self.eye_height

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
        if not self._is_blocked(next_position, self.eye_height):
            current_position[0] = next_position[0]

        next_position = current_position.copy()
        next_position[1] += delta_y
        if not self._is_blocked(next_position, self.eye_height):
            current_position[1] = next_position[1]

        self.camera.position = current_position


class PlayerThirdPerson(PlayerController):
    def __init__(
        self,
        camera: CameraThirdPerson,
        shaders,
        player_mesh,
        position,
        visual_meshes=None,
        animated_visual=None,
        mesh_rotation_offset=None,
        mesh_position_offset=None,
        mesh_heading_offset=-90.0,
        colliders=None,
        terrain_bounds=None,
        ground_height_fn=None,
        terrain_contains_fn=None,
        always_play_walk=False,
    ) -> None:
        super().__init__(
            camera,
            shaders,
            colliders=colliders,
            terrain_bounds=terrain_bounds,
            ground_height_fn=ground_height_fn,
            terrain_contains_fn=terrain_contains_fn,
        )
        self.player_mesh = player_mesh
        self.visual_meshes = visual_meshes or [player_mesh]
        self.animated_visual = animated_visual
        self.position = numpy.array(position, dtype=numpy.float32)
        self.mesh_rotation_offset = mesh_rotation_offset or (0.0, 0.0, 0.0)
        self.mesh_position_offset = numpy.array(mesh_position_offset or (0.0, 0.0, 0.0), dtype=numpy.float32)
        self.mesh_heading_offset = float(mesh_heading_offset)
        self.always_play_walk = bool(always_play_walk)
        self.speed = 1.8
        self.radius = 0.52
        self.character_height = 1.3
        self.look_sensitivity = 0.18
        self.facing_yaw = self.camera.theta
        self.target_facing_yaw = float(self.camera.theta)
        self.turn_speed = 220.0
        self.last_world_direction = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        self.active_animation_state = None
        self.camera.update_focus(self.position)
        self._sync_visuals()

    def update(self, delta_time):
        move_vector = self._get_move_vector()
        is_moving = numpy.linalg.norm(move_vector[:2]) > 1e-6
        self._update_facing(delta_time)
        if is_moving:
            move_scale = self._get_turn_movement_factor()
            self._move(move_vector * self.speed * delta_time * move_scale)
        self._update_animation(delta_time, is_moving)

        self.ground_height = self._sample_ground_height(self.position[0], self.position[1])
        self.position[2] = self.ground_height
        self._sync_visuals()
        self.camera.update_focus(self.position)
        self.camera.update(self.shaders)

    def _get_move_vector(self):
        input_x = 0.0
        input_y = 0.0
        if arcade.key.D in self.keys_down:
            input_x += 1.0
        if arcade.key.A in self.keys_down:
            input_x -= 1.0
        if arcade.key.W in self.keys_down:
            input_y += 1.0
        if arcade.key.S in self.keys_down:
            input_y -= 1.0

        if input_x == 0.0 and input_y == 0.0:
            return numpy.zeros(3, dtype=numpy.float32)

        movement = numpy.array([input_x, input_y], dtype=numpy.float32)
        movement /= max(float(numpy.linalg.norm(movement)), 1e-6)

        # Character facing is derived directly from camera heading plus input direction.
        # This keeps W aligned with the current camera forward.
        input_angle = numpy.degrees(numpy.arctan2(-movement[0], movement[1]))
        self.target_facing_yaw = float((self.camera.theta + input_angle) % 360.0)

        camera_theta = numpy.radians(self.camera.theta)
        camera_forward = numpy.array(
            [numpy.cos(camera_theta, dtype=numpy.float32), numpy.sin(camera_theta, dtype=numpy.float32)],
            dtype=numpy.float32,
        )
        camera_right = numpy.array([camera_forward[1], -camera_forward[0]], dtype=numpy.float32)
        world_direction = camera_forward * movement[1] + camera_right * movement[0]
        world_direction /= max(float(numpy.linalg.norm(world_direction)), 1e-6)
        return numpy.array([world_direction[0], world_direction[1], 0.0], dtype=numpy.float32)

    def _move(self, movement):
        current_position = self.position.copy()

        next_position = current_position.copy()
        next_position[0] += movement[0]
        if not self._is_blocked(next_position, self.character_height):
            current_position[0] = next_position[0]

        next_position = current_position.copy()
        next_position[1] += movement[1]
        if not self._is_blocked(next_position, self.character_height):
            current_position[1] = next_position[1]

        current_position[2] = self._sample_ground_height(current_position[0], current_position[1])
        self.position = current_position

    def _update_facing(self, delta_time):
        angle_delta = ((self.target_facing_yaw - self.facing_yaw + 180.0) % 360.0) - 180.0
        max_step = self.turn_speed * float(delta_time)
        if abs(angle_delta) <= max_step:
            self.facing_yaw = self.target_facing_yaw
            return
        self.facing_yaw = float((self.facing_yaw + numpy.sign(angle_delta) * max_step) % 360.0)

    def _get_turn_movement_factor(self):
        angle_delta = abs(((self.target_facing_yaw - self.facing_yaw + 180.0) % 360.0) - 180.0)
        if angle_delta <= 20.0:
            return 1.0
        if angle_delta >= 110.0:
            return 0.0
        return float((110.0 - angle_delta) / 90.0)

    def _update_animation(self, delta_time, is_moving):
        if self.animated_visual is None:
            return

        has_walk = self.animated_visual.has_animation("walk")
        has_idle = self.animated_visual.has_animation("idle")
        if not has_walk and not has_idle:
            return

        target_state = "walk" if (is_moving or self.always_play_walk) else "idle"
        if target_state == "walk" and not has_walk:
            target_state = "idle"

        if target_state == "idle":
            if has_idle:
                self.animated_visual.play(
                    "idle",
                    loop=True,
                    paused=False,
                    restart=self.active_animation_state != "idle",
                )
            elif has_walk:
                self.animated_visual.play(
                    "walk",
                    loop=True,
                    paused=True,
                    restart=self.active_animation_state != "idle",
                    hold_time=0.0,
                )
        elif target_state == "walk":
            self.animated_visual.play(
                "walk",
                loop=True,
                paused=False,
                restart=self.active_animation_state != "walk",
            )

        self.active_animation_state = target_state
        self.animated_visual.update(delta_time)

    def _sync_visuals(self):
        mesh_position = self.position + self.mesh_position_offset
        rotation_x, rotation_y, rotation_z = self.mesh_rotation_offset
        for mesh in self.visual_meshes:
            mesh.position[0] = float(mesh_position[0])
            mesh.position[1] = float(mesh_position[1])
            mesh.position[2] = float(mesh_position[2])
            mesh.set_rotation(
                x=rotation_x,
                y=rotation_y,
                z=rotation_z - self.facing_yaw + self.mesh_heading_offset,
            )
