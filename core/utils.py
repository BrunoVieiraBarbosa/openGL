from contextlib import nullcontext

import arcade
import numpy

from core.core import CameraFirstPerson, CameraThirdPerson

DIAGONAL_NORMALIZER = numpy.float32(0.70710677)


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
        profiler=None,
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
        self.profiler = profiler
        self._key_w = arcade.key.W
        self._key_a = arcade.key.A
        self._key_s = arcade.key.S
        self._key_d = arcade.key.D
        self._key_lshift = arcade.key.LSHIFT
        self._key_rshift = arcade.key.RSHIFT

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

    def _profile_section(self, name):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.section(name)


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
        profiler=None,
    ) -> None:
        super().__init__(
            camera,
            shaders,
            colliders=colliders,
            terrain_bounds=terrain_bounds,
            ground_height_fn=ground_height_fn,
            terrain_contains_fn=terrain_contains_fn,
            profiler=profiler,
        )
        self.speed = 6.0
        self.radius = 0.4
        self.eye_height = 1.7
        self.camera.position[2] = self.ground_height + self.eye_height

    def update(self, delta_time):
        if self._key_w in self.keys_down:
            self.move(0, self.speed * delta_time)
        if self._key_a in self.keys_down:
            self.move(90, self.speed * delta_time)
        if self._key_s in self.keys_down:
            self.move(180, self.speed * delta_time)
        if self._key_d in self.keys_down:
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
        profiler=None,
    ) -> None:
        super().__init__(
            camera,
            shaders,
            colliders=colliders,
            terrain_bounds=terrain_bounds,
            ground_height_fn=ground_height_fn,
            terrain_contains_fn=terrain_contains_fn,
            profiler=profiler,
        )
        self.player_mesh = player_mesh
        self.visual_meshes = visual_meshes or [player_mesh]
        self.animated_visual = animated_visual
        self.position = numpy.array(position, dtype=numpy.float32)
        self.mesh_rotation_offset = mesh_rotation_offset or (0.0, 0.0, 0.0)
        self.mesh_position_offset = numpy.array(mesh_position_offset or (0.0, 0.0, 0.0), dtype=numpy.float32)
        self.mesh_heading_offset = float(mesh_heading_offset)
        self.always_play_walk = bool(always_play_walk)
        self.walk_speed = 1.8
        self.run_speed = 4.8
        self.speed = self.walk_speed
        self.radius = 0.52
        self.character_height = 1.3
        self.look_sensitivity = 0.18
        self.facing_yaw = self.camera.theta
        self.target_facing_yaw = float(self.camera.theta)
        self.turn_speed = 220.0
        self.last_world_direction = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        self.active_animation_state = None
        self._has_walk_animation = bool(self.animated_visual is not None and self.animated_visual.has_animation("walk"))
        self._has_run_animation = bool(self.animated_visual is not None and self.animated_visual.has_animation("run"))
        self._has_idle_animation = bool(self.animated_visual is not None and self.animated_visual.has_animation("idle"))
        self._last_visual_signature = None
        self.camera.update_focus(self.position)
        self._sync_visuals()

    def update(self, delta_time):
        with self._profile_section("player.input"):
            move_x, move_y, is_moving = self._get_move_vector()
            is_running = is_moving and self._is_run_pressed()
            self.speed = self.run_speed if is_running else self.walk_speed

        with self._profile_section("player.facing"):
            self._update_facing(delta_time)

        with self._profile_section("player.move"):
            if is_moving:
                move_scale = self._get_turn_movement_factor()
                move_amount = self.speed * float(delta_time) * move_scale
                self._move(move_x * move_amount, move_y * move_amount)

        with self._profile_section("player.animation"):
            self._update_animation(delta_time, is_moving, is_running)

        with self._profile_section("player.ground_sync"):
            self.ground_height = self._sample_ground_height(self.position[0], self.position[1])
            self.position[2] = self.ground_height

        with self._profile_section("player.visual_sync"):
            self._sync_visuals()

        with self._profile_section("player.camera"):
            self.camera.update_focus(self.position)
            self.camera.update(self.shaders)

    def _get_move_vector(self):
        input_x = float((self._key_d in self.keys_down) - (self._key_a in self.keys_down))
        input_y = float((self._key_w in self.keys_down) - (self._key_s in self.keys_down))

        if input_x == 0.0 and input_y == 0.0:
            return 0.0, 0.0, False

        if input_x != 0.0 and input_y != 0.0:
            input_x *= DIAGONAL_NORMALIZER
            input_y *= DIAGONAL_NORMALIZER

        # Character facing is derived directly from camera heading plus input direction.
        # This keeps W aligned with the current camera forward.
        input_angle = numpy.degrees(numpy.arctan2(-input_x, input_y))
        self.target_facing_yaw = float((self.camera.theta + input_angle) % 360.0)

        camera_theta = numpy.radians(self.camera.theta)
        camera_cos = float(numpy.cos(camera_theta, dtype=numpy.float32))
        camera_sin = float(numpy.sin(camera_theta, dtype=numpy.float32))
        world_x = (camera_cos * input_y) + (camera_sin * input_x)
        world_y = (camera_sin * input_y) - (camera_cos * input_x)
        return world_x, world_y, True

    def _move(self, delta_x, delta_y):
        current_x = float(self.position[0])
        current_y = float(self.position[1])

        if delta_x != 0.0:
            next_position = self.position.copy()
            next_position[0] = current_x + delta_x
            if not self._is_blocked(next_position, self.character_height):
                current_x = float(next_position[0])

        if delta_y != 0.0:
            next_position = self.position.copy()
            next_position[0] = current_x
            next_position[1] = current_y + delta_y
            if not self._is_blocked(next_position, self.character_height):
                current_y = float(next_position[1])

        self.position[0] = current_x
        self.position[1] = current_y
        self.position[2] = self._sample_ground_height(current_x, current_y)

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

    def _is_run_pressed(self):
        return self._key_lshift in self.keys_down or self._key_rshift in self.keys_down

    def _update_animation(self, delta_time, is_moving, is_running):
        if self.animated_visual is None:
            return

        if not self._has_walk_animation and not self._has_run_animation and not self._has_idle_animation:
            return

        target_state = "run" if is_running else ("walk" if (is_moving or self.always_play_walk) else "idle")
        if target_state == "run" and not self._has_run_animation:
            target_state = "walk"
        if target_state == "walk" and not self._has_walk_animation:
            target_state = "idle"

        if target_state != self.active_animation_state:
            if target_state == "idle":
                if self._has_idle_animation:
                    self.animated_visual.play(
                        "idle",
                        loop=True,
                        paused=False,
                        restart=True,
                    )
                elif self._has_walk_animation:
                    self.animated_visual.play(
                        "walk",
                        loop=True,
                        paused=True,
                        restart=True,
                        hold_time=0.0,
                    )
            elif target_state == "walk":
                self.animated_visual.play(
                    "walk",
                    loop=True,
                    paused=False,
                    restart=True,
                )
            elif target_state == "run":
                self.animated_visual.play(
                    "run",
                    loop=True,
                    paused=False,
                    restart=True,
                )
            self.active_animation_state = target_state

        self.animated_visual.update(delta_time)

    def _sync_visuals(self):
        mesh_x = float(self.position[0] + self.mesh_position_offset[0])
        mesh_y = float(self.position[1] + self.mesh_position_offset[1])
        mesh_z = float(self.position[2] + self.mesh_position_offset[2])
        rotation_x, rotation_y, rotation_z = self.mesh_rotation_offset
        final_rotation_z = rotation_z - self.facing_yaw + self.mesh_heading_offset
        visual_signature = (mesh_x, mesh_y, mesh_z, float(rotation_x), float(rotation_y), float(final_rotation_z))
        if self._last_visual_signature == visual_signature:
            return

        self._last_visual_signature = visual_signature
        for mesh in self.visual_meshes:
            mesh.position[0] = mesh_x
            mesh.position[1] = mesh_y
            mesh.position[2] = mesh_z
            mesh.set_rotation(
                x=rotation_x,
                y=rotation_y,
                z=final_rotation_z,
            )
