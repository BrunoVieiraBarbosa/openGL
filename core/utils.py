from contextlib import nullcontext

import arcade
import numpy

from core.animation import CharacterAnimationController
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
        self.animation_controller = CharacterAnimationController(self.animated_visual)
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
        self.animation_controller.update_locomotion(
            delta_time,
            is_moving=is_moving,
            is_running=is_running,
            always_walk=self.always_play_walk,
        )

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


class BaseNPC:
    def __init__(
        self,
        primary_mesh,
        position,
        visual_meshes=None,
        animated_visual=None,
        ground_height_fn=None,
        mesh_rotation_offset=None,
        mesh_position_offset=None,
        mesh_heading_offset=-90.0,
        move_speed=1.6,
        turn_speed=180.0,
        waypoint_tolerance=0.18,
        wait_time=1.0,
        look_target=None,
        look_target_radius=2.6,
        investigate_speed=2.1,
        investigate_radius=3.6,
        investigate_duration=2.5,
        investigate_stop_radius=1.15,
        vision_angle_deg=75.0,
        perception_rotation_offset_deg=0.0,
        profiler=None,
    ) -> None:
        self.primary_mesh = primary_mesh
        self.visual_meshes = visual_meshes or [primary_mesh]
        self.animated_visual = animated_visual
        self.position = numpy.array(position, dtype=numpy.float32)
        self.ground_height_fn = ground_height_fn
        self.mesh_rotation_offset = mesh_rotation_offset or (0.0, 0.0, 0.0)
        self.mesh_position_offset = numpy.array(mesh_position_offset or (0.0, 0.0, 0.0), dtype=numpy.float32)
        self.mesh_heading_offset = float(mesh_heading_offset)
        self.move_speed = float(move_speed)
        self.turn_speed = float(turn_speed)
        self.waypoint_tolerance = float(waypoint_tolerance)
        self.wait_time = float(wait_time)
        self.wait_timer = 0.0
        self.look_target = look_target
        self.look_target_radius = float(look_target_radius)
        self.investigate_speed = float(investigate_speed)
        self.investigate_radius = float(investigate_radius)
        self.investigate_duration = float(investigate_duration)
        self.investigate_stop_radius = float(investigate_stop_radius)
        self.vision_angle_deg = float(vision_angle_deg)
        self.vision_cos_threshold = float(numpy.cos(numpy.radians(self.vision_angle_deg * 0.5)))
        self.perception_rotation_offset_deg = float(perception_rotation_offset_deg)
        self.profiler = profiler
        self.investigate_timer = 0.0
        self.facing_yaw = 0.0
        self.target_facing_yaw = 0.0
        self.is_moving = False
        self.is_interacting_with_player = False
        self._debug_cone_forward_vector = numpy.array([1.0, 0.0], dtype=numpy.float32)
        self._debug_cone_rotation_z = 0.0
        self.animation_controller = CharacterAnimationController(self.animated_visual)
        self.animation_time = 0.0
        self._last_visual_signature = None

        if self.ground_height_fn is not None:
            self.position[2] = float(self.ground_height_fn(self.position[0], self.position[1]))

        self._initialize_facing()
        self._refresh_debug_cone_state()
        self._sync_visuals()

    def update(self, delta_time):
        with self._profile_section("npc.ai"):
            self.is_moving = False
            self.is_interacting_with_player = False
            self._update_behavior(delta_time)

        with self._profile_section("npc.ground_sync"):
            if self.ground_height_fn is not None:
                self.position[2] = float(self.ground_height_fn(self.position[0], self.position[1]))

        with self._profile_section("npc.animation"):
            self.animation_time += float(delta_time)
            self._update_animation(delta_time, self.is_moving, self.is_interacting_with_player)

        with self._profile_section("npc.visual_sync"):
            self._sync_visuals()

    def _initialize_facing(self):
        return

    def _update_behavior(self, delta_time):
        raise NotImplementedError

    def _get_target_delta(self):
        if self.look_target is None:
            return None
        target_position = self.look_target.position if hasattr(self.look_target, "position") else self.look_target
        delta_x = float(target_position[0] - self.position[0])
        delta_y = float(target_position[1] - self.position[1])
        distance_sq = delta_x * delta_x + delta_y * delta_y
        return delta_x, delta_y, distance_sq

    def _begin_investigation_if_target_visible(self, target_delta):
        if target_delta is None:
            return
        delta_x, delta_y, distance_sq = target_delta
        if distance_sq <= self.investigate_radius * self.investigate_radius and self._can_see_target(delta_x, delta_y):
            self.investigate_timer = self.investigate_duration

    def _update_investigation(self, delta_time, target_delta):
        if self.investigate_timer <= 0.0 or target_delta is None:
            return False

        delta_x, delta_y, distance_sq = target_delta
        self.investigate_timer = max(0.0, self.investigate_timer - float(delta_time))
        self.target_facing_yaw = float(numpy.degrees(numpy.arctan2(delta_y, delta_x)))
        self._update_facing(delta_time)
        self.is_interacting_with_player = True

        if distance_sq > self.investigate_stop_radius * self.investigate_stop_radius:
            move_step = min(
                self.investigate_speed * float(delta_time),
                max(0.0, float(numpy.sqrt(distance_sq)) - self.investigate_stop_radius),
            )
            self.is_moving = self._move_in_direction(delta_x, delta_y, move_step) or self.is_moving
        return True

    def _move_in_direction(self, delta_x, delta_y, move_step):
        distance_sq = delta_x * delta_x + delta_y * delta_y
        if distance_sq <= 1e-8 or move_step <= 0.0:
            return False
        distance = float(numpy.sqrt(distance_sq))
        self.position[0] += (delta_x / distance) * move_step
        self.position[1] += (delta_y / distance) * move_step
        return True

    def _move_towards_point(self, delta_time, point, speed, stop_distance=0.0):
        delta_x = float(point[0] - self.position[0])
        delta_y = float(point[1] - self.position[1])
        distance_sq = delta_x * delta_x + delta_y * delta_y
        if distance_sq <= stop_distance * stop_distance:
            return False

        self.target_facing_yaw = float(numpy.degrees(numpy.arctan2(delta_y, delta_x)))
        self._update_facing(delta_time)
        move_step = min(speed * float(delta_time), max(0.0, float(numpy.sqrt(distance_sq)) - stop_distance))
        if self._move_in_direction(delta_x, delta_y, move_step):
            self.is_moving = True
            return True
        return False

    def _update_facing(self, delta_time):
        angle_delta = ((self.target_facing_yaw - self.facing_yaw + 180.0) % 360.0) - 180.0
        max_step = self.turn_speed * float(delta_time)
        if abs(angle_delta) <= max_step:
            self.facing_yaw = self.target_facing_yaw
            self._refresh_debug_cone_state()
            return
        self.facing_yaw = float((self.facing_yaw + numpy.sign(angle_delta) * max_step) % 360.0)
        self._refresh_debug_cone_state()

    def _refresh_debug_cone_state(self):
        base_rotation_radians = numpy.radians(self.get_visual_rotation_z())
        base_x = float(numpy.cos(base_rotation_radians, dtype=numpy.float32))
        base_y = float(numpy.sin(base_rotation_radians, dtype=numpy.float32))
        heading_correction = numpy.radians(-self.mesh_heading_offset)
        cos_angle = float(numpy.cos(heading_correction, dtype=numpy.float32))
        sin_angle = float(numpy.sin(heading_correction, dtype=numpy.float32))
        self._debug_cone_forward_vector[0] = (base_x * cos_angle) - (base_y * sin_angle)
        self._debug_cone_forward_vector[1] = (base_x * sin_angle) + (base_y * cos_angle)
        self._debug_cone_rotation_z = float(
            numpy.degrees(
                numpy.arctan2(
                    self._debug_cone_forward_vector[1],
                    self._debug_cone_forward_vector[0],
                )
            )
        )

    def _can_see_target(self, delta_x, delta_y):
        distance_sq = (delta_x * delta_x) + (delta_y * delta_y)
        if distance_sq <= 1e-8:
            return True

        distance = float(numpy.sqrt(distance_sq))
        target_dir_x = delta_x / distance
        target_dir_y = delta_y / distance
        dot = (
            float(self._debug_cone_forward_vector[0]) * target_dir_x
        ) + (
            float(self._debug_cone_forward_vector[1]) * target_dir_y
        )
        return dot >= self.vision_cos_threshold

    def _update_animation(self, delta_time, is_moving, is_interacting_with_player):
        self.animation_controller.update_locomotion(
            delta_time,
            is_moving=is_moving,
            force_idle=(is_interacting_with_player and not is_moving),
            shared_time=self.animation_time,
        )

    def _sync_visuals(self):
        mesh_x = float(self.position[0] + self.mesh_position_offset[0])
        mesh_y = float(self.position[1] + self.mesh_position_offset[1])
        mesh_z = float(self.position[2] + self.mesh_position_offset[2])
        rotation_x, rotation_y, rotation_z = self.mesh_rotation_offset
        final_rotation_z = self.get_visual_rotation_z()
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

    def get_visual_rotation_z(self):
        return self.mesh_rotation_offset[2] - self.facing_yaw + self.mesh_heading_offset

    def get_perception_rotation_z(self):
        return self._debug_cone_rotation_z

    def get_perception_forward_vector(self):
        return self._debug_cone_forward_vector

    def get_debug_cone_forward_vector(self):
        return self._debug_cone_forward_vector

    def _profile_section(self, name):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.section(name)


class PatrolNPC(BaseNPC):
    def __init__(
        self,
        primary_mesh,
        position,
        waypoints,
        **kwargs,
    ) -> None:
        self.waypoints = [numpy.array(point, dtype=numpy.float32) for point in waypoints]
        self.waypoint_index = 0
        super().__init__(primary_mesh, position, **kwargs)

    def _initialize_facing(self):
        if not self.waypoints:
            return
        first_target = self.waypoints[0] - self.position
        if abs(float(first_target[0])) > 1e-6 or abs(float(first_target[1])) > 1e-6:
            self.facing_yaw = float(numpy.degrees(numpy.arctan2(first_target[1], first_target[0])))
            self.target_facing_yaw = self.facing_yaw

    def _update_behavior(self, delta_time):
        target_delta = self._get_target_delta()
        self._begin_investigation_if_target_visible(target_delta)
        if self._update_investigation(delta_time, target_delta):
            return

        if self.wait_timer > 0.0:
            self.wait_timer = max(0.0, self.wait_timer - float(delta_time))
            return

        if not self.waypoints:
            return

        waypoint = self.waypoints[self.waypoint_index]
        delta_x = float(waypoint[0] - self.position[0])
        delta_y = float(waypoint[1] - self.position[1])
        distance_sq = delta_x * delta_x + delta_y * delta_y
        if distance_sq <= self.waypoint_tolerance * self.waypoint_tolerance:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.wait_timer = self.wait_time
            return

        self._move_towards_point(delta_time, waypoint, self.move_speed)


class SentryNPC(BaseNPC):
    def __init__(
        self,
        primary_mesh,
        position,
        home_position=None,
        facing_yaw=0.0,
        scan_half_angle=45.0,
        scan_speed=55.0,
        return_speed=1.5,
        **kwargs,
    ) -> None:
        if home_position is None:
            home_position = position
        self.home_position = numpy.array(home_position, dtype=numpy.float32)
        self.base_facing_yaw = float(facing_yaw)
        self.scan_half_angle = float(scan_half_angle)
        self.scan_speed = float(scan_speed)
        self.return_speed = float(return_speed)
        self.scan_direction = 1.0
        super().__init__(primary_mesh, position, **kwargs)

    def _initialize_facing(self):
        self.facing_yaw = self.base_facing_yaw
        self.target_facing_yaw = self.base_facing_yaw

    def _update_behavior(self, delta_time):
        target_delta = self._get_target_delta()
        self._begin_investigation_if_target_visible(target_delta)
        if self._update_investigation(delta_time, target_delta):
            return

        home_delta_x = float(self.home_position[0] - self.position[0])
        home_delta_y = float(self.home_position[1] - self.position[1])
        home_distance_sq = home_delta_x * home_delta_x + home_delta_y * home_delta_y
        if home_distance_sq > self.waypoint_tolerance * self.waypoint_tolerance:
            self._move_towards_point(delta_time, self.home_position, self.return_speed)
            return

        self.position[0] = float(self.home_position[0])
        self.position[1] = float(self.home_position[1])
        next_yaw = self.target_facing_yaw + (self.scan_direction * self.scan_speed * float(delta_time))
        max_yaw = self.base_facing_yaw + self.scan_half_angle
        min_yaw = self.base_facing_yaw - self.scan_half_angle
        if next_yaw >= max_yaw:
            next_yaw = max_yaw
            self.scan_direction = -1.0
        elif next_yaw <= min_yaw:
            next_yaw = min_yaw
            self.scan_direction = 1.0
        self.target_facing_yaw = next_yaw
        self._update_facing(delta_time)
