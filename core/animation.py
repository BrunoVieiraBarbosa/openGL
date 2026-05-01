class CharacterAnimationController:
    def __init__(self, animated_visual) -> None:
        self.animated_visual = animated_visual
        self.active_state = None
        self.uses_paused_idle_fallback = False
        self.has_walk = bool(animated_visual is not None and animated_visual.has_animation("walk"))
        self.has_run = bool(animated_visual is not None and animated_visual.has_animation("run"))
        self.has_idle = bool(animated_visual is not None and animated_visual.has_animation("idle"))

    def update_locomotion(self, delta_time, is_moving, is_running=False, always_walk=False, force_idle=False, shared_time=None):
        if self.animated_visual is None:
            return

        if not self.has_walk and not self.has_run and not self.has_idle:
            return

        if force_idle:
            target_state = "idle"
        else:
            target_state = "run" if is_running else ("walk" if (is_moving or always_walk) else "idle")

        if target_state == "run" and not self.has_run:
            target_state = "walk"
        if target_state == "walk" and not self.has_walk:
            target_state = "idle"

        if target_state == "idle" and not self.has_idle and self.has_walk:
            if self.active_state != "idle" or not self.uses_paused_idle_fallback:
                self.animated_visual.play(
                    "walk",
                    loop=True,
                    paused=True,
                    restart=True,
                    hold_time=0.0,
                )
                self.active_state = "idle"
                self.uses_paused_idle_fallback = True
                self.animated_visual.update(delta_time)
            return

        hold_time = self._resolve_hold_time(target_state, shared_time)
        if target_state != self.active_state:
            self.animated_visual.play(
                target_state,
                loop=True,
                paused=False,
                restart=True,
                hold_time=hold_time,
            )
            self.active_state = target_state
            self.uses_paused_idle_fallback = False

        if hold_time is not None and hasattr(self.animated_visual, "set_animation_time"):
            self.animated_visual.set_animation_time(hold_time)
            return

        self.animated_visual.update(delta_time)

    def _resolve_hold_time(self, target_state, shared_time):
        if shared_time is None or not hasattr(self.animated_visual, "get_animation_duration"):
            return None

        duration = self.animated_visual.get_animation_duration(target_state)
        if duration is None or duration <= 0.0:
            return 0.0
        return float(shared_time) % float(duration)
