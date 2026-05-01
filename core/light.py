from OpenGL.GL import *
import numpy


def get_uniform_location(shader, name):
    cache = getattr(get_uniform_location, "_cache", None)
    if cache is None:
        cache = {}
        get_uniform_location._cache = cache

    shader_cache = cache.setdefault(shader, {})
    location = shader_cache.get(name)
    if location is None:
        location = glGetUniformLocation(shader, name)
        shader_cache[name] = location
    return location


class Light:
    def __init__(self, shaders, color, position, strength, index, shaders_settings: list = [True], 
                            direction = [0.0, 0.0, 0.0], dynamic=False) -> None:
        self.shaders = shaders if type(shaders) == list else [shaders]
        self.color = numpy.array(color, dtype=numpy.float32)
        self.shaders_settings = shaders_settings
        self.strength = strength
        self.index = index
        while len(shaders_settings) < len(self.shaders):
            self.shaders_settings.append(True)
        self.position = numpy.array(position, dtype=numpy.float32)
        self.direction = numpy.array(direction, dtype=numpy.float32)
        self.dynamic = bool(dynamic)


    @staticmethod
    def reset_lights(shaders: list):
        for shader in shaders:
            glUseProgram(shader)
            for i in range(8):
                enable_location = get_uniform_location(shader, f"lights[{i}].enable")
                if enable_location >= 0:
                    glUniform1i(enable_location, 0)


    def update(self):
        for i, shader in enumerate(self.shaders):
            glUseProgram(shader)
            if self.shaders_settings[i]:
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].pos"), 1, self.position)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].dir"), 1, self.direction)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].color"), 1, self.color)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].strength"), self.strength)
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].enable"), 1)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].constant"), self.constant)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].linear"), self.linear)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].quadratic"), self.quadratic)


class PointLight(Light):
    def __init__(self, shaders, position, color, strength, index, shaders_settings: list=[True], dynamic=False) -> None:
        super().__init__(shaders, color, position, strength, index, shaders_settings, [0, 0, 0], dynamic=dynamic)
        self.constant = 1.0
        self.linear = 0.09
        self.quadratic = .032


    def update(self):
        for i, shader in enumerate(self.shaders):
            glUseProgram(shader)
            if self.shaders_settings[i]:
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].type"), 1)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].pos"), 1, self.position)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].color"), 1, self.color)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].strength"), self.strength)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].constant"), self.constant)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].linear"), self.linear)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].quadratic"), self.quadratic)
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].enable"), 1)


class DirectionalLight(Light):
    def __init__(self, shaders, direction, color, strength, index, shaders_settings: list=[True], dynamic=False) -> None:
        super().__init__(shaders, color, [0.0, 0.0, 2.0], strength, index, shaders_settings, direction, dynamic=dynamic)


    def update(self):
        for i, shader in enumerate(self.shaders):
            glUseProgram(shader)
            if self.shaders_settings[i]:
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].type"), 0)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].dir"), 1, self.direction)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].color"), 1, self.color)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].strength"), self.strength)
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].enable"), 1)



class FlashLight(Light):
    def __init__(self, shaders, position, direction, color, strength, index, cutOff=12.5, outerCutOff=17.5, shaders_settings=[True], dynamic=False) -> None:
        super().__init__(shaders, color, position, strength, index, shaders_settings, direction, dynamic=dynamic)
        self.cutOff = cutOff
        self.outerCutOff = outerCutOff


    def update(self):
        for i, shader in enumerate(self.shaders):
            glUseProgram(shader)
            if self.shaders_settings[i]:
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].type"), 2)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].pos"), 1, self.position)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].dir"), 1, self.direction)
                glUniform3fv(get_uniform_location(shader, f"lights[{self.index}].color"), 1, self.color)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].strength"), self.strength)
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].cutOff"), numpy.radians(self.cutOff))
                glUniform1f(get_uniform_location(shader, f"lights[{self.index}].outerCutOff"), numpy.radians(self.outerCutOff))
                glUniform1i(get_uniform_location(shader, f"lights[{self.index}].enable"), 1)
