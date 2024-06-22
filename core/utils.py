from core.core import CameraFirstPerson
from OpenGL.GL import *
import pygame


class PlayerFirstPerson:
    def __init__(self, screen_size, camera: CameraFirstPerson, shaders) -> None:
        self.screen_size = screen_size
        self.camera = camera
        self.shaders = shaders
        self.speed = 15


    def update(self, delta_time):
        key_ = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pos()

        if key_[pygame.K_w]:
            self.move(0, self.speed * delta_time)
        if key_[pygame.K_a]:
            self.move(90, self.speed * delta_time)
        if key_[pygame.K_s]:
            self.move(180, self.speed * delta_time)
        if key_[pygame.K_d]:
            self.move(-90, self.speed * delta_time)

        # print(fps)
        horizontal = self.speed * delta_time * (self.screen_size[0]/2 - mouse[0])
        vertical = self.speed * delta_time * (self.screen_size[1]/2 - mouse[1])
        self.rotate(horizontal, vertical)
        pygame.mouse.set_pos((self.screen_size[0]//2, self.screen_size[1]//2))

        self.camera.update(self.shaders)


    def move(self, direction, amount):
        self.camera.move(direction, amount)


    def rotate(self, horizontal, vertical):
        self.camera.increment_direction(horizontal, vertical)
