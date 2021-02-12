from constants import *
from collections import deque


class Rectangle:
    def __init__(self, display, color, rect):
        self.display = display
        self.color = color
        self.rect = rect


class Snake:

    def __init__(self, x=WIDTH // 2, y=HEIGHT // 2, max_x=WIDTH, max_y=HEIGHT):
        self.direc = RIGHT
        self.segments = deque()
        self.x = x
        self.y = y
        self.max_x = max_x
        self.max_y = max_y

    def direction(self):
        return self.direc

    def change_direction(self, d):
        self.direc = d

    def update_position(self, x, y):
        self.x = x
        self.y = y

    def insert_segment(self, rect: Rectangle):
        self.segments.insert(0, rect)

    def pop_segment(self):
        self.segments.pop()

    def reset_snake(self):
        self.direc = RIGHT
        self.x = WIDTH // 2
        self.y = WIDTH // 2
        self.segments.clear()

    def is_collision(self) -> bool:
        """
        Checks for a collision with the snake with the wall, and with itself
        :return: False if no collision, True if collided
        """
        for i, seg in enumerate(self.segments):
            if seg.rect[0] - RECT_SIZE < self.x < seg.rect[0] + RECT_SIZE \
                    and seg.rect[1] - RECT_SIZE < self.y < seg.rect[1] + RECT_SIZE and i != 0:
                return True

        if self.x > (self.max_x - RECT_SIZE) or self.x < 0:
            return True

        if self.y > self.max_y or self.y < 0 - RECT_SIZE:
            return True

        return False

    def did_get_food(self, apple_x, apple_y) -> bool:
        return apple_x - RECT_SIZE < self.x < apple_x + RECT_SIZE and apple_y - RECT_SIZE < self.y < apple_y + RECT_SIZE
