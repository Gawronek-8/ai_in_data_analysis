import enum

class ObjectCategory(enum.IntFlag):

    PLAYER = 1
    WALL = 2
    ROCK = 4
    TARGET = 8