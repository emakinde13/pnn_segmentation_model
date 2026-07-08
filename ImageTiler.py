import numpy as np
from Tile import Tile

class ImageTiler:
    def __init__(self, image, tile_size=64):
        # area = len(image[0]) * len(image)
        # if area % tile_size != int(0):
        #     raise ValueError(f"The area {area} is not equally divisible by {tile_size}!") # TODO: Solve this also I think this is just wrong
        height = len(image)
        width = len(image[0])
        self.tiles = []
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile_image = image[y: y + tile_size, x: x + tile_size]
                tile = Tile(tile_image, x, y, image)
                self.tiles.append(tile)