from PIL import Image
from tqdm import tqdm
import numpy as np
import math
import os


IMAGE_FILE = 'ImageSamples/steam_engine.png'
HORZ_KERNEL = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
VERT_KERNEL = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
THRESHOLD = 255 / 4


def main():
    original = Image.open(IMAGE_FILE)
    original.save('original.jpg')
    result = trace(original)
    thing = Image.fromarray(result)
    thing.save('result.jpg')


def trace(original):
    grayscale = original.convert('L')
    arr1 = np.pad(np.array(grayscale), 1)
    arr2 = arr1.copy()
    rows, cols = arr1.shape
    for y in tqdm(range(1, rows - 1)):
        for x in range(1, cols - 1):
            sel = arr1[y - 1:y + 2, x - 1:x + 2]
            gx = np.sum(sel * HORZ_KERNEL)
            gy = np.sum(sel * VERT_KERNEL)
            g_magnitude = math.sqrt(gx ** 2 + gy ** 2)
            arr2[y][x] = g_magnitude
    return arr2


def sharpen(original):
    grayscale = original.convert('L')
    arr1 = np.pad(np.array(grayscale), 1)
    arr2 = arr1.copy()
    rows, cols = arr1.shape
    for y in tqdm(range(1, rows - 1)):
        for x in range(1, cols - 1):
            sel = arr1[y - 1:y + 2, x - 1:x + 2]
            gx = np.sum(sel * HORZ_KERNEL)
            gy = np.sum(sel * VERT_KERNEL)
            g_magnitude = math.sqrt(gx ** 2 + gy ** 2)
            if g_magnitude >= THRESHOLD:
                g_magnitude = 255
            else:
                g_magnitude = 0
            arr2[y][x] = g_magnitude
    return arr2


if __name__ == '__main__':
    main()