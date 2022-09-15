import os
import time
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool

from torchvision import transforms as T
from torchvision.datasets.folder import pil_loader

in_root = r"C:\Users\pennstateuas\projects\datasets\fgvc-aircraft-2013b\data\images"
out_root = r"C:\Users\pennstateuas\projects\datasets\fgvc-aircraft-2013b-128\images"
out_size = int(128)
threads = 6

resize = T.Resize(out_size)
os.makedirs(out_root, exist_ok=True)
in_image_path = [os.path.join(in_root, path) for path in os.listdir(in_root)]

def resize_save(image_path):
    in_img = pil_loader(image_path)
    head, tail = os.path.split(image_path)
    out_img = resize(in_img)
    out_path = os.path.join(out_root, tail)
    out_img.save(out_path)

start_time = time.time()
print(" * Starting resize...")

pool = ThreadPool(threads)
pool.map(resize_save, in_image_path)

duration = time.time() - start_time
print(" * Resize Complete")
print(" * Duration {:.2f} Seconds".format(duration))
print(" * {:.2f} Images per Second".format(len(in_image_path)/duration))
