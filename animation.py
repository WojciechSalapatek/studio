import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import re


def scan_frames(input_frames_dir):
    key_pat = re.compile(r"^frame(.*).txt$")

    def key(item):
        m = key_pat.match(item)
        return int(m.group(1))

    files = []
    for file in os.listdir(input_frames_dir):
        files.append(file)
    files.sort(key=key)
    print(f"Found {len(files)} frames in {input_frames_dir}")
    return list(map(lambda x: input_frames_dir + "/" + x, files))


def frames_to_images(files, shape):
    images = []
    for file in files:
        data = np.loadtxt(file, dtype=float)
        rescaled = (255.0 / data.max() * (data - data.min()))
        rescaled = np.flip(rescaled, 0)
        black = np.zeros((shape[1], shape[0], 4))
        black[:, :, 3] = rescaled
        images.append(Image.fromarray(black.astype(np.uint8)))
    return images


def make_animation(input_frames_dir, background, shape, out_file):
    print("preparing animation")

    frame_files = scan_frames(input_frames_dir)
    frame_images = frames_to_images(frame_files, shape)

    img = Image.open(background)
    img = img.resize(shape)
    fig = plt.figure()
    ims = []

    for image in frame_images:
        comp = Image.alpha_composite(img, image.convert('RGBA'))
        im = plt.imshow(comp, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=7, blit=True,
                                    repeat_delay=100)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(out_file, writer=writer)
    plt.show()
    print(f"Animation saved at {out_file}")