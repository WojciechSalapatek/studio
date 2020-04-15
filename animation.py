import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import re
import shutil


def scan_frames(input_frames_dir, max_frames=None):
    print("Scanning files")
    key_pat = re.compile(r"^frame(.*).txt$")

    def key(item):
        m = key_pat.match(item)
        return int(m.group(1))

    files = []
    for file in os.listdir(input_frames_dir):
        files.append(file)
    files.sort(key=key)
    print(f"Found {len(files)} frames in {input_frames_dir}")
    files = list(map(lambda x: input_frames_dir + "/" + x, files))
    if max_frames is not None:
        return files[:max_frames]
    return files


def frames_to_images(files, shape, output_frames_dir):
    print("Transforming matrices to images")
    ctr = 0
    all_frames = len(files)
    image_files = []
    for file in files:
        data = np.loadtxt(file, dtype=float)
        rescaled = (255.0 / data.max() * (data - data.min()))
        rescaled = np.flip(rescaled, 0)
        black = np.zeros((shape[1], shape[0], 4))
        black[:, :, 3] = rescaled
        im = Image.fromarray(black.astype(np.uint8))
        image_file = output_frames_dir + "frame{}.png".format(ctr)
        im.save(image_file)
        image_files.append(image_file)
        ctr += 1
        print(f"{ctr} of {all_frames} frames transformed")
    return image_files


def make_animation(input_frames_dir, output_frames_dir, background, shape, out_file, max_frames=None):
    print("Preparing animation")

    print(f"Cleaning output frames directory {output_frames_dir}")
    shutil.rmtree(output_frames_dir, onerror=None)
    os.makedirs(output_frames_dir)

    frame_files = scan_frames(input_frames_dir, max_frames)
    image_files = frames_to_images(frame_files, shape, output_frames_dir)

    img = Image.open(background)
    img = img.resize(shape)
    fig = plt.figure()
    ims = []

    print("Preprocessing frames")
    ctr = 0
    for image_file in image_files:
        image = Image.open(image_file)
        comp = Image.alpha_composite(img, image.convert('RGBA'))
        im = plt.imshow(comp, animated=True)
        ims.append([im])
        print(f"{ctr} frames processed")
        ctr += 1

    print("Combining frames into animation")
    ani = animation.ArtistAnimation(fig, ims, interval=7, blit=True,
                                    repeat_delay=100)

    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(out_file, writer=writer)
    print(f"Animation saved at {out_file}")
    plt.show()
