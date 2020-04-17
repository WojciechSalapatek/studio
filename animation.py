import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import re
import shutil
from PIL import ImageOps


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
    np.seterr(all='raise')
    print("Transforming matrices to images")
    ctr = 0
    all_frames = len(files)
    image_files = []
    for file in files:
        data = np.loadtxt(file, dtype=float)
        # rescaled = 255.0 * np.tanh(data/2500)
        data[data<0]=0
        data[data>255] = 255
        rescaled=data
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


def prepare_background(background, latitude_range, longitude_range, shape):
    print("Preparing background")

    img = Image.open(background)
    width, height = img.size

    # assume entire images fits area as follows
    min_lat = 18
    max_lat = 31
    min_lon = -101
    max_lon = -82

    lat_to_pixel = height/(max_lat-min_lat)
    lon_to_pixel = width/(max_lon-min_lon)

    up_crop = round((latitude_range[0] - min_lat)*lat_to_pixel)
    down_crop = round((max_lat - latitude_range[1])*lat_to_pixel)
    left_crop = round((longitude_range[0] - min_lon)*lon_to_pixel)
    right_crop = round((max_lon - longitude_range[1])*lon_to_pixel)

    border = (left_crop, down_crop, right_crop, up_crop)  # left, up, right, bottom
    img = ImageOps.crop(img, border)
    img = img.resize(shape)
    return img


def make_animation(input_frames_dir, output_frames_dir, background, latitude_range, longitude_range, shape, out_file, max_frames=None):
    print("Preparing animation")

    background_img = prepare_background(background, latitude_range, longitude_range, shape)

    print(f"Cleaning output frames directory {output_frames_dir}")
    shutil.rmtree(output_frames_dir, onerror=None)
    os.makedirs(output_frames_dir)

    frame_files = scan_frames(input_frames_dir, max_frames)
    image_files = frames_to_images(frame_files, shape, output_frames_dir)

    fig = plt.figure()
    ims = []

    print("Preprocessing frames")
    ctr = 0
    for image_file in image_files:
        image = Image.open(image_file)
        comp = Image.alpha_composite(background_img, image.convert('RGBA'))
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
