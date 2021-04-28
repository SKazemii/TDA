print("[INFO] importing libraries....")
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import matplotlib.cm as cm
import config as cfg


print("[INFO] reading dataset....")
with h5py.File(cfg.dataset_file, "r") as hdf:
    barefoots = hdf.get("/barefoot/data")[:]
    # shod_commons_slow = hdf.get("/shod_common/slow/data")[:]
    # shod_commons_natural = hdf.get("/shod_common/natural/data")[:]
    # shod_commons_fast = hdf.get("/shod_common/fast/data")[:]
    # shod_others = hdf.get("/shod_other/data")[:]


print("[INFO] extracting images from the dataset....")
subject = 10
imgs_bare = np.squeeze(barefoots[subject, :, :, :])
# imgs_shod_slow = np.squeeze(shod_commons_slow[subject, :, :, :])
# imgs_shod_natural = np.squeeze(shod_commons_natural[subject, :, :, :])
# imgs_shod_fast = np.squeeze(shod_commons_fast[subject, :, :, :])
# imgs_shod_other = np.squeeze(shod_others[subject, :, :, :])


frames_bare = []
# frames_shod_slow = []
# frames_shod_natural = []
# frames_shod_fast = []
# frames_shod_other = []

fig = plt.figure()

for i in range(imgs_bare.shape[0]):
    frames_bare.append([plt.imshow(imgs_bare[i, :, :], animated=True)])

ani_1 = ArtistAnimation(fig, frames_bare, interval=50, blit=True, repeat_delay=100)
ani_1.save("barefoot_seg.gif", fps=6)
plt.show()

# fig = plt.figure()

# for i in range(imgs_shod_slow.shape[0]):
#     frames_shod_slow.append([plt.imshow(imgs_shod_slow[i, :, :], animated=True)])

# ani_2 = ArtistAnimation(
#     fig, frames_shod_slow, interval=50, blit=True, repeat_delay=1000
# )
# ani_2.save("shod_slow_seg.gif", writer="Pillow", fps=6)


# fig = plt.figure()
# for i in range(imgs_shod_natural.shape[0]):
#     frames_shod_natural.append([plt.imshow(imgs_shod_natural[i, :, :], animated=True)])

# ani_3 = ArtistAnimation(
#     fig, frames_shod_natural, interval=50, blit=True, repeat_delay=100
# )
# ani_3.save("shod_natural_seg.gif", writer="Pillow", fps=6)


# fig = plt.figure()
# for i in range(imgs_shod_fast.shape[0]):
#     frames_shod_fast.append([plt.imshow(imgs_shod_fast[i, :, :], animated=True)])
# ani_4 = ArtistAnimation(fig, frames_shod_fast, interval=50, blit=True, repeat_delay=100)
# ani_4.save("shod_fast_seg.gif", writer="Pillow", fps=6)


# fig = plt.figure()

# for i in range(imgs_shod_other.shape[0]):
#     frames_shod_other.append([plt.imshow(imgs_shod_other[i, :, :], animated=True)])
# ani_5 = ArtistAnimation(
#     fig, frames_shod_other, interval=50, blit=True, repeat_delay=100
# )
# ani_5.save("shod_other_seg.gif", writer="Pillow", fps=6)
# plt.show()