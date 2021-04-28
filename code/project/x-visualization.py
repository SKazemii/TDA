print("[INFO] importing libraries....")

import pickle

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# from matplotlib.animation import ArtistAnimation


# with h5py.File(cfg.dataset_file, "r") as hdf:
#     barefoots = hdf.get("/barefoot/data")[:]
# imgs_bare = np.squeeze(barefoots[1, :, :, :])
# frames_bare = []
# fig = plt.figure()

# for i in range(60):
#     frames_bare.append([plt.imshow(imgs_bare[i, :, :], animated=True)])

# ani_1 = animation.ArtistAnimation(
#     fig, frames_bare, interval=50, blit=True, repeat_delay=100
# )
# # ani_1.save("barefoot_seg.gif", fps=6)
# plt.show()


print("[INFO] importing pickles files....")
with open(os.path.join(cfg.pickle_dir, "df_sum.pickle"), "rb") as handle:
    df_sum = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max.pickle"), "rb") as handle:
    df_max = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe.pickle"), "rb") as handle:
    df_xCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe.pickle"), "rb") as handle:
    df_yCe = pickle.load(handle)

Sub = "Sample_1"

print(cfg.pickle_dir)


class SubplotAnimation(animation.TimedAnimation):
    def __init__(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 4)

        self.t = np.linspace(0, 59, 60)
        # self.x = np.cos(2 * np.pi * self.t / 10.0)
        self.x = df_xCe[Sub].values  #
        self.y = df_yCe[Sub].values  #
        # self.y = np.sin(2 * np.pi * self.t / 10.0)
        self.z = self.t

        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        self.line1 = Line2D([], [], color="black")
        self.line1a = Line2D([], [], color="red", linewidth=2)
        self.line1e = Line2D([], [], color="red", marker="o", markeredgecolor="r")
        ax1.add_line(self.line1)
        ax1.add_line(self.line1a)
        ax1.add_line(self.line1e)
        ax1.set_xlim(0, 75)
        ax1.set_ylim(0, 75)
        ax1.set_aspect("equal", "datalim")

        ax2.set_xlabel("y")
        ax2.set_ylabel("z")
        self.line2 = Line2D([], [], color="black")
        self.line2a = Line2D([], [], color="red", linewidth=2)
        self.line2e = Line2D([], [], color="red", marker="o", markeredgecolor="r")
        ax2.add_line(self.line2)
        ax2.add_line(self.line2a)
        ax2.add_line(self.line2e)
        ax2.set_xlim(0, 71)
        ax2.set_ylim(11, 70)

        ax3.set_xlabel("x")
        ax3.set_ylabel("z")
        self.line3 = Line2D([], [], color="black")
        self.line3a = Line2D([], [], color="red", linewidth=2)
        self.line3e = Line2D([], [], color="red", marker="o", markeredgecolor="r")
        ax3.add_line(self.line3)
        ax3.add_line(self.line3a)
        ax3.add_line(self.line3e)
        ax3.set_xlim(0, 71)
        ax3.set_ylim(11, 70)

        animation.TimedAnimation.__init__(self, fig, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])

        self.line2.set_data(self.z[:i], self.y[:i])
        self.line2a.set_data(self.y[head_slice], self.z[head_slice])
        self.line2e.set_data(self.z[head], self.y[head])

        self.line3.set_data(self.z[:i], self.x[:i])
        self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        self.line3e.set_data(self.z[head], self.x[head])

        self._drawn_artists = [
            self.line1,
            self.line1a,
            self.line1e,
            self.line2,
            self.line2a,
            self.line2e,
            self.line3,
            self.line3a,
            self.line3e,
        ]

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _init_draw(self):
        lines = [
            self.line1,
            self.line1a,
            self.line1e,
            self.line2,
            self.line2a,
            self.line2e,
            self.line3,
            self.line3a,
            self.line3e,
        ]
        for l in lines:
            l.set_data([], [])


ani = SubplotAnimation()
# ani.save("test_sub.mp4")
plt.show()