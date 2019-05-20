import numpy as np
import matplotlib.pyplot as plt


def draw_comp_hist(src, out, file_name):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))

    ax0.hist(src.flatten(), np.arange(257), density=True, histtype='stepfilled', color='g', alpha=0.75)
    ax0.set_title('Original')

    ax1.hist(out.flatten(), np.arange(257), density=True, histtype='stepfilled', color='b', alpha=0.75)
    ax1.set_title('Equalized')

    fig.tight_layout()
    plt.savefig(file_name)

def draw_simple_hist(src, file_name):
    fig, ax0 = plt.subplots(ncols=1, figsize=(4, 4))

    ax0.hist(src.flatten(), np.arange(257), density=True, histtype='stepfilled', color='g', alpha=0.75)
    ax0.set_title('Original')

    fig.tight_layout()
    plt.savefig(file_name)

