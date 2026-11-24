# Ported from chirp_competition d040123 (src/plot_what_is_a_chirp_and_why_use_spetrograms.py).
# Changes: spectrogram helpers from legacy_spectrogram (NumPy copy of gridtools v1/torchaudio);
# input from PROCESSED; figure via paper_config.figure(); EOD noise seeded with SEED;
# 17.4 x 10.9 cm (was 16 x 10 cm): full width, like the other figures.
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from legacy_spectrogram import compute_spectrogram, to_decibel
from matplotlib import patches
from PIL import Image
from thunderfish.fakefish import wavefish_eods
from pathlib import Path

from paper_config import SEED, figure, processed
from style import letter_subplots, set_light_style

set_light_style()
samplerate = 20000
duration = 2

filename = Path(__file__).stem
targetpath = figure(filename)


# def circle_annotate(ax, xy, xy_adjust_text, text):
#     xy = np.array(xy)
#     xy_adjust_text = np.array(xy_adjust_text)
#     ax.text(
#         *xy,
#         " ",
#         ha="center",
#         va="center",
#         color="black",
#         zorder=1100,
#         transform=ax.transData,
#         clip_on=False,
#         bbox=dict(boxstyle="circle", fc="w", ec="k"),
#         fontsize=8,
#     )
#
#     xy_text = xy + xy_adjust_text
#     ax.text(
#         *xy_text,
#         text,
#         ha="center",
#         va="center_baseline",
#         color="black",
#         zorder=1101,
#         transform=ax.transData,
#         clip_on=False,
#         fontsize=8,
#     )
#
#
# def circle(ax, xy, radius, kwargs=None):
#     """Create circle on figure with axes of different sizes.
#
#     Plots a circle on the current axes using `plt.Circle`, taking into account
#     the figure size and the axes units.
#
#     It is done by plotting in the figure coordinate system, taking the aspect
#     ratio into account. In this way, the data dimensions do not matter.
#     However, if you adjust `xlim` or `ylim` after plotting `circle`, it will
#     screw them up; set `plt.axis` before calling `circle`.
#
#     Parameters
#     ----------
#     xy, radius, kwars :
#         As required for `plt.Circle`.
#
#     """
#     # Get current figure and axis
#     fig = plt.gcf()
#
#     # Calculate figure dimension ratio width/height
#     pr = fig.get_figwidth() / fig.get_figheight()
#     print(pr)
#     xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
#     yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
#     print(xrange, yrange)
#     pr = xrange / yrange
#     print(pr)
#     pr = ax.get_data_ratio()
#     print(pr)
#
#     # Get the transScale (important if one of the axis is in log-scale)
#     tscale = ax.transScale + (ax.transLimits + ax.transAxes)
#     ctscale = tscale.transform_point(xy)
#     cfig = fig.transFigure.inverted().transform(ctscale)
#     # cfig = xy
#
#     # Create circle
#     if kwargs == None:
#         circ = patches.Ellipse(
#             cfig,
#             radius,
#             radius * pr,
#             # transform=fig.transFigure,
#             transform=ax.transData,
#             clip_on=False,
#         )
#     else:
#         circ = patches.Ellipse(
#             cfig,
#             radius,
#             radius * pr,
#             # transform=fig.transFigure,
#             transform=ax.transData,
#             clip_on=False,
#             **kwargs,
#         )
#
#     # get circle width and height
#     circ_width = radius
#     circ_height = radius * pr
#
#     # move text coordinates slightly to the right and down
#     cfig = (cfig[0] + circ_width / 17, cfig[1] - circ_height / 10)
#
#     text = ax.text(
#         cfig[0],
#         cfig[1],
#         "1",
#         ha="center",
#         va="center",
#         color="black",
#         zorder=1100,
#         # transform=fig.transFigure,
#         transform=ax.transData,
#         clip_on=False,
#     )
#     # Draw circle
#     ax.add_patch(circ)


def main():
    dataset = np.load(processed("chirp_overview_plot_data.npz"))

    chirps = dataset["chirps"]
    heights = dataset["heights"]
    widths = dataset["widths"]
    time = dataset["time"]

    small_chirp = chirps[(heights < 80) & (widths < 0.02)][0]
    long_chirp = chirps[(heights > 400) & (widths > 0.08)][3]

    colors = sns.color_palette("mako", 4)

    chirpc = colors[1]
    basec = colors[2]
    envc = colors[3]

    # set fundamental frequencies
    eodf1 = 650
    eodf2 = 924
    eodfchirp = 700

    # make freq traces
    nsamples = int(duration * samplerate)
    f1 = np.zeros(nsamples) + eodf1
    f2 = np.zeros(nsamples) + eodf2
    f_longchirp = np.zeros(nsamples) + eodfchirp
    f_smallchirp = np.zeros(nsamples) + eodfchirp
    f_longchirp_bl = np.zeros(nsamples) + eodfchirp
    f_smallchirp_bl = np.zeros(nsamples) + eodfchirp

    # add chirps to the freq traces into the center
    f_longchirp[
        int(nsamples // 2) - len(long_chirp) // 2 : int(nsamples // 2)
        + len(long_chirp) // 2
    ] += long_chirp
    f_smallchirp[
        int(nsamples // 2) - len(small_chirp) // 2 : int(nsamples // 2)
        + len(small_chirp) // 2
    ] += small_chirp

    # make eods of the fish (wavefish_eods adds noise from the global numpy RNG)
    np.random.seed(SEED)
    eod_smallchirp = wavefish_eods("Alepto", f_smallchirp, samplerate)
    eod_longchirp = wavefish_eods("Alepto", f_longchirp, samplerate)
    eod1 = wavefish_eods("Alepto", f1, samplerate, noise_std=0.1)
    eod2 = wavefish_eods("Alepto", f2, samplerate, noise_std=0.2)

    # add the AM envelope
    smallchirp_contrast = 0.2
    longchirp_contrast = 0.7
    am_smallchirp = (
        (-(f_smallchirp - eodfchirp) / np.abs(np.max(f_smallchirp - eodfchirp)))
        * smallchirp_contrast
    ) + 1
    am_longchirp = (
        (-(f_longchirp - eodfchirp) / np.abs(np.max(f_longchirp - eodfchirp)))
        * longchirp_contrast
    ) + 1

    # apply the AM envelope
    eod_smallchirp *= am_smallchirp
    eod_longchirp *= am_longchirp

    # normalize the eods
    # eod_smallchirp /= np.max(np.abs(eod_smallchirp))
    # eod_longchirp /= np.max(np.abs(eod_longchirp))

    # variable amplitudes of the other fish
    eod1 *= 1.2
    eod2 *= 0.3

    # add a motion envelope
    approach = np.linspace(0, 1, nsamples)
    retreat = np.linspace(1, 0, nsamples)
    eod1 *= approach
    eod2 *= retreat

    soundscape_smallchirp = eod1 + eod2 + eod_smallchirp
    soundscape_longchirp = eod1 + eod2 + eod_longchirp

    eod_smallchirp -= np.mean(eod_smallchirp)
    eod_longchirp -= np.mean(eod_longchirp)
    soundscape_smallchirp -= np.mean(soundscape_smallchirp)
    soundscape_longchirp -= np.mean(soundscape_longchirp)

    # make a new time axis
    time = np.linspace(-duration / 2, duration / 2, nsamples) * 1000

    # compute the spectrograms of the soundscape
    nfft = 4096
    hop_len = nfft // 10
    S_smallchirp, spec_time, spec_freq = compute_spectrogram(
        soundscape_smallchirp, samplerate, nfft=nfft, hop_length=hop_len
    )
    S_longchirp, _, _ = compute_spectrogram(
        soundscape_longchirp, samplerate, nfft=nfft, hop_length=hop_len
    )
    S_smallchirp = to_decibel(S_smallchirp)
    S_longchirp = to_decibel(S_longchirp)

    # convert specs to 255 int values
    S_smallchirp = (S_smallchirp - np.min(S_smallchirp)) / (
        np.max(S_smallchirp) - np.min(S_smallchirp)
    )
    S_smallchirp = (S_smallchirp * 255).astype(np.uint8)
    S_longchirp = (S_longchirp - np.min(S_longchirp)) / (
        np.max(S_longchirp) - np.min(S_longchirp)
    )
    S_longchirp = (S_longchirp * 255).astype(np.uint8)

    # interpolate to get the same length
    spec_time = np.linspace(-duration / 2, duration / 2, nsamples)
    S_smallchirp = Image.fromarray(S_smallchirp).resize(
        (nsamples, S_smallchirp.shape[0])
    )
    S_longchirp = Image.fromarray(S_longchirp).resize((nsamples, S_longchirp.shape[0]))

    cm = 1 / 2.54
    figsize = (17.4 * cm, 10.9 * cm)  # full width; the aspect of the former 16 x 10 cm
    fig, ax = plt.subplots(
        4,
        2,
        figsize=figsize,
        constrained_layout=True,
        height_ratios=[1, 1, 1, 2],
        gridspec_kw={"hspace": 0.0, "wspace": 0.0},
        dpi=300,
    )

    ftraces_ylims = [600, 1200]
    ftraces_xlims = [-120, 120]
    ax[0, 0].plot(time, f_smallchirp, color=chirpc, zorder=100, lw=1.5)
    ax[0, 0].plot(time, f1, color=basec, lw=1.5)
    ax[0, 0].plot(time, f2, color=basec, lw=1.5)
    ax[0, 0].plot(
        time,
        f_smallchirp_bl,
        color="grey",
        lw=1,
        ls="dotted",
        zorder=0,
        alpha=1,
    )
    ax[0, 0].set_ylim(ftraces_ylims)
    ax[0, 1].plot(time, f_longchirp, color=chirpc, zorder=100, lw=1.5)
    ax[0, 1].plot(time, f1, color=basec, lw=1.5)
    ax[0, 1].plot(time, f2, color=basec, lw=1.5)
    ax[0, 1].plot(
        time,
        f_longchirp_bl,
        color="grey",
        lw=1,
        ls="dotted",
        zorder=0,
        alpha=1,
    )
    ax[0, 1].set_ylim(ftraces_ylims)
    [a.set_xlim(ftraces_xlims) for a in ax[0, :]]
    [a.set_xticks([]) for a in ax[0, :]]
    [
        a.spines[s].set_visible(False)
        for a in ax[0, :]
        for s in ["top", "right", "bottom"]
    ]
    # set major ticks
    [a.set_yticks([600, 900, 1200]) for a in ax[0, :]]
    ax[0, 0].set_ylabel("$\\text{EOD}f$ [Hz]")
    # add some labels to the lines
    [
        a.text(
            90,
            eodf1 - 40,
            "$\\text{EOD}f_{1}$",
            ha="left",
            va="top",
            fontsize="small",
        )
        for a in ax[0, :]
    ]
    [
        a.text(
            90,
            eodf2 + 20,
            "$\\text{EOD}f_{2}$",
            ha="left",
            va="bottom",
            fontsize="small",
        )
        for a in ax[0, :]
    ]
    [
        a.text(
            90,
            eodfchirp + 20,
            "$\\text{EOD}f_{0}$",
            ha="left",
            va="bottom",
            fontsize="small",
        )
        for a in ax[0, :]
    ]

    ax[1, 0].plot(time, eod_smallchirp, color=chirpc, lw=0.5)
    ax[1, 1].plot(time, eod_longchirp, color=chirpc, lw=0.5)
    [a.set_xlim(ftraces_xlims) for a in ax[1, :]]
    [a.set_xticks([]) for a in ax[1, :]]
    [
        a.spines[s].set_visible(False)
        for a in ax[1, :]
        for s in ["top", "right", "bottom"]
    ]
    [a.set_yticks([-1, 0, 1]) for a in ax[1, :]]
    [a.set_ylim(-1.4, 1.4) for a in ax[1, :]]
    [sns.despine(ax=a, trim=True, bottom=True) for a in ax[1, :]]
    ax[1, 0].set_ylabel("$\\text{EOD}_{0}$")

    # add rectangle around the chirp area

    # padaxes
    [a.plot([70, 120], [-1.3, -1.3], color="black", lw=2) for a in ax[1, :]]
    [a.text(95, -1.45, "50$\\,$ms", ha="center", va="top") for a in ax[1, :]]

    ax[2, 0].plot(time, soundscape_smallchirp, color=chirpc, lw=0.5)
    ax[2, 1].plot(time, soundscape_longchirp, color=chirpc, lw=0.5)
    [a.set_xlim(-500, 500) for a in ax[2, :]]
    [a.set_xticks([]) for a in ax[2, :]]
    [a.set_yticks([-2, 0, 2]) for a in ax[2, :]]
    [a.set_ylim([-2.5, 2.5]) for a in ax[2, :]]
    [sns.despine(ax=a, trim=True, bottom=True) for a in ax[2, :]]
    [
        a.spines[s].set_visible(False)
        for a in ax[2, :]
        for s in ["top", "right", "bottom"]
    ]
    ax[2, 0].set_ylabel("$\\sum \\text{EOD}_{n}$")

    cmap = sns.color_palette("mako", as_cmap=True)
    spec_min = np.min([np.min(S_smallchirp), np.min(S_longchirp)])
    spec_max = np.max([np.max(S_smallchirp), np.max(S_longchirp)])
    vmin = spec_min + 0.25 * (spec_max - spec_min)
    vmax = spec_max

    ax[3, 0].imshow(
        S_smallchirp,
        extent=[time[0], time[-1], spec_freq[0], spec_freq[-1]],
        aspect="auto",
        cmap=cmap,
        interpolation="gaussian",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax[3, 1].imshow(
        S_longchirp,
        extent=[time[0], time[-1], spec_freq[0], spec_freq[-1]],
        aspect="auto",
        cmap=cmap,
        interpolation="gaussian",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    [a.set_xlim(-500, 500) for a in ax[3, :]]
    [a.set_ylim([600, 1200]) for a in ax[3, :]]
    [a.set_yticklabels([]) for a in ax[:, 1]]
    [a.set_yticks([600, 800, 1000, 1200]) for a in ax[3, :]]
    [a.set_xticks([]) for a in ax[3, :]]
    [
        a.spines[s].set_visible(False)
        for a in ax[3, :]
        for s in ["top", "right", "bottom"]
    ]
    ax[3, 0].set_ylabel("$\\text{EOD}f$ [Hz]")

    # rectangle shenanigans
    inset_kwargs = dict(
        color="k",
        lw=0.5,
        alpha=1,
        zorder=11,
        # ls="dotted",
    )
    ax[1, 0].add_patch(
        plt.Rectangle(
            (-20, -1.3),
            40,
            2.6,
            color=envc,
            lw=0,
            fill=True,
            alpha=0.2,
            zorder=-10,
        )
    )
    ax[1, 0].add_patch(
        plt.Rectangle(
            (-20, -1.3),
            40,
            2.6,
            fill=False,
            **inset_kwargs,
        )
    )

    # connect the rectangles
    ax[2, 0].add_patch(
        plt.Rectangle(
            (-20, -2.4),
            40,
            4.8,
            color=envc,
            lw=0,
            fill=True,
            alpha=0.2,
            zorder=-10,
        )
    )
    ax[2, 0].add_patch(
        plt.Rectangle(
            (-20, -2.4),
            40,
            4.8,
            fill=False,
            **inset_kwargs,
        )
    )
    # get coordinates to the upper corners
    conl = patches.ConnectionPatch(
        xyA=(-20, -1.3),
        xyB=(-20, 2.4),
        coordsA="data",
        coordsB="data",
        axesA=ax[1, 0],
        axesB=ax[2, 0],
        **inset_kwargs,
    )
    conr = patches.ConnectionPatch(
        xyA=(20, -1.3),
        xyB=(20, 2.4),
        coordsA="data",
        coordsB="data",
        axesA=ax[1, 0],
        axesB=ax[2, 0],
        **inset_kwargs,
    )
    fig.add_artist(conl)
    fig.add_artist(conr)

    ax[1, 1].add_patch(
        plt.Rectangle(
            (-60, -1.2),
            120,
            2.4,
            color=envc,
            lw=0,
            fill=True,
            alpha=0.2,
            zorder=-10,
        )
    )
    ax[1, 1].add_patch(
        plt.Rectangle(
            (-60, -1.2),
            120,
            2.4,
            fill=False,
            **inset_kwargs,
        )
    )
    ax[2, 1].add_patch(
        plt.Rectangle(
            (-60, -2.2),
            120,
            4.4,
            color=envc,
            lw=0,
            fill=True,
            alpha=0.2,
            zorder=-10,
        )
    )
    ax[2, 1].add_patch(
        plt.Rectangle(
            (-60, -2.2),
            120,
            4.4,
            fill=False,
            **inset_kwargs,
        )
    )
    # get coordinates to the upper corners
    conl = patches.ConnectionPatch(
        xyA=(-60, -1.2),
        xyB=(-60, 2.2),
        coordsA="data",
        coordsB="data",
        axesA=ax[1, 1],
        axesB=ax[2, 1],
        **inset_kwargs,
    )
    conr = patches.ConnectionPatch(
        xyA=(60, -1.2),
        xyB=(60, 2.2),
        coordsA="data",
        coordsB="data",
        axesA=ax[1, 1],
        axesB=ax[2, 1],
        **inset_kwargs,
    )
    fig.add_artist(conl)
    fig.add_artist(conr)

    # circle_annotate(ax[1, 0], (20, 1.3), (0.5, 0), "1")
    # circle_annotate(ax[2, 0], (20, 2.4), (1.05, 0), "1")
    # circle_annotate(ax[1, 1], (70, 1.2), (0.1, 0.02), "2")
    # circle_annotate(ax[2, 1], (70, 2.2), (1, 0), "2")

    [a.plot([300, 500], [-2.3, -2.3], color="k", lw=2, clip_on=False) for a in ax[2, :]]
    [
        a.text(
            400,
            -2.6,
            "200$\\,$ms",
            ha="center",
            va="top",
            color="k",
            clip_on=False,
        )
        for a in ax[2, :]
    ]

    # add padding between y axis and data for all axes
    for a in ax.flatten():
        sns.despine(ax=a, trim=True, offset=5, bottom=True)

    # annotate small chirp in left first subplot with arrow
    ax[0, 0].text(
        -40,
        1170,
        "Small chirp",
        ha="right",
        va="top",
        color="k",
        fontsize="small",
    )
    # add arrow
    ax[0, 0].annotate(
        "",
        xy=(-35, 1125),
        xycoords="data",
        xytext=(0, 800),
        textcoords="data",
        arrowprops=dict(arrowstyle="<-", connectionstyle="angle3"),
    )
    ax[0, 1].text(
        0.5,
        1.05,
        "Long chirp",
        ha="center",
        va="top",
        color="k",
        fontsize="small",
        transform=ax[0, 1].transAxes,
    )
    # ax[0, 1].text(
    #     -88,
    #     1250,
    #     "Long\nchirp",
    #     ha="right",
    #     va="top",
    #     color="k",
    #     fontsize="small",
    # )
    # add arrow
    # ax[0, 1].annotate(
    #     "",
    #     xy=(-85, 1125),
    #     xycoords="data",
    #     xytext=(-62.5, 1100),
    #     textcoords="data",
    #     arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.05"),
    # )

    # annotate nfft in spectrograms
    # [
    #     a.text(
    #         -250,
    #         1150,
    #         # f"nfft: {nfft}",
    #         ha="right",
    #         va="top",
    #         color="white",
    #         size="small",
    #     )
    #     for a in ax[3, :]
    # ]

    # add bounding boxes to chirps

    linestyle = {
        "edgecolor": "lightgrey",
        "facecolor": "none",
        "lw": 1,
        "zorder": 100,
    }

    ax[3, 0].add_patch(
        patches.Rectangle(
            (-80, 620),
            180,
            200,
            **linestyle,
        )
    )
    ax[3, 1].add_patch(
        patches.Rectangle(
            (-120, 620),
            250,
            530,
            **linestyle,
        )
    )

    left_axes = [ax[0, 0], ax[1, 0], ax[2, 0], ax[3, 0]]
    left_xoffsets = [-0.28 for _ in left_axes]
    right_axes = [ax[0, 1], ax[1, 1], ax[2, 1], ax[3, 1]]
    right_xoffsets = [-0.1 for _ in right_axes]
    yoffs = [1.2, 1.05, 1.05, 1.05]

    axes = np.concatenate((left_axes, right_axes))
    xoffsets = np.concatenate((left_xoffsets, right_xoffsets))
    yoffsets = np.concatenate((yoffs, yoffs))
    letters = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]
    letter_subplots(axes, letters=letters, xoffset=xoffsets, yoffset=yoffsets)
    fig.align_ylabels()

    fmts = [
        # "png",
        "pdf",
        # "svg",
    ]
    for fmt in fmts:
        print(f"Saving {targetpath.with_suffix(f'.{fmt}')} ...")
        fig.savefig(
            targetpath.with_suffix(f".{fmt}"),
            dpi=300,
        )
    plt.show()


if __name__ == "__main__":
    main()
