"""
https://matplotlib.org/1.2.1/examples/pylab_examples/boxplot_demo2.html
Thanks Josh Hemann for the example
"""
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [
    'Times New Roman', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'
]
rcParams['font.size'] = 16
import matplotlib.patches as mpatches
import statistics, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot(d={}, title='', dirname='results', ylabel='Score', log=False):
    title = title.title()
    labels = []
    for s in d.keys():
        labels.append(s.title())
    data = []
    # scales
    maxx = 1
    minn = 0
    for v in d.values():
        v = np.array(v)
        if not log and ylabel == 'Score':
            v = v * 1e-6
        data.append(v)
        if min(v) < minn:
            minn = min(v) - 0.3
        if max(v) > maxx:
            maxx = max(v) + 0.3

    fig = plt.figure(figsize=(10, 6))
    fig.canvas.set_window_title(title)
    ax1 = fig.add_subplot(111)

    plt.ylim([minn, maxx])
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)
    bp = plt.boxplot(data, notch=0, sym='+', vert=1, whis=1.5, labels=labels)
    if log:
        plt.yscale('symlog')

    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(
        True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title(title)
    # ax1.set_xlabel('Algorithms')
    if not log and ylabel == 'Score':
        ax1.set_ylabel(ylabel + ' [million]')
    else:
        ax1.set_ylabel(ylabel)

    # handles = [
    #     mpatches.Patch(color=TABLEAU20[i], label=names[i])
    #     for i in range(0, len(names))
    # ]
    # plt.legend(loc=2, handles=handles)

    # Now fill the boxes with desired colors
    # boxColors = ['darkkhaki', 'royalblue']
    # numBoxes = len(data)
    # medians = list(map(lambda x: statistics.median(x), data))

    # Set the axes ranges and axes labels
    # ax1.set_xlim(0.5, numBoxes + 0.5)

    # top = 40
    # bottom = -5
    # ax1.set_ylim(bottom, top)
    # xtickNames = plt.setp(ax1, xticklabels=np.repeat(randomDists, 2))
    # plt.setp(xtickNames, rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    # pos = np.arange(numBoxes) + 1
    # upperLabels = [str(np.round(s, 2)) for s in medians]
    # weights = ['bold', 'semibold']
    # for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
    #     k = tick % 2
    #     ax1.text(
    #         pos[tick],
    #         top - (top * 0.05),
    #         upperLabels[tick],
    #         horizontalalignment='center',
    #         size='x-small',
    #         weight=weights[k],
    #         color=boxColors[k])

    # Finally, add a basic legend
    # plt.figtext(
    #     0.80,
    #     0.08,
    #     ' Random Numbers',
    #     color='black',
    #     # weight='roman',
    #     # size='x-small'
    # )
    # plt.figtext(
    #     0.80,
    #     0.045,
    #     'IID Bootstrap Resample',
    #     color='black',
    #     # weight='roman',
    #     # size='x-small'
    # )
    # plt.show()
    if log:
        plt.savefig(dirname + '/' + title + '-log.png')
    else:
        plt.savefig(dirname + '/' + title + '.png')
    plt.close()


TABLEAU20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199,
                                                                 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(TABLEAU20)):
    r, g, b = TABLEAU20[i]
    TABLEAU20[i] = (r / 255., g / 255., b / 255.)
