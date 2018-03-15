"""
File with functions to plot results
e.g. barcharts, histograms
"""

import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = [
    'Times New Roman', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'
]
rcParams['font.size'] = 14
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# import scenarios
from utils import utils, io, plot

# These are the "Tableau 20" colors as RGB.
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

### --------------------------------------------------------------------
### Plot Images
### --------------------------------------------------------------------


def multiple(data):
    # data :: [matrix]
    n = len(data)
    plt.figure(figsize=(12, 12 * n))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(data[i], cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


### --------------------------------------------------------------------
### Plot line
### --------------------------------------------------------------------


def single_line(data, name='plot', gui=False, plotfolder=False):
    name = name.title()
    fig, ax = plt.subplots()
    plt.plot(data)
    ax.set_ylim(bottom=0)

    plt.ylabel('Score')
    io.make_dir('results')
    if plotfolder:
        utils.make_dir('results/plots')
        plt.savefig('results/plots/' + name + '.png')
    else:
        plt.savefig('results/' + name + '.png')
    plt.close()


def graph(data, gui=False, name="graph", legend=[]):
    name = name.title()
    fig, ax = plt.subplots()
    ax.imshow(data, interpolation='nearest')

    fig.canvas.set_window_title(name + '...')
    fig.set_title(name)
    fig.set_xlabel('Time [iterations]')
    fig.set_ylabel('Score')

    io.make_dir('results')
    plt.savefig('results/' + name + 'graph.png')
    if gui:
        plt.show()
    plt.close()


def score_during_world(data={}, title='graph', dirname='results'):
    plot_dict(data, title, dirname)
    plot_dict(data, title, dirname, log=True)
    plot_dict(data, title, dirname, relative=True)
    plot_dict(data, title, dirname, relative=True, log=True)


def score_over_iterations(data=[], title='graph', dirname='results'):
    plot_dict({title: data}, title, dirname)
    plot_dict({title: data}, title, dirname, log=True)
    plot_dict({title: data}, title, dirname, relative=True)
    plot_dict({title: data}, title, dirname, relative=True, log=True)


def plot_dict(d, name, dn, log=False, relative=False):
    # all values are divided by 1e6
    name = name.title()
    index = 0
    names = []
    for s in d.keys():
        names.append(s.title())
    n = len(names)

    maxx = 1
    minn = 0
    for k, v in d.items():
        v = np.array(v)
        if not log:
            v = v * 1e-6
        # x = [0,1,2...]
        # y axis limits
        if min(v) < minn:
            minn = min(v) - 0.3
        if max(v) > maxx:
            maxx = max(v) + 0.3
        plt.plot(v, lw=2, color=TABLEAU20[index])

        index += 1

    if not relative:
        plt.ylim([minn, maxx])
    if log:
        plt.yscale('symlog')

    handles = [
        mpatches.Patch(color=TABLEAU20[i], label=names[i])
        for i in range(0, n)
    ]
    # legend inside subplot

    plt.legend(loc=4, handles=handles)
    # legend on top of subplot
    # plt.legend(
    #     handles=handles,
    #     bbox_to_anchor=(0., 1.02, 1., .102),
    #     loc=3,
    #     ncol=2,
    #     mode="expand",
    #     borderaxespad=0.)

    plt.title(name)
    plt.xlabel('Time [iterations]')
    if not log:
        plt.ylabel('Score [million]')
    else:
        plt.ylabel('Score')

    # plt.text(50, 12, "Snake mean genes over time", fontsize=17, ha="center")
    io.make_dir('results')
    if log:
        name += '-log'
    if relative:
        name += '-rel'
    plt.savefig(dn + '/' + name + '.png')
    plt.clf()
    plt.close()


### --------------------------------------------------------------------
### Plot Grid/World/2D-array
### --------------------------------------------------------------------


def worlds(worlds={}, dirname='', pre=''):
    print("--- Plot world ---")
    for name, world in worlds.items():
        img = world.to_image(scenarios.colors, scenarios.default_color)
        plot_world(img, False, pre + name, custom_dir=dirname)
        larger_img = img.resize(2)
        plot_world(larger_img, False, pre + name + '-x4', custom_dir=dirname)


def large_world(world, dirname='results', amt=2):
    img = world.to_image(scenarios.colors, scenarios.default_color)
    larger_img = img.resize(amt)
    plot_world(larger_img, False, 'large', dirname)


def plot_world(data, gui=False, name="graph", custom_dir=False):
    name = name.title()
    # data # :: 2d array np
    # data = 10*np.random.rand(5, 3)
    fig, ax = plt.subplots()
    ax.imshow(data, interpolation='nearest')
    # numrows, numcols = data.shape
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title(name)

    # legend :: [ (color, 'name')]
    # red_patch = mpatches.Patch(color='red', label='The red data')
    # plt.legend()

    # axis titles (with scale) (e.g. 1:100 m)
    io.make_dir('results')
    plt.savefig('results/' + name + '-world.png')
    if custom_dir:
        plt.savefig(custom_dir + '/' + name + '-world.png')
    if gui:
        plt.show()
    else:
        plt.close()
