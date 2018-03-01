import os, sys, time, datetime, pandas, statistics, collections, shutil, re
from utils import plot
from utils import boxplot

### --------------------------------------------------------------------
### I/O
### --------------------------------------------------------------------


def replace_special_chars(string, char='-'):
    return re.sub('[^A-Za-z0-9]+', char, string)


def make_dir(name="results_abc", post='', timestamp=False):
    name = replace_special_chars(name)
    post = replace_special_chars(post)
    if not os.path.isdir(name):
        os.mkdir(name)
    if timestamp:
        dirname = name + "/" + shortTimeStamp() + ' ' + post
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname
    return name


def make_subdir(parent, name="img"):
    name = parent + "/" + name
    if not os.path.isdir(parent):
        print("error parent")
        os.mkdir(parent)
    if not os.path.isdir(name):
        os.mkdir(name)
    return name


def shortTimeStamp(s=5):
    n = 1000 * 1000  # larger n -> larger timestamp
    t = time.time()
    sub = int(t / n)
    t = str(round(t - sub * n, s))
    date = datetime.datetime.date(datetime.datetime.now())
    return str(date) + "_" + str(t)


def unique_dir(name, post=''):
    # name = 'iterative' | 'constructive'
    # generate unique dir
    dirname = make_dir('results-' + name, timestamp=True, post=post)
    img_dir = make_subdir(dirname, 'img')
    return dirname, img_dir


def save_configs(name, dirname, img_dir, worlds):
    # params, scenario should be equal for all worlds
    world = list(worlds.values())[-1]
    print_dict(dirname, world.params, "params")
    print_dict(dirname, world.scenario, "scenario")
    # print_dict(dirname, world.info_dict(), "world.info")

    for alg, w in worlds.items():
        print_dict(dirname, world.info_dict(), "world-info-" + alg)

    # plots
    plot.worlds(worlds, img_dir)


def save_simple(name, data, durations, scenario_name):
    dirname, img_dir = unique_dir(name, scenario_name)
    for alg, scores in data.items():
        save_to_csv(dirname, 'algorithm-' + alg, scores)
        plot.score_during_world({alg: scores}, alg, img_dir)
    save_to_csv(dirname, 'durations', durations)
    return dirname, img_dir


def save_constructive(name, data, durations, scenario_name):
    dirname, img_dir = unique_dir(name, scenario_name)
    save_to_csv(dirname, 'data', data)
    save_to_csv(dirname, 'durations', durations)
    boxplot.plot(data, 'Scores', img_dir)
    boxplot.plot(data, 'Scores', img_dir, log=True)
    boxplot.plot(durations, 'Durations', img_dir, ylabel='Duration [s]')
    boxplot.plot(
        durations, 'Durations', img_dir, ylabel='Duration [s]', log=True)
    return dirname, img_dir


def save_iterative(name, data, data_final_t, durations, scenario_name):
    # data :: { 'name': { 'observation n': [score] } }
    # data_final_t :: { 'name': [score] }
    dirname, img_dir = unique_dir(name, scenario_name)
    # TODO save and reread results; allow for independent data processing

    save_to_csv(dirname, 'data-final_t', data_final_t)
    save_to_csv(dirname, 'durations', durations)
    boxplot.plot(data_final_t, 'Scores final iteration', img_dir)
    boxplot.plot(data_final_t, 'Scores final iteration', img_dir, log=True)
    boxplot.plot(durations, 'Durations', img_dir, ylabel='Duration [s]')
    boxplot.plot(
        durations, 'Durations', img_dir, ylabel='Duration [s]', log=True)

    for algorithm, obs in data.items():
        print(algorithm)
        # obs within an algorithm have an equal length
        save_to_csv(dirname, 'algorithm-' + algorithm + '-raw', obs)
        d4 = summarize4(obs)
        save_to_csv(dirname, 'algorithm-' + algorithm + '-stats', d4)
        plot.score_during_world(d4, algorithm, img_dir)

    return dirname, img_dir


def print_dict(dirname="", d={}, name="text"):
    if not dirname == "":
        dirname += "/"
    name += ".txt"
    with open(dirname + "0_" + name, "w") as text_file:
        print(name + "\n", file=text_file)
        for k, v in d.items():
            # print(f"{k}:{v}", file=text_file) # pythonw, python3
            print('{:s}, {:s}'.format(str(k), str(v)), file=text_file)


# def print_tables(dirname="", data='', name="text"):
#     name += ".txt"
#     with open(dirname + "0_" + name, "w") as text_file:
#         print(name + "\n", file=text_file)
#         for table in data:
#             print('-----------')
#             print(table)
#             print('{:s}'.format(str(table)), file=text_file)
#             print('\n \n', file=text_file)


def save_to_csv(dirname, name, data):
    # panda df requires data to be NOT of type {key: scalar}
    filename = dirname + "/" + name + ".csv"
    df = pandas.DataFrame(data=data)
    df.to_csv(filename, sep=',', index=False)
    # mkdir filename
    # for k in d.keys(): gen png
    return filename


def summarize4(obs):
    # compute min,max,mean,median
    # obs :: { 'observation n': [scores over time] }
    # d :: {statistic:stat(scores over time)}
    d = collections.defaultdict(list)
    # pick an arbitrary observation
    n, values = obs.popitem()
    # put it back
    obs[n] = values
    for t, _ in enumerate(values):
        ls = list(map(lambda k: obs[k][t], obs.keys()))
        d['mean'].append(statistics.mean(ls))
        d['median'].append(statistics.median(ls))
        d['max.'].append(max(ls))
        d['min.'].append(min(ls))
    return d
