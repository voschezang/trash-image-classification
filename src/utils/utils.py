"""
Utilities for data processing
"""
import os, sys, time, datetime, texttable, copy, random
import numpy as np, statistics, collections
from scipy import stats
import config


def time_function(f, arg):
    t = time.time()
    results = f(arg)
    dt = time.time() - t
    print("time was", dt, t, time.time() - t)
    return results, dt


def random_skewed(lowest=0, highest=1, skew=1):
    # random value from a skewed distribution:
    #   rand^power * max_range
    #   power < 1 -> higher probability for high values
    #   power > 1 -> higher probability for low values
    # random.seed(seed) TODO
    if lowest >= 0:
        return lowest + random.random()**skew * (highest - lowest)
    else:
        # (kind of) symmetric distribution
        if random.random() < 0.5:
            # positive outcome
            return random.random()**skew * highest
        else:
            # negative outcome
            return random.random()**skew * lowest


def statistical_tests(data={},
                      name='iterative-scores',
                      print_results=True,
                      base=False):
    """ Statistical test (ttest)
    :data :: { 'name': [score] }
    :s = statistic
    :p = p-value, the probability that observed outcome was random
    :a = significance level
    """
    if config.main_: print(" - \n" + name.title() + '\n')

    if not base:
        # start with an arbitrary k,v
        best, best_scores = data.popitem()
        best_score = max(best_scores)
        # search for better values
        for k, scores in data.items():
            if max(scores) > best_score:
                # put the old best back in the dict
                data[best] = best_scores
                best = k
                best_scores = data.pop(best)
    else:
        best = base
        best_scores = data.pop(best)

    # 'best' will be the default/baseline to compare the other algorithms to
    results_dict = {}  # results = []
    for k, scores in data.items():
        compare([best, k], best_scores, scores, results_dict, print_results)
        # results.append(compare([best, k], best_scores, scores))
        # if print_results:
        #     for item in results[-1]:
        #         print(item)
    results = collections.OrderedDict(
        sorted(results_dict.items(), key=lambda t: t[0]))
    return results, best


def compare(names=[], a=[], b=[], d={}, print_results=True):
    """ Compare two lists of numbers using statistical test
    :a, b = lists with data
    :names = names of a,b
    :d = dict to add results to
    :s = statistic
    :p = p-value, the probability that observed outcome was random
    :a = significance level

    """
    print("\n\n\n\n\n\n")
    print(names, a, b)
    alpha = 0.05
    mean_a, median_a, min_a, max_a = summary(a)
    mean_b, median_b, min_b, max_b = summary(b)
    best_mean, _ = max(enumerate([mean_a, mean_b]), key=lambda x: x[1])
    best_median, _ = max(enumerate([median_a, median_b]), key=lambda x: x[1])
    if not best_mean == best_median:
        # a and b seem to be equal
        if config.result_:
            print('RESULT - best mean and median differ', best_mean,
                  best_median)
        if config.result_: print('RESULT', 'mean', best_mean, best_median)

    # Significance test
    result, s, p = ttest(alpha, a, b)
    if result:
        if config.result_: print('  winner', names[best_mean])

    # Make table
    table = texttable.Texttable()
    rows = [['Algorithm', 'Mean', 'Median', 'Min', 'Max']]
    rows.append([names[0].title(), mean_a, median_a, min_a, max_a])
    rows.append([names[1].title(), mean_b, median_b, min_b, max_b])
    rows.append([
        names[0] + ' > ' + names[1], mean_a > mean_b, median_a > median_b,
        min_a > min_b, max_a > max_b
    ])
    for r in rows:
        print(r)
    rows.append(['statistic', s, 'p-value', p, result])
    table.add_rows(rows)
    string = table.draw() + ' 1 = True \n'
    if print_results:
        print(string)
        # print(table.draw(), '1 = True \n')

    d['0'] = ['mean', 'median', 'min', 'max']
    d['0 ' + names[0]] = [mean_a, median_a, min_a, max_a]  # may be overwritten
    d['0 ' + names[1]] = [mean_b, median_b, min_b, max_b]
    name1 = names[0] + ' > ' + names[1]
    d[name1] = [
        mean_a > mean_b, median_a > median_b, min_a > min_b, max_a > max_b
    ]
    name2 = names[0] + ' vs ' + names[1]
    d[name2] = ['statistic, p-value', s, p, result]

    # d[name + ' - ' + names[0]] = [mean_a, median_a, min_a, max_a]
    # d[name + ' - ' + names[1]] = [mean_b, median_b, min_b, max_b]

    return (d, result, names[best_mean])


def summary(a=[]):
    return statistics.mean(a), statistics.median(a), min(a), max(a)


def ttest(alpha=0.05, a=[], b=[]):
    s, p = stats.ttest_ind(a, b)
    if p < alpha:
        result = True
        if config.result_: print('RESULT - significant difference found')
    else:
        result = False
        if config.result_: print('RESULT - NO significant difference found')
    return result, s, p


def dict_to_table(data={}, durations={}, print_results=False, txt=''):
    # data :: { 'name': [score] }
    # durations :: { 'name': [time] }
    result_dict = {0: ['best score', 'duration']}
    table = texttable.Texttable()
    rows = [['Algorithm', 'Score', 'Duration']]
    for alg, scores in data.items():
        # select best score
        i, score = max(enumerate(scores), key=lambda x: x[1])
        # add row: name, score, durations for that score
        print(alg, i, score)
        print(len(data[alg]), len(durations[alg]))
        t = durations[alg][i]
        rows.append([alg, score, t])
        result_dict[alg] = [score, t]
    table.add_rows(rows)
    if print_results:
        print(txt)
        print(table.draw())
    return result_dict


def extract_final_t(data):
    # data :: { 'name': { 'observation n': [score] } }
    results = collections.defaultdict(list)
    for name, observation_dict in data.items():
        for n, score_list in observation_dict.items():
            results[name].append(score_list[-1])
    return results
