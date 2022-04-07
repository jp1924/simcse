# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
STS-{2012,2013,2014,2015,2016} (unsupervised) and
STS-benchmark (supervised) tasks
'''

from __future__ import absolute_import, division, unicode_literals

import os
import io
import numpy as np
import logging

from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cityblock, euclidean
from .utils import cosine
from .sick import SICKEval
import scipy.special as special
from datasets import load_dataset, concatenate_datasets
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
import torch
import pandas as pd
class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = params.similarity
        else:  # Default similarity is cosine
            self.cosine_similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))
        return prepare(params, self.samples)

    def pearsonr_test(self, x, y):# ================ custom ================
        n = len(x)
        if n != len(y):
            raise ValueError('x and y must have the same length.')

        if n < 2:
            raise ValueError('x and y must have length at least 2.')

        x = np.asarray(x)
        y = np.asarray(y)

        # If an input is constant, the correlation coefficient is not defined.
        if (x == x[0]).all() or (y == y[0]).all():
            warnings.warn(PearsonRConstantInputWarning())
            return np.nan, np.nan

        # dtype is the data type for the calculations.  This expression ensures
        # that the data type is at least 64 bit floating point.  It might have
        # more precision if the input is, for example, np.longdouble.
        dtype = type(1.0 + x[0] + y[0])

        if n == 2:
            return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0])), 1.0

        xmean = x.mean(dtype=dtype)
        ymean = y.mean(dtype=dtype)

        # By using `astype(dtype)`, we ensure that the intermediate calculations
        # use at least 64 bit floating point.
        xm = x.astype(dtype) - xmean
        ym = y.astype(dtype) - ymean

        # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
        # scipy.linalg.norm(xm) does not overflow if xm is, for example,
        # [-5e210, 5e210, 3e200, -3e200]
        normxm = np.linalg.norm(xm)
        normym = np.linalg.norm(ym)

        threshold = 1e-13
        if normxm < threshold*abs(xmean) or normym < threshold*abs(ymean):
            # If all the values in x (likewise y) are very close to the mean,
            # the loss of precision that occurs in the subtraction xm = x - xmean
            # might result in large errors in r.
            warnings.warn(PearsonRNearConstantInputWarning())

        r = np.dot(xm/normxm, ym/normym)

        # Presumably, if abs(r) > 1, then it is only some small artifact of
        # floating point arithmetic.
        r = max(min(r, 1.0), -1.0)

        # As explained in the docstring, the p-value can be computed as
        #     p = 2*dist.cdf(-abs(r))
        # where dist is the beta distribution on [-1, 1] with shape parameters
        # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
        # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
        # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
        # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
        # to avoid a TypeError raised by btdtr when r is higher precision.)
        ab = n/2 - 1
        prob = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(r))))

        return r, prob

    def run(self, params, batcher):
        results = {}
        all_sys_scores = []
        all_gs_scores = []
        for dataset in self.datasets: # 보통 [A, B, C] 이런 형식으로 데스크가 들어오게 된다.
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]

            if type(self).__name__.lower() == "korsts" or \
               type(self).__name__.lower() == "kluests":
                enc1 = []
                enc2 = []
                for ii in range(0, len(gs_scores), params.batch_size):
                    batch1 = input1[ii:ii + params.batch_size]
                    batch2 = input2[ii:ii + params.batch_size]

                    # we assume get_batch already throws out the faulty ones
                    if len(batch1) == len(batch2) and len(batch1) > 0:
                        enc1.append(batcher(params, batch1))
                        enc2.append(batcher(params, batch2))

                enc1 = torch.concat(enc1)
                enc2 = torch.concat(enc2)

                name = type(self).__name__.lower().split("sts")[0]

                cosine_scores = (1 - paired_cosine_distances(enc1, enc2)).tolist()
                manhattan_distances = (-paired_manhattan_distances(enc1, enc2)).tolist()
                euclidean_distances = (-paired_euclidean_distances(enc1, enc2)).tolist()
                #dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(enc1.detach().tolist(), enc2.detach().tolist())]

                pearson_test, _ = self.pearsonr_test(gs_scores, cosine_scores)

                pearson_cosine, _ = pearsonr(gs_scores, cosine_scores)
                spearman_cosine, _ = spearmanr(gs_scores, cosine_scores)

                pearson_manhattan, _ = pearsonr(gs_scores, manhattan_distances)
                spearman_manhattan, _ = spearmanr(gs_scores, manhattan_distances)

                pearson_euclidean, _ = pearsonr(gs_scores, euclidean_distances)
                spearman_euclidean, _ = spearmanr(gs_scores, euclidean_distances)

                # pearson_dot, _ = pearsonr(gs_scores, dot_products)
                # spearman_dot, _ = spearmanr(gs_scores, dot_products)

                results = \
                {f'{name}_pear_cos': f"{pearson_cosine:.4f}",
                 f'{name}_pear_man': f"{pearson_manhattan:.4f}",
                 f'{name}_pear_euclid': f"{pearson_euclidean:.4f}",
                #  f'{name}_pear_dot': f"{pearson_dot:.4f}",
                 f'{name}_spear_cos': f"{spearman_cosine:.4f}",
                 f'{name}_spear_man': f"{spearman_manhattan:.4f}",
                 f'{name}_spear_euclid': f"{spearman_euclidean:.4f}",
                #  f'{name}_spear_dot': f"{spearman_dot:.4f}",
                  'nsamples': len(gs_scores)}
                    
                return results, input1, input2, gs_scores, cosine_scores, manhattan_distances, euclidean_distances#, dot_products
            
            cosine_scores = []
            euclidean_scores = []
            manhattan_scores = []
            dot_scores = []

            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        cosine_result = self.cosine_similarity(enc1[kk], enc2[kk])

                        euclidean_result = euclidean(enc1[kk], enc2[kk])
                        manhattan_result = cityblock(enc1[kk], enc2[kk])

                        cosine_scores.append(cosine_result)
                        euclidean_scores.append(euclidean_result)
                        manhattan_scores.append(manhattan_result)

                        # sys_score = self.similarity(enc1[kk], enc2[kk])
                        # sys_scores.append(sys_score)
            
            results[dataset] = {'cosine_pearsonr': pearsonr(cosine_scores, gs_scores),
                                'cosine_spearmanr': spearmanr(cosine_scores, gs_scores),
                                'euclidean_peasonr':pearsonr(euclidean_scores, gs_scores),
                                'euclidean_spearmanr':spearmanr(euclidean_scores, gs_scores),
                                'manhattan_pearsonr':pearsonr(manhattan_scores, gs_scores),
                                'manhattan_spearmanr':spearmanr(manhattan_scores, gs_scores),
                                'nsamples': len(gs_scores)}

            return results

            all_sys_scores.extend(sys_scores)
            all_gs_scores.extend(gs_scores)
            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.info('%s : pearson = %.4f, spearman = %.4f' %
                        (dataset, results[dataset]['pearson'][0],
                        results[dataset]['spearman'][0]))

            weights = [results[dset]['nsamples'] for dset in results.keys()]
            list_prs = np.array([results[dset]['pearson'][0] for
                                dset in results.keys()])
            list_spr = np.array([results[dset]['spearman'][0] for
                                dset in results.keys()])

            avg_pearson = np.average(list_prs)
            avg_spearman = np.average(list_spr)
            wavg_pearson = np.average(list_prs, weights=weights)
            wavg_spearman = np.average(list_spr, weights=weights)
            all_pearson = pearsonr(all_sys_scores, all_gs_scores)
            all_spearman = spearmanr(all_sys_scores, all_gs_scores)

            results['all'] = {'pearson': {'all': all_pearson[0],
                                        'mean': avg_pearson,
                                        'wmean': wavg_pearson},
                            'spearman': {'all': all_spearman[0],
                                        'mean': avg_spearman,
                                        'wmean': wavg_spearman}}
            logging.debug('ALL : Pearson = %.4f, \
                Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
            logging.debug('ALL (weighted average) : Pearson = %.4f, \
                Spearman = %.4f' % (wavg_pearson, wavg_spearman))
            logging.debug('ALL (average) : Pearson = %.4f, \
                Spearman = %.4f\n' % (avg_pearson, avg_spearman))

            return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)


class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class STSBenchmarkFinetune(SICKEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        train = self.loadFile(os.path.join(task_path, 'sts-train.csv'))
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def loadFile(self, fpath):
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sick_data['X_A'].append(text[5].split())
                sick_data['X_B'].append(text[6].split())
                sick_data['y'].append(text[4])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        return sick_data
        
class SICKRelatednessEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SICKRelatedness*****\n\n')
        self.seed = seed
        self.samples = []
        train = self.loadFile(os.path.join(task_path, 'SICK_train.txt'))
        dev = self.loadFile(os.path.join(task_path, 'SICK_trial.txt'))
        test = self.loadFile(os.path.join(task_path, 'SICK_test_annotated.txt'))
        self.datasets = ['train', 'dev', 'test']
        self.data = {'train': train, 'dev': dev, 'test': test}
    
    def loadFile(self, fpath):
        skipFirstLine = True
        sick_data = {'X_A': [], 'X_B': [], 'y': []}
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if skipFirstLine:
                    skipFirstLine = False
                else:
                    text = line.strip().split('\t')
                    sick_data['X_A'].append(text[1].split())
                    sick_data['X_B'].append(text[2].split())
                    sick_data['y'].append(text[3])

        sick_data['y'] = [float(s) for s in sick_data['y']]
        self.samples += sick_data['X_A'] + sick_data["X_B"]
        return (sick_data['X_A'], sick_data["X_B"], sick_data['y'])

class KorSTS(STSEval):
    def __init__(self):
        STS_data = load_dataset("kor_nlu", "sts", ignore_verifications=True)

        split_data = [STS_data[name] for name in STS_data]
        pd_STS = concatenate_datasets(split_data).to_pandas().dropna(axis=0)


        sel_idx = np.where((pd_STS["sentence2"].str.len()<=200)&(pd_STS["sentence1"].str.len()<=200), True, False)
        pd_STS = pd_STS.loc[sel_idx, ["sentence1", "sentence2", "score"]]
        pd_STS = pd_STS.loc[1:, ["sentence1", "sentence2", "score"]]

        gs_scores = pd_STS["score"].values.tolist()[:]
        sentence1 = sum(pd_STS.loc[:,["sentence1"]].values.tolist(), [])[:]
        sentence2 = sum(pd_STS.loc[:,["sentence2"]].values.tolist(), [])[:]

        self.datasets = ["dev"]
        self.data = {"dev": [sentence1, sentence2, gs_scores]}
        self.samples = sentence1

class KLUESTS(STSEval):
    def __init__(self):
        use_dev = False
        if use_dev:
            pd_STS = load_dataset("klue", "sts", ignore_verifications=True)["validation"].to_pandas()
            pd_STS = pd_STS.dropna(axis=0).loc[:,["sentence1", "sentence2", "labels"]]
            sel_gs_scores = lambda x: x["label"]
            pd_STS["labels"] = pd_STS["labels"].apply(func=sel_gs_scores)
            gs_scores = pd_STS["labels"].values.tolist()[:]

            sentence1 = sum(pd_STS.loc[:,["sentence1"]].values.tolist(), [])[:]
            sentence2 = sum(pd_STS.loc[:,["sentence2"]].values.tolist(), [])[:]
            
            self.datasets = ["dev"]
            self.data = {"dev": [sentence1, sentence2, gs_scores]}
            self.samples = sentence1
        else:
            STS_data = load_dataset("klue", "sts", ignore_verifications=True)

            split_data = [STS_data[name] for name in STS_data]
            pd_STS = concatenate_datasets(split_data).to_pandas()
            pd_STS = pd_STS.dropna(axis=0).loc[:,["sentence1", "sentence2", "labels"]]

            sel_gs_scores = lambda x: x["label"]
            pd_STS["labels"] = pd_STS["labels"].apply(func=sel_gs_scores)
            gs_scores = pd_STS["labels"].values.tolist()[:]

            sentence1 = sum(pd_STS.loc[:,["sentence1"]].values.tolist(), [])[:]
            sentence2 = sum(pd_STS.loc[:,["sentence2"]].values.tolist(), [])[:]
            
            self.datasets = ["dev"]
            self.data = {"dev": [sentence1, sentence2, gs_scores]}
            self.samples = sentence1