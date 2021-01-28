from __future__ import division
from math import log,sqrt
from scipy.stats import spearmanr, pearsonr
import operator

# Hack to shut up deprecation warning wrt something in the stemmer
import sys, importlib
sys.modules['sklearn.externals.six'] = importlib.import_module('six')

from nltk.stem import *
from nltk.stem.porter import *
import matplotlib.pyplot as plt

from load_map import *

from collections import defaultdict

STEMMER = PorterStemmer()

# helper function to get the count of a word (string)
def w_count(word):
  return o_counts[word2wid[word]]

def tw_stemmer(word):
  '''Stems the word using Porter stemmer, unless it is a
  username (starts with @).  If so, returns the word unchanged.
  :type word: str
  :param word: the word to be stemmed
  :rtype: str
  :return: the stemmed word
  '''
  if word[0] == '@': #don't stem these
    return word
  else:
    return STEMMER.stem(word)

def PMI(c_xy, c_x, c_y, N):
  '''Compute the pointwise mutual information using cooccurrence counts.
  :type c_xy: int
  :type c_x: int
  :type c_y: int
  :type N: int
  :param c_xy: coocurrence count of x and y
  :param c_x: occurrence count of x
  :param c_y: occurrence count of y
  :param N: total observation count
  :rtype: float
  :return: the pmi value
  '''
  return log(c_xy*N/c_x/c_y, 2);

#Do a simple error check using value computed by hand
if(PMI(2,4,3,12) != 1): # these numbers are from our y,z example
    print("Warning: PMI is incorrectly defined")
else:
    print("PMI check passed")

def cos_sim(v0,v1):
  '''Compute the cosine similarity between two sparse vectors.
  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  :return: cosine between v0 and v1
  '''
  # We recommend that you store the sparse vectors as dictionaries
  # with keys giving the indices of the non-zero entries, and values
  # giving the values at those dimensions.

  len0 = sqrt(sum([v0[x]*v0[x] for x in v0]));
  len1 = sqrt(sum([v1[x]*v1[x] for x in v1]));

  return (sum([v1[x]*v0[x] for x in v0]))/len0/len1;


def create_ppmi_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using PPMI.
    These should be sparse vectors.
    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        vector = defaultdict(float);
        for wid1 in co_counts[wid0]:
            c_x = o_counts[wid0];
            c_y = o_counts[wid1];
            c_xy = co_counts[wid0][wid1];
            PPMI = PMI(c_xy,c_x,c_y,tot_count);
            if PPMI > 0:
                vector[wid1] = PPMI;

        vectors[wid0] = vector;
    return vectors

def read_counts(filename, wids):
  '''Reads the counts from file. It returns counts for all words, but to
  save memory it only returns cooccurrence counts for the words
  whose ids are listed in wids.
  :type filename: string
  :type wids: list
  :param filename: where to read info from
  :param wids: a list of word ids
  :returns: occurence counts, cooccurence counts, and tot number of observations
  '''
  o_counts = {} # Occurence counts
  co_counts = {} # Cooccurence counts
  fp = open(filename)
  N = float(next(fp))
  for line in fp:
    line = line.strip().split("\t")
    wid0 = int(line[0])
    o_counts[wid0] = int(line[1])
    if(wid0 in wids):
        co_counts[wid0] = dict([int(y) for y in x.split(" ")] for x in line[2:])
  return (o_counts, co_counts, N)

def print_sorted_pairs(similarities, o_counts, first=0, last=100):
  '''Sorts the pairs of words by their similarity scores and prints
  out the sorted list from index first to last, along with the
  counts of each word in each pair.
  :type similarities: dict
  :type o_counts: dict
  :type first: int
  :type last: int
  :param similarities: the word id pairs (keys) with similarity scores (values)
  :param o_counts: the counts of each word id
  :param first: index to start printing from
  :param last: index to stop printing
  :return: none
  '''
  if first < 0: last = len(similarities)
  count = 1
  for pair in sorted(similarities.keys(), key=lambda x: similarities[x], reverse = True)[first:last]:
    word_pair = (wid2word[pair[0]], wid2word[pair[1]])
    # print("{:.2f}\t{:30}\t{}\t{}".format(similarities[pair],str(word_pair),
    #                                      o_counts[pair[0]],o_counts[pair[1]]))

    print("{:30}\t&{:.2f}&{}\\\\\n\\hline".format(str(word_pair),similarities[pair],count))
    count+=1

def freq_v_sim(sims):
  xs = []
  ys = []
  for pair in sims.items():
    ys.append(pair[1])
    c0 = o_counts[pair[0][0]]
    c1 = o_counts[pair[0][1]]
    xs.append(min(c0,c1))
  plt.clf() # clear previous plots (if any)
  plt.xscale('log') #set x axis to log scale. Must do *before* creating plot
  plt.plot(xs, ys, 'k.') # create the scatter plot
  plt.xlabel('Min Freq')
  plt.ylabel('Similarity')
  print("Freq vs Similarity Spearman correlation = {:.2f}".format(spearmanr(xs,ys)[0]))
#  plt.show() #display the set of plots

def make_pairs(items):
  '''Takes a list of items and creates a list of the unique pairs
  with each pair sorted, so that if (a, b) is a pair, (b, a) is not
  also included. Self-pairs (a, a) are also not included.
  :type items: list
  :param items: the list to pair up
  :return: list of pairs
  '''
  return [(x, y) for x in items for y in items if x < y]

def create_ttest_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using T-test.
    These should be sparse vectors.
    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    # <todo>
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        vector = defaultdict(float);
        for wid1 in co_counts[wid0]:
            c_x = o_counts[wid0];
            c_y = o_counts[wid1];
            c_xy = co_counts[wid0][wid1];
            value = (N*c_xy-c_x*c_y)/(N*sqrt(c_x*c_y));
            vector[wid1] = value;
        vectors[wid0] = vector;
    return vectors

def create_prob_vectors(wids, o_counts, co_counts, tot_count):
    '''Creates context vectors for the words in wids, using Prob.
    These should be sparse vectors.
    :type wids: list of int
    :type o_counts: dict
    :type co_counts: dict of dict
    :type tot_count: int
    :param wids: the ids of the words to make vectors for
    :param o_counts: the counts of each word (indexed by id)
    :param co_counts: the cooccurrence counts of each word pair (indexed by ids)
    :param tot_count: the total number of observations
    :rtype: dict
    # <todo>
    :return: the context vectors, indexed by word id
    '''
    vectors = {}
    for wid0 in wids:
        vector = defaultdict(float);
        c_all = sum([ co_counts[wid0][x] for x in co_counts[wid0]])
        for wid1 in co_counts[wid0]:
            c_xy = co_counts[wid0][wid1];
            value = c_xy/c_all;
            vector[wid1] = value;
        vectors[wid0] = vector;
    return vectors;

def Jaccard_sim(v0,v1):
  '''Compute the Jaccard similarity between two sparse vectors.
  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  # <todo>
  :return: Jaccard similarity between v0 and v1
  '''
  all_keys = set([*v0.keys(), *v1.keys()])
  sum_max = 0.0;
  sum_min = 0.0;
  for key in all_keys:
      sum_max += max(v0[key],v1[key]);
      sum_min += min(v0[key],v1[key]);
  return sum_min/sum_max;

def dice_sim(v0,v1):
  '''Compute the Dice similarity between two sparse vectors.
  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  # <todo>
  :return: Dice similarity between v0 and v1
  '''
  all_keys = set([*v0.keys(), *v1.keys()])
  sum_max = 0.0;
  sum_min = 0.0;
  for key in all_keys:
      sum_max += v0[key]+v1[key];
      sum_min += 2*min(v0[key],v1[key]);
  return sum_min/sum_max;

def Pearson_sim(v0,v1):
  '''Compute the Pearson similarity between two sparse vectors.
  :type v0: dict
  :type v1: dict
  :param v0: first sparse vector
  :param v1: second sparse vector
  :rtype: float
  # <todo>
  :return: Pearson similarity between v0 and v1
  '''

  sum_0 = sum([v0[x] for x in v0])
  sum_1 = sum([v1[x] for x in v1])
  sqrt_sum_0 = sum([v0[x]**2 for x in v0])
  sqrt_sum_1 = sum([v1[x]**2 for x in v1])
  sum_all = sum([v0[x]*v1[x] for x in v0])
  Pearson = (sum_all - sum_0*sum_1/N) /(sqrt(sqrt_sum_0-(sum_0**2)/N)*sqrt(sqrt_sum_1-(sum_1**2)/N))
  return Pearson;


def Analogy(v0, v1, v2):
    '''
    Apply vector calculations using dictionary data structure
    :param v0: Vector
    :param v1: Vector
    :param v2: Vector
    :return: v0-v1+v2
    '''
    all_keys = set([*v0.keys(), *v1.keys(), *v2.keys()])
    return {x: v0[x]-v1[x]+v2[x] for x in all_keys}


def Pearsonr(X, Y):
    '''
    :param X: Sample X - the word pair frequency
    :param Y: Sample Y - the word pair similarity
    :return: The Pearson correlation
    '''
    return pearsonr(X, Y)

# test_words = ["cat", "dog", "computer", "mouse", "@justinbieber"]
test_words = ["kitchen", "kitchn","cook", "queen", "king", "woman", "man"]

stemmed_words = [tw_stemmer(w) for w in test_words]
all_wids = set([word2wid[x] for x in stemmed_words]) #stemming might create duplicates; remove them

# you could choose to just select some pairs and add them by hand instead
# but here we automatically create all pairs
wid_pairs = make_pairs(all_wids)


#read in the count information (same as in lab)
(o_counts, co_counts, N) = read_counts("data/counts", all_wids)

# make the word vectors
vectors_list = [];
# Three methods used : Prob, PPMI and T-test
vector_method_name_list = ["prob", "PPMI","T-test"]
vectors_list.append( create_prob_vectors(all_wids, o_counts, co_counts, N))
vectors_list.append( create_ppmi_vectors(all_wids, o_counts, co_counts, N))
vectors_list.append( create_ttest_vectors(all_wids, o_counts, co_counts, N))


count = 0
for vectors in vectors_list:
    print(vector_method_name_list[count])
    # compute three kinds of similarites for all pairs we consider
    c_sims = {(wid0,wid1): cos_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
    Ja_sims = {(wid0,wid1): Jaccard_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}
    Pearson_sims = {(wid0,wid1): Pearson_sim(vectors[wid0],vectors[wid1]) for (wid0,wid1) in wid_pairs}

    # for analogy testing queen - king = woman - man
    v0 = vectors[word2wid['queen']]
    v1 = vectors[word2wid['king']]
    v2 = vectors[word2wid['man']]
    v3 = vectors[word2wid['woman']]
    # X sample of Pearson correlation - the frequency of the word pairs
    X = [log(sqrt(o_counts[x]*o_counts[y]),2) for (x,y) in wid_pairs]


    print("message: the output format are changed for convenience for Latex table ")
    # print cosine result
    print("Sort by cosine similarity")
    print_sorted_pairs(c_sims, o_counts)
    Y = [c_sims[(x,y)] for (x,y) in wid_pairs]
    print("\nSpearman correlation: {}\n".format(Pearsonr(X,Y)))
    print("(queen-king+man, man) vs (woman, man)")
    print(" {:.2f} vs {:.2f}".format(cos_sim(Analogy(v0,v1,v2),v2), cos_sim(v3,v2)))
    print("\n\n")


    print("Sort by Jaccard similarity")
    print_sorted_pairs(Ja_sims, o_counts)
    Y = [Ja_sims[(x,y)] for (x,y) in wid_pairs]
    print("\nSpearman: {}\n".format(Pearsonr(X,Y)))
    print("queen-king+man, man vs woman,man")
    print(" {:.2f} vs {:.2f}".format(Jaccard_sim(Analogy(v0,v1,v2),v2), Jaccard_sim(v3,v2)) )
    print("\n\n")


    print("Sort by Pearson similarity")
    print_sorted_pairs(Pearson_sims, o_counts)
    Y = [Pearson_sims[(x,y)] for (x,y) in wid_pairs]
    print("\nSpearman: {}\n".format(Pearsonr(X,Y)))
    print("queen-king+man, man vs woman,man")
    print(" {:.2f} vs {:.2f}".format(Pearson_sim(Analogy(v0,v1,v2),v2), Pearson_sim(v3,v2)) )
    print("\n\n\n\n")
    count += 1;


# # show the lowest/highest occurring words
# a = sorted(o_counts, key=lambda x: o_counts[x], reverse = True)[0:50];
# print([wid2word[x] for x in a])
# print([o_counts[x] for x in a])
#
# b = sorted(o_counts, key=lambda x: o_counts[x], reverse = False)[0:200];
# print([wid2word[x] for x in b])
# print([o_counts[x] for x in b])