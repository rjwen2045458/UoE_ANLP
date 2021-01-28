import re
import sys
from random import random
from math import log
from collections import defaultdict
import random
import Backoff


# initialize a vocabulary including valid characters a-z, #, ., 0, space.
def init_vocabulary():
    vocabulary = [' ', '#', '.', '0'];
    for i in range(97, 123):
        vocabulary.append(chr(i));
    return vocabulary;


# this function preprocess a line by removing or changing characters not in the vocabulary, then add #s.
# input : a string
# output : a string with all characters in vocabulary
# i.e. "cCión1 56..A" => "##ccin0 00..a#"
def preprocess_line(line):
    line = line.lower();
    line = list(line);  # turn the line into list
    lenth = len(line);
    i = 0;  # i is the index of the current character to be checked
    while i < lenth:   # literate until all characters checked
        chara = line[i];
        # pass if chara is valid, else delete or change it.
        if ('a' <= chara <= 'z') or (chara == '.') or (chara == ' '):
            i += 1;
        elif '0' <= chara <= '9':
            line[i] = '0';
            i += 1;
        else:
            line.pop(i);
            lenth -= 1;
    # add "#"s for tri-gram training.
    line = '##' + ''.join(line) + '#';
    return line


# reads from training resource and trains a model and save it
def train(infile="training.en", vocabulary={}, outfile="model.en"):

    tri_counts = defaultdict(int);  # counts of all trigrams in input
    bi_counts = defaultdict(int);   # counts of all bigrams in input, except the last bigram in one line.

    with open(infile) as f:
        for line in f:
            line = preprocess_line(line)  # modify characters and add "#"s

            # count all trigram
            for j in range(len(line) - 2):
                trigram = line[j:j + 3];
                tri_counts[trigram] += 1;

            # count all bi-gram except the last one
            # i.e.
            # the last bi-gram of "#abc#" is "c#",
            # but it won't be a prefix or any tri-gram in this sentence, thus not counted
            for j in range(len(line) - 2):
                bigram = line[j:j + 2];
                bi_counts[bigram] += 1;
        f.close();

    # calculate the probability and save it
    # smoothing using add-a method with a = 0.001
    with open(outfile, 'w') as f:

        for i in vocabulary:
            for j in vocabulary:
                for k in vocabulary:
                    trigram = i + j + k;
                    bigram = i + j;
                    f.write(trigram + " " + "%.3e" % ((tri_counts[trigram] + 0.001) / (
                                bi_counts[bigram] + 0.001 * len(vocabulary))) + "\n");

        f.close();


# generate from specific trigram model and the vocabulary
def generate_from_LM(inmodel,vocabulary):

    # download model from file
    trained_model = defaultdict(float);
    with open(inmodel, 'r') as f:
        lines = f.readlines();
        for i in lines:
            trained_model[i[0:3]] = float(i[4::]);
        f.close();
    word_count = 0;

    # add "##" to start generate
    result = "##";
    while word_count < 300:

        # take the last two characters to predict next one
        temp = result[-2::];
        distri = defaultdict(float);
        probability_sum = 0.0;

        # calculate the probability for each trigram.
        # calculate the sum of probabilities.
        for i in vocabulary:
            tri = temp + i;
            distri[i] = trained_model[tri];
            probability_sum += distri[i];

        # decide the next character
        rand = random.uniform(0.0, probability_sum);
        num = 0.0;
        word_count += 1;
        for i in distri.keys():
            num += distri[i];
            if rand <= num:
                # if the terminal symbel "#" is generated, add "##" as new start. Not count in the 300 character counts
                if i == '#':
                    result += '##'
                    word_count -= 1;
                else:
                    result += i;
                break;

    return result[2::].replace("##", "\n");


# generate 300 characters from two models
def generate(vocabulary):
    inmodel = "model-br.en";
    text = generate_from_LM(inmodel, vocabulary);

    print("text from model-br.en: \n" + text + "\n");
    inmodel = "model.en";
    text = generate_from_LM(inmodel, vocabulary);
    print("text from model.en: \n" + text + "\n");


# calculate perplexity of test document
def calcu_PP(intext="test", inmodel="model-br.en"):
    trained_model = defaultdict(float);
    with open(inmodel, 'r') as f:
        lines = f.readlines();
        for i in lines:
            trained_model[i[0:3]] = float(i[4::]);
        f.close();

    PP = 1.0;
    N = 0;

    # count N - the number of trigrams in text.
    with open(intext, 'r') as f:
        for line in f:
            line = preprocess_line(line)
            N += len(line)-2;
        f.close();


    with open(intext, 'r') as f:
        for line in f:
            line = preprocess_line(line)

            for j in range(len(line) - 2):
                trigram = line[j:j + 3];
                PP = PP * pow(trained_model[trigram], -1 / N);
        f.close();
    return PP;


# main
if __name__ == "__main__":
    # initialize the vocabulary
    vocabulary = init_vocabulary();

    for i in [".en", ".de", ".es"]:
        # train with three training text, save in corresponding models
        train("training"+i, vocabulary, "model"+i);

    # generate the 300 characters sentence with provided model and my english model
    generate(vocabulary);

    # calculate the PP for each model
    for i in [".en", ".de", ".es"]:
        inmodel = "model" + i;
        PP = calcu_PP("test",inmodel);
        print ("Perplexity of {}: {}\n".format(inmodel, PP));




