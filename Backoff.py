import re
import sys
from random import random
from math import log
from collections import defaultdict
import random


def preprocess_line(line):
    line = line.lower();
    line = list(line);
    lenth = len(line);

    # literate until all characters checked
    i = 0;
    while i < lenth:
        chara = line[i];
        if ('a' <= chara <= 'z') or (chara == '.') or (chara == ' '):
            i += 1;
        elif '0' <= chara <= '9':
            line[i] = '0';
            i += 1;
        else:
            line.pop(i);
            lenth -= 1;

    # only add one "#"
    # try adding "#####" here and you will see the generated result filled with "\n"
    line = '#' + ''.join(line) + '#';
    return line


def train_backoff(vocabulary):
    if __name__ == "__main__":

        infile = "training.en"
        six_counts = defaultdict(int);
        five_counts = defaultdict(int);
        four_counts = defaultdict(int);
        tri_counts = defaultdict(int);
        bi_counts = defaultdict(int);
        one_counts = defaultdict(int);

        counts = [one_counts, bi_counts, tri_counts, four_counts, five_counts, six_counts];

        with open(infile) as f:
            for line in f:
                line = preprocess_line(line)  # modify characters and add "#"s

                for i in range(0,6):

                    for j in range(len(line) - i):
                        i_gram = line[j:j + i+1];
                        counts[i][i_gram] += 1;

            f.close();

        # save the counts instead of probability
        with open("backoff.en", 'w') as f:

            for i in counts:
                for j in i.keys():
                    f.write(j+":" + str(i[j]) + "\n");

            f.close();

        N = 0;
        for i in counts[0].values():
            N += i;
        return N;


# back of with λ = 0.4;
def use_backoff(vocabulary, N):

    lamda = 0.4;

    # download model from file
    model = defaultdict(int);
    with open("backoff.en", 'r') as f:
        lines = f.readlines();
        for i in lines:
            iss = i.split(":");
            model[iss[0]] = int(iss[1]);
        f.close();

    # generate start
    word_count = 0;
    result = "#";
    while word_count < 300:

        distri = defaultdict(float);
        probability_sum = 0.0;

        # calculate the probability for each trigram.
        for i in vocabulary:
            # this flag determines whether the back-off terminates in unigram
            flag = True;
            prior = 1.0;
            # iterate from 6-gram to bigram
            for j in range(-5,0):
                temp = result[j::]
                jplusone_gram = temp + i;
                count = model[jplusone_gram];
                if count != 0:
                    # choose this n-gram
                    distri[i] = prior * count / model[temp];
                    probability_sum += distri[i];
                    flag = False;
                    break;
                # before search the next level n-gram, change the prior
                prior *= lamda;
            # apply unigram formula
            if(flag):
                distri[i] = prior * model[i]/N;
                probability_sum += distri[i];

        # decide the next character
        rand = random.uniform(0.0, probability_sum);
        num = 0.0;
        word_count += 1;
        for i in distri.keys():
            num += distri[i];
            if rand <= num:
                # if the terminal symbel "#" is generated, add "#" as new start. Not count in the 300 character counts
                if i == '#':
                    result += "#"
                    word_count -= 1;
                else:
                    result += i;
                break;

    # print (result[1::]);
    return result[1::].replace("#", "\n");


# calculate perplexity of test document
def calcu_PP():
    model = defaultdict(int);
    intext = "test";
    with open("backoff.en", 'r') as f:
        lines = f.readlines();
        for i in lines:
            iss = i.split(":");
            model[iss[0]] = int(iss[1]);
        f.close();

    PP = 1.0;
    N = 0;

    # count N - the number of trigrams in text.
    with open(intext, 'r') as f:
        for line in f:
            for j in range(len(line) +1):
                N += 1;
        f.close();

    with open(intext, 'r') as f:
        for line in f:
            line = preprocess_line(line)
            # check all characters except the first "#"
            for i in range(1,len(line)):
                flag = True;
                prior = 1.0;
                lamda = 0.4;
                # iterate from 6-gram to bigram
                for j in range(-5,0):
                    gram = line[i+j:i+1];
                    count = model[gram];
                    res = 1.0;
                    if count != 0:
                        res = prior * count / float(model[line[i+j:i]]);
                        PP = PP * pow(res, -1 / N);
                        flag = False;
                        break;
                    prior *= lamda;
                # calculate PP using unigram
                if (flag):
                    res = prior * model[line[i]] / float(model[line[i+j:i]]);
                    PP = PP * pow(res, -1 / N);
        f.close();

    return PP;


if __name__ == '__main__':

    vocabulary = [' ', '#', '.', '0'];
    for i in range(97, 123):
        vocabulary.append(chr(i));

    # train
    N = train_backoff(vocabulary);

    # generate text
    Generated_text = use_backoff(vocabulary, N);
    print("Generated_text: \n" + use_backoff(vocabulary, N));

    # calculate
    PP = calcu_PP();
    print("PP: {}".format(PP));