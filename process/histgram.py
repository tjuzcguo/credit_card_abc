import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

prob_score = pd.read_csv(r'D:\Users\zcguo\PycharmProjects\credit_score\data\score_prob.csv')

d=prob_score['score'].hist(bins=20).get_figure()
d.savefig('score.png')
d1=prob_score['probility'].hist(bins=20).get_figure()
d1.savefig('probability.png')
