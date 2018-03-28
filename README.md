# Kaggle: Jigsaw's Toxic Comment Classification Competition

Kaggle Jigsaw Toxic Comment Classification Jigsaw, a subsidiary of Alphabet Inc, is "dedicated to understanding global challenges and applying technological solutions, from countering extremism, online censorship and cyber-attacks, to protecting access to information." In this competition they ask us to analyze Wikipedia comments and detect various types of toxicity. The data set consists of 160,000 Wikipedia comments rated by humans with binary labels for the presence of toxicity, severe toxicity, obscenities, threats, insults, and hate speech.

## Exploratory Data Analysis

The key takeaways from the exploratory data analysis were:

The data is highly imbalanced, in multiple ways. 22% of comments are dirty. 10% of comments are toxic and 1% are hate speech. 10% have multiple tags.

Although the train and test distributions are of similar size, there appear to be substantial differences between the distributions.
Thankfully, there are no missing values.

There is evidence of minor data quality issues and inconsistencies, likely arising from the subjectivity of perceiving a comment to being toxic, which ultimately make achieving 100% accuracy impossible.

## Predictive Modeling

My solution consisted of blending together a multitude of models with varying pre-processing steps, architectures, and hyper-parameters. The primary models were:

Baseline logistic regression, with word-level and character-level n-gram TFIDFs and engineered features. For this simple SKLearn solution a model was independently trained on each class.

Recurrent neural nets with various embeddings (FastText, Word2Vec, Glove) and architectural settings (sentence length, number of recurrent units, dense layer size, dropout rate). The architecture typically consisted of two bi-directional recurrent layers (either GRU or LSTM), followed by a dense layer, with a sigmoid activation function output. Since in this case each class is solved simultaneously using the binary_crossentropy metric, performance gains will be achieved not only by increased model complexity but also by a more complex objective function.

Factorization machines optimized with a follow-the-regularized-leader algorithm. 

Character-level very-deep convolutional neural nets. Although this model did not achieve as high ROC-AUC scores as the other approaches, it was incredibly uncorrelated with other predictions, and therefore great for blending.

These four models were able to achieve high performance (ROC-AUC > 0.98) with very low inter-model correlations, and therefore ideal for blending/stacking. Prior to the inter-model blending step, I took a simple average of intra-model predictions with different hyper-parameters settings, which I found to substantially improve local CV scores.

## An Aside on Evaluation Metrics

Halfway through the competition, users and Kaggle/Jigsaw discovered that there were substantial differences between train/test distributions, which they were able to exploit by multiplying the final output by a "magic number" to improve scores. To avoid promoting this type of unprincipled "hacking" behavior, Kaggle/Jigsaw switched the evaluation metric from log-loss to mean, column-wise ROC-AUC, which is indifferent to the underlying distribution.

Another alternative could have been F1 score, which is the harmonic mean between recall and precision. In an inspection of model performance I found that while ROC-AUC and precision scores are fantastic for all six classes, the recall tends to be terrible. In other words, although there are very few false-positives, there are a ton of false negatives, which is ultimately arising for the high degree of data skew. This essentially means that our system may not be great for, say, automated tagging, but can be great for ranking toxicity. A moderator, with a limited time budget, can then be provided with the most likely toxic comments for manual review.

## Blending vs. Stacking

I found that my stacking approach,  based on LightGBM on the base-level model outputs of a hold-out set and engineered features, was unable to improve performance. Although the local CV score slightly improved, the public LB score did not increase nearly as much as the blended solution did, suggested that LightGBM was over-fitting to the hold-out set, which consisted of 10% or 16,000 examples. Other Kaggle users were instead training their models on the full-training set with out-of-fold predictions, which may have been a better strategy.

It turned out a simple average blending approach was superior to stacking. Between arithmetic, geometric, harmonic, and power average means, the simple arithmetic mean had the best performance. 

Further gains were achieved by weighted average blending with weights determined by a stochastic optimization approach I developed.

## The Finish Line

In the end, I submitted two entries. The first, my highest performing model as determined by local CV. The second, a hand-picked weighted-average blend with weights as determined by public CV. The highest scoring entry turned out to be the hand-picked blend, somewhat to my disappointment. My more principled, entirely local CV based model did well too though, it would also have been good enough to win a bronze model, and increased in the rankings by around 300 places. The hand-picked entry in contrast only dropped 21 places. I suppose the lesson is that although in the real world you should always trust local CV, in Kaggle competitions the Public LB can yield useful information due to differences between train and test distributions. 

## Looking Back

There were several things I wanted to do, but ran out of time. For some of the recurrent-neural-nets I augmented data my translating the data using a method proposed by another Kaggle user. This strategy ended up being key for the competition's 1st place solution. Unfortunately I didn't get around to using the method on other models. I also did not end up having enough time to perform more careful feature engineering. such as inspecting the ROC-AUC score of each individual feature. I wish that I had gone deeper with investigating and understanding the behavior of the VDCNN model, I might've been able to improve its score even further. Using Bayesian optimization rather than random search CV may have sped things up. And developing a deeper understanding of the factorized FTRL model should have helped as well.

And then there were the things I wish I did, that seem obvious in hindsight, based off of reading other people's solutions: concatenating embedding vectors and focusing more energy on understanding those effects, pseudo-labeling, blending TFIDF-LR with varying word and char n-gram levels, and including more interaction features.
