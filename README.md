# UCSentiE 
This is an open source implementation of our UCSentiE model to obtain sentiment-aware cross-lingual word embeddings which eliminate both linguistic and sentiment gap between two languages, described in our paper: A Cross-lingual Sentiment Embedding Model with Semantic and Sentiment Joint Learning. UCSentiE leverages prior sentiment information from the source language to integrate sentiment information into CLWE without compromising cross-lingual word semantics. We evaluate UCSentiE on two tasks: Bilingual Dictionary Induction(BLI) and Cross-lingual Sentiment Analysis(CLSA). The package includes a script to build sentiment-aware cross-lingual word embeddings as described in the paper, as well as the two evaluation tools of BLI and CLSA.
# Requirements
* Python 3
* Numpy
* SciPy
* CuPy (optional, only required for CUDA support)
# Usage
In order to build sentiment-aware cross-lingual word embeddings, you should first train monolingual word embeddings for each language using your favorite tool (e.g. [word2vec](https://github.com/tmikolov/word2vec) or [fasttext](https://github.com/facebookresearch/fastText)) and then map them to a common space with our model as described below. Having done that, you can evaluate the resulting sentiment-aware cross-lingual word embeddings using our included tools as discussed next.
## Mapping
You can generate the sentiment-aware cross-lingual word embeddings by our UCSentiE model with the following code:
```
python UCL-SWE.py
```
You can also obtain the cross-lingual embeddings by [VecMap](https://github.com/lishiqimagic/vecmap) model with the following code:
```
python VecMap.py
```
## Evaluation
### 1. Bilingual Lexicon Induction
For the BLI task, we use CSLS retrieval algorithm to select the target language words with the nearest CSLS value for each word in the source language as translation words to construct a bilingual dictionary. Then, the [MUSE bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) provided by Facebook were used as the benchmark to compare the bilingual dictionaries constructed by the model and calculate the accuracy:
```
python WordSimilarity.py
```
### 2. Cross-lingual Sentiment Analysis
For the CLSA task, we use metrics of accuracy (Acc), Precision(Pre) and F1-score to evaluate the performance of multilingual sentiment classification based on the generated cross-lingual word embeddings.:
```
python CLSA_SVM.py
```
