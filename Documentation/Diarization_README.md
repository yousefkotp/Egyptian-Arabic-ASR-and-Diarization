# <a name="diarization-for-egyptian-dialect"></a> Diarization for Egyptian Dialect

This file is a documentation for the work exerted in diarization process for Egyptian dialect, experments done, insights gained, approaches selected and results obtained so, make a cup of coffee and have fun


## Table of Contents
- [Diarization for Egyptian Dialect](#diarization-for-egyptian-dialect)
  * [Introduction](#introduction)
  * [Experiments on sample data](#experiment-sample-data)
    + [PCA experiments](#pca-experiments)
    + [Embedding models experiments](#embedding-models-experiments)
    + [Window size experiments](#window-size-experiments)
    + [Insights](#insights)
  * [Datasets Exploration](#datasets-exploration)
  * [Experiments on callhome test data](#experiments-callhome)
    + [Insights](#insights)
  * [Dataset collection and synthsis trials](#dataset-collection-synthsis-trials)
  * [Insights](#insights)
  * [Contributors](#contributors)
  * [Supervisor](#supervisor)
  * [References](#references)
 
## <a name="introduction"></a> Introduction
Speaker Diarization is the task of segmenting and co-indexing audio recordings by speaker. in other words, diarization implies finding speaker boundaries and grouping segments that belong to the same speaker, and, as a by-product, determining the number of distinct speakers. the goal is not to identify known speakers, but to co-index segments that are attributed to the same speaker.

### Aprroaches
Current approaches for speaker diarization can be summarized in two main approaches: Multi-stage (pipeline) and End to End speaker diarization
<div>
<img src = "https://github.com/user-attachments/assets/ab612ca6-de15-4a76-a9fc-4f8a0a794e0b" width="50%">
<img src = "https://github.com/user-attachments/assets/f7ff23d0-d29b-4fac-9362-d69c54bd903f" width="50%">
</div>

## Contributors
- [Abdelrahman Elnenaey](https://github.com/AbdelrhmanElnenaey)
- [Rana Barakat](https://github.com/ranabarakat)
- [Louai Zahran](https://github.com/LouaiZahran)

## Supervisor
- [Ismail El-Yamany](https://github.com/IsmailElYamany)

## References
- [Improving Diarization Robustness using Diversification, Randomization and the DOVER Algorithm](https://arxiv.org/abs/1910.11691)
