# News: 
 07242025：
 We will stop the maintainance for this repo. All the right for commercial use and research use are released. Feel free to use :) .
 

FNSPID has been selected as KDD2024 Applied Data Science Track Paper



# FNSPID: A Comprehensive Financial News Dataset in Time Series
FNSPID (Financial News and Stock Price Integration Dataset), is a comprehensive financial dataset designed to enhance stock market predictions by combining quantitative and qualitative data. It contains 29.7 million stock prices and 15.7 million financial news records for 4,775 S&P500 companies from 1999 to 2023, gathered from four stock market news websites. This dataset stands out for its scale, diversity, and unique incorporation of sentiment information from financial news. Research using FNSPID has shown that its extensive size and quality can significantly improve the accuracy of market predictions. Furthermore, integrating sentiment scores into analyses modestly boosts the performance of transformer-based models. FNSPID also introduces a reproducible method for dataset updates, offering valuable resources for financial research, including complete work, code, documentation, and examples available online. This dataset presents new opportunities for advancing predictive modeling and analysis in the financial research community.


### Dataset location
Due to the large volume of the dataset, the dataset is available at the [Hugging Face](https://huggingface.co/datasets/Zihan1004/FNSPID/).

### What can this repo do? 
The FNSPID repository offers the FNSPID dataset, experimental results, and a news content scraper tool. It provides comprehensive financial data combining stock prices and news records for S&P500 companies, demonstrates the dataset's impact on prediction accuracy, and includes a tool for updating the dataset with new financial news. 

In this GitHub repo, we did three main tasks：
## 1. Data scraper. 
In folder `data_scraper`, we provided tools to collect news data from Nasdaq.
## 2. Data processor.
In folder `data_processor`, we explained how we integrate our data into workable data.
## 3. Dataset experiments.
In folder `dataset_test`, we provided ways using DL models to test the dataset.


### For details of how to use them, you can find instructions `data_scraper.md`, `data_processor.md`, and `dataset_test.md` in these folders

### Download
```bash
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_price/full_history.zip
wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv
```

## Weighted Sentiment CNN Experiment Summary (2026-02)
This section summarizes the local experiment documented in `dataset_test/CNN-for-Time-Series-Prediction/experiment_summary.md`.

### Goal
Compare:
- Baseline sentiment feature (`Scaled_sentiment` as-is)
- Weighted sentiment feature using news availability and volume spike

Weighted feature design:
```python
sent_center = (Sentiment_gpt - 3) / 2  # 1~5 -> -1~1
vol_z = zscore(Volume, rolling window=N)
intensity = News_flag * (1 + k * vol_z)
intensity = clip(intensity, lower=0)
Scaled_sentiment = sent_center * intensity
```

### Setup
- Model: `dataset_test/CNN-for-Time-Series-Prediction`
- Split: `train_test_split = 0.85`
- Sequence length: 50
- Prediction length: 3
- Tested weighted datasets:
  - `data_wN10_k0p1`, `data_wN10_k0p3`, `data_wN10_k0p6`
  - `data_wN20_k0p1`, `data_wN20_k0p3`, `data_wN20_k0p6`
- Not yet tested: `data_wN60_*`

### Baseline results (sentiment, original data)
- AMD: MAE 0.097505, MSE 0.014424, R2 0.513303
- GOOG: MAE 0.052328, MSE 0.003694, R2 0.194527
- KO: MAE 0.037197, MSE 0.002010, R2 0.547467
- TSM: MAE 0.088045, MSE 0.011136, R2 0.516412
- WMT: MAE 0.024453, MSE 0.000930, R2 0.351031

### Key observations from weighted runs
- TSM and GOOG improved in most weighted configurations.
- KO showed mixed behavior (small improvements at low k, degradation at higher k in several settings).
- AMD mostly degraded across weighted settings.
- WMT degraded in all tested weighted settings (largest drop at k=0.6).

### Best directional improvements observed
- TSM: best around `wN20, k=0.6` (dMAE -0.021497, dMSE -0.003821, dR2 +0.165917)
- GOOG: best around `wN10, k=0.6` (dMAE -0.009605, dMSE -0.000932, dR2 +0.203152)

### Repro/Reference files
- `dataset_test/CNN-for-Time-Series-Prediction/experiment_summary.md`
- `dataset_test/CNN-for-Time-Series-Prediction/comparison_cnn_weighted_5.csv`

### Related Financial Datasets: 
[Financial-News-Datasets 2013](https://github.com/philipperemy/financial-news-dataset)

[Benzinga](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)



## RNN/LSTM/GRU Experiment: Sequential Sentiment Decay (2026-02).

### Goal
This experiment investigates the temporal persistence of news sentiment.
Unlike CNNs that focus on local feature extraction, we hypothesize that sentiment impact decays exponentially over time and influences subsequent stock prices through sequential memory.


### 📊 Experiment A: Model Performance Comparison (vs. Base CNN)
In this phase, we compared the best-performing configurations of each sequential model against the Baseline CNN to evaluate the effectiveness of capturing "sentiment residue."

#### Key results ($R^2$ inprovement)
| Stock | CNN (Weighted) | RNN | LSTM | GRU |
| :--- | :---: | :---: | :---: | :---: |
| **GOOG** | +0.2064 | **+0.5863** | +0.5844 | +0.4931 |
| **TSM** | +0.1659 | +0.4040 | **+0.4173** | +0.4105 |
| **AMD** | -0.0103 | +0.3455 | **+0.3694** | +0.3618 |
| **WMT** | -0.2865 | **+0.4503** | +0.4175 | +0.4472 |
| **KO** | +0.0591 | +0.3630 | +0.3782 | **+0.3829** |

#### **Findings**
- Sequential Superiority: Every sequential model (RNN/LSTM/GRU) significantly outperformed the CNN across all tickers
- Recovery of Poor Performers: Stocks like WMT and AMD, which showed degradation in the weighted CNN experiment, achieved massive $R^2$ gains (up to +0.45) when processed through sequential layers


### 📊 Experiment B: Parametric Sensitivity Analysis ($N$ and $k$)
In this phase, we analyzed how the rolling window size ($N$) and volume sensitivity ($k$) affect the $R^2$ performance of each model.

#### 1. GOOG result
| Window ($N$) | Sensitivity ($k$) | CNN | RNN | LSTM | GRU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | 0.2935 | 0.7260 | 0.2922 | -1.5704 |
| 10 | 0.3 | 0.0835 | **0.7808** | 0.6041 | 0.0780 |
| 10 | 0.6 | 0.3975 | 0.5530 | 0.6694 | -1.9824 |
| 20 | 0.1 | 0.3925 | 0.5910 | 0.7071 | -1.1768 |
| 20 | 0.3 | **0.4005** | 0.5788 | 0.7168 | **0.6876** |
| 20 | 0.6 | 0.2985 | 0.7549 | **0.7789** | -0.2375 |

#### 2. TSM result
| Window ($N$) | Sensitivity ($k$) | CNN | RNN | LSTM | GRU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | 0.5134 | 0.8796 | 0.9114 | 0.8854 |
| 10 | 0.3 | 0.6394 | 0.8817 | **0.9337** | 0.9225 |
| 10 | 0.6 | 0.6564 | **0.9204** | 0.8832 | 0.9215 |
| 20 | 0.1 | 0.6594 | 0.8354 | 0.9240 | **0.9269** |
| 20 | 0.3 | 0.6664 | 0.9078 | 0.9182 | 0.9025 |
| 20 | 0.6 | **0.6814** | 0.9146 | 0.9239 | 0.8922 |

#### 3. AMD result
| Window ($N$) | Sensitivity ($k$) | CNN | RNN | LSTM | GRU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | **0.50** | 0.8187 | 0.8610 | 0.7280 |
| 10 | 0.3 | 0.39 | 0.8268 | 0.8816 | 0.8709 |
| 10 | 0.6 | 0.43 | 0.7850 | 0.8774 | **0.8751** |
| 20 | 0.1 | 0.40 | 0.8176 | **0.8827** | 0.8726 |
| 20 | 0.3 | 0.18 | 0.7944 | 0.8804 | 0.8223 |
| 20 | 0.6 | 0.43 | **0.8588** | 0.8807 | 0.8437 |

#### 4. WMT result
| Window ($N$) | Sensitivity ($k$) | CNN | RNN | LSTM | GRU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | -0.49 | 0.3081 | **0.7685** | 0.5741 |
| 10 | 0.3 | -0.109 | 0.2354 | 0.7496 | 0.6853 |
| 10 | 0.6 | -0.941 | 0.7334 | 0.7265 | 0.7032 |
| 20 | 0.1 | -0.577 | 0.7117 | 0.6955 | 0.7290 |
| 20 | 0.3 | -0.311 | **0.8013** | 0.6588 | **0.7982** |
| 20 | 0.6 | **0.065** | 0.7756 | 0.6719 | 0.7850 |

#### 5. KO result
| Window ($N$) | Sensitivity ($k$) | CNN | RNN | LSTM | GRU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 0.1 | **0.61** | -0.6099 | 0.8812 | 0.8826 |
| 10 | 0.3 | 0.56 | 0.8882 | **0.9236** | **0.9304** |
| 10 | 0.6 | 0.51 | 0.9029 | 0.9062 | 0.9194 |
| 20 | 0.1 | 0.34 | 0.8936 | 0.8973 | 0.9139 |
| 20 | 0.3 | 0.39 | **0.9105** | 0.9257 | 0.9175 |
| 20 | 0.6 | 0.52 | 0.8825 | 0.9187 | 0.8888 |

#### **Findings**
| Stock | Best Model | Best Params | $R^2$ (Peak) | Observations |
| :--- | :---: | :---: | :---: | :--- |
| **TSM** | **LSTM** | $N=10, k=0.3$ | **0.9337** | Highly stable across all sequential models. |
| **AMD** | **LSTM** | $N=20, k=0.1$ | **0.8827** | Massive jump from CNN (0.5) to sequential models (0.8+). |
| **WMT** | **RNN** | $N=20, k=0.3$ | **0.8013** | Performance normalized at $N=20$ where CNN failed. |
| **KO** | **GRU** | $N=10, k=0.3$ | **0.9304** | Highest peak, but RNN showed instability without gating. |


# Disclaimer
## Reliability and Security

The code provided in this GitHub repository is shared without any guarantee for its reliability and security. The developers and contributors of this project expressly disclaim any warranty, either implied or explicit, regarding the code's performance, security, or suitability for any particular purpose. The users should employ this code at their own risk, acknowledging that the developers shall not be held responsible for any damages or issues arising from its use.


## Purpose of Use

This code is primarily intended to illustrate our workflow processes and to serve as a medium for educational exchange and learning among users. It is made available for the purpose of showcasing our technical approaches and facilitating learning within the community. It is not designed for direct application in production environments or critical systems.

## Prohibition of Commercial Use

The use of this code for commercial purposes is strictly prohibited without prior authorization. If you wish to utilize this code in a commercial setting or for any revenue-generating activities, you are required to obtain explicit permission from the original authors. Please contact us at puma122707@gmail.com to discuss licensing arrangements or to seek approval for commercial use.


## Acknowledgement

By accessing, using, or contributing to this code, you acknowledge having read this disclaimer and agree to its terms. If you do not agree with these conditions, you should refrain from using or interacting with the code in any manner.


## Citation
```bibtex
@misc{dong2024fnspid,
      title={FNSPID: A Comprehensive Financial News Dataset in Time Series}, 
      author={Zihan Dong and Xinyu Fan and Zhiyuan Peng},
      year={2024},
      eprint={2402.06698},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST}
}

