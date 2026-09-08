# Fresh Balanced V3 Held-Out Test Evaluation Report

## Checkpoint Identity
- **Checkpoint Path:** `specialists/optical_sar/checkpoints/experiment_balanced_v3_100scenes/cmaf_landcover_balanced_v3_100scenes.pth`
- **SHA-256:** `26288ce0e8d3f251c7b962638b0a8228288954655b6b4b0514a4edd482a4c76b`
- **Parameters:** `19,755,144`
- **Strict Load:** `PASSED (0 missing, 0 unexpected)`
- **Trained Epoch:** `31` (Stage 2 fine-tuning)
- **Validation mIoU:** `0.2437`

## Dataset Provenance & Split
- **Dataset:** Official Wuhan University WHU-OPT-SAR Dataset
- **Held-Out Test Scenes:** 15 scenes (zero leakage, seed 42)
- **Total Test Tiles:** `4950`
- **Valid Pixels:** `308,687,656`
- **Border Ignored Pixels (255):** `15,715,544`

## Fresh Test Metrics
- **Overall Pixel Accuracy (OA):** `71.71%`
- **Mean IoU (mIoU):** `35.08%`
- **Macro Precision:** `46.58%`
- **Macro Recall:** `52.48%`
- **Macro F1 Score:** `46.62%`
- **Weighted F1 Score:** `74.18%`

## Per-Class Breakdown
| Class | IoU | F1 / Dice | Precision | Recall | GT Pixels | Pred Pixels |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Background** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1,973` | `2,946,950` |
| **Farmland** | `0.5920` | `0.7437` | `0.7596` | `0.7285` | `111,315,326` | `106,754,018` |
| **City** | `0.4457` | `0.6166` | `0.6843` | `0.5611` | `13,218,868` | `10,839,447` |
| **Village** | `0.3332` | `0.4999` | `0.4415` | `0.5760` | `16,869,004` | `22,009,292` |
| **Water** | `0.5071` | `0.6729` | `0.6924` | `0.6545` | `40,822,113` | `38,583,509` |
| **Forest** | `0.7369` | `0.8485` | `0.9227` | `0.7854` | `118,860,724` | `101,179,047` |
| **Road** | `0.1168` | `0.2091` | `0.1236` | `0.6773` | `3,083,943` | `16,895,066` |
| **Others** | `0.0746` | `0.1389` | `0.1025` | `0.2152` | `4,515,705` | `9,480,327` |

## Active Classes
- **Active Classes Count:** `8 / 8`
- **Active Classes:** `Background, Farmland, City, Village, Water, Forest, Road, Others`

## Historical Artifact vs. Fresh Evaluation
| Metric | Historical Artifact | Fresh Evaluation | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | `56.15%` | `71.71%` | `0.155637` |
| **Mean IoU (mIoU)** | `23.15%` | `35.08%` | `0.119312` |
| **Macro F1** | `34.60%` | `46.62%` | `0.120200` |
| **Background IoU** | `0.0000` | `0.0000` | `0.000000` |
| **Farmland IoU** | `0.3540` | `0.5920` | `0.237983` |
| **City IoU** | `0.2845` | `0.4457` | `0.161228` |
| **Village IoU** | `0.1670` | `0.3332` | `0.166268` |
| **Water IoU** | `0.3816` | `0.5071` | `0.125415` |
| **Forest IoU** | `0.5070` | `0.7369` | `0.229978` |
| **Road IoU** | `0.1020` | `0.1168` | `0.014768` |
| **Others IoU** | `0.0558` | `0.0746` | `0.018855` |

## Top 5 Error Confusion Modes
1. **Forest $\rightarrow$ Farmland**: `14,411,773` pixels (12.12% of Forest)
2. **Farmland $\rightarrow$ Water**: `8,151,523` pixels (7.32% of Farmland)
3. **Farmland $\rightarrow$ Village**: `6,447,174` pixels (5.79% of Farmland)
4. **Water $\rightarrow$ Farmland**: `5,954,585` pixels (14.59% of Water)
5. **Farmland $\rightarrow$ Forest**: `5,560,010` pixels (4.99% of Farmland)

## Inference Latency & Throughput
- **Mean Tile Latency:** `7.93 ms`
- **Median Batch Latency (P50):** `82.81 ms`
- **P95 Batch Latency:** `361.9 ms`
- **Throughput:** `19.76 FPS`
