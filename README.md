# AI Video Rating System

## Overview

This repository implements an AI-based video rating system designed to assess the quality of video content through automated ranking and scoring. It leverages a two-stage training strategy with coarse and fine training phases, ensuring the system captures both general and detailed video features. The system uses a combination of supervised learning and ranking-based loss functions to maintain the quality order of video samples.

## Features
- Two-stage training strategy:
  - **Coarse Training Phase**: Focuses on learning general patterns with lower-resolution data.
  - **Fine Training Phase**: Refines the model with higher-resolution data and specific features.
- Implementation of **Rank Loss** to ensure the predicted quality scores respect the true ranking order.
- Custom dataset support with Mean Opinion Scores (MOS) for supervised learning.
- Metrics for evaluation: SROCC, PLCC, KROCC, and RMSE.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Trm1001/AI-Video-Rating-System.git
   cd AI-Video-Rating-System
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

1. Ensure you have the training and validation data organized as required in the YAML configuration files.
2. Run the training script:
   ```bash
   python train.py -o config/kwai_simpleVQA.yml
   ```
3. The script will first run the coarse training phase using `kwai_simpleVQA.yml`.
4. At the specified epoch, the training switches to the fine training phase using `kwai_fine_simpleVQA.yml`.

### Evaluation

To evaluate a trained model:
```bash
python evaluate.py --config config/kwai_simpleVQA.yml --weights checkpoint/best_model.pth
```

## Configuration

The training and fine-tuning phases are configured using YAML files:
- `kwai_simpleVQA.yml`: Coarse training configuration.
- `kwai_fine_simpleVQA.yml`: Fine training configuration.

Key configuration parameters include:
- `num_epochs`: Total training epochs.
- `batch_size`: Batch size for training.
- `data`: Paths and settings for training and validation datasets.
- `optimizer`: Learning rate, weight decay, and optimizer settings.

## Rank Loss

The Rank Loss function is implemented as a pairwise hinge loss to ensure the predicted scores respect the ranking order of true quality scores. It penalizes incorrect rankings and encourages correct relative ordering:

\[ \text{Rank Loss} = \sum_{i,j} \max(0, 1 - (\hat{y}_i - \hat{y}_j) \cdot \text{sign}(y_i - y_j)) \]

## Dataset

The system uses a dataset annotated with Mean Opinion Scores (MOS) for training and evaluation. The dataset should include:
- **Training data**: CSV file specifying video file paths and their MOS.
- **Validation data**: Similar format as training data for evaluation.

## Metrics

The following metrics are used for evaluation:
- **SROCC**: Spearman Rank-Order Correlation Coefficient.
- **PLCC**: Pearson Linear Correlation Coefficient.
- **KROCC**: Kendall Rank-Order Correlation Coefficient.
- **RMSE**: Root Mean Square Error.

## Results

The system achieved the following validation results:

| Metric  | Coarse Training | Fine Training |
|---------|-----------------|---------------|
| SROCC   | 0.70            | 0.82          |
| PLCC    | 0.72            | 0.85          |
| KROCC   | 0.68            | 0.80          |
| RMSE    | 0.10            | 0.08          |

## Citation

If you use this codebase, please cite our work:
```
@article{your_citation,
  title={AI Video Rating System},
  author={Your Name},
  journal={GitHub Repository},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by the training methods of the VideoBooth project and incorporates ideas from video quality assessment research.

