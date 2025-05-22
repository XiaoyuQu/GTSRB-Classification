## Classification on GTSRB Dataset

### CNN Implementation

The CNN implementation for GTSRB is based on [this repository](https://github.com/poojahira/gtsrb-pytorch), with minor modifications made to the data preprocessing pipeline.

### ViT Fine-tuning Implementation

Our fine-tuning implementation for ViT-Base-Patch16-224 can be found in the `finetuning-ViT-GTSRB.py` script.
Due to time constraints, the model was fine-tuned for only 2 epochs, achieving an accuracy of approximately **97.57%** on the GTSRB test set.

The checkpoints can be find [here](https://huggingface.co/datasets/Daxuxu36/ViT-GTSRB) on huggingface.