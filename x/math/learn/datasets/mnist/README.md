Downloaded from https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/

The format is
    label, pix-11, pix-12, pix-13, ...

Refer to [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/)

## MNIST Training Results Comparison

Two CNN models were tested on this dataset with different architectures and training configurations:

### TestMNIST (Smaller Model)
- **Architecture**: Conv2D(1→16) → Conv2D(16→32) → Dense(1568→128) → Dense(128→10)
- **Training**: 1000 samples, 5 epochs, Adam lr=0.001, Xavier initialization
- **Results**:
  - Final training accuracy: **95.10%**
  - Test accuracy: **87.50%**
  - Test loss: 0.3046

### TestMNISTLarge (Larger Model)
- **Architecture**: Conv2D(1→16) → MaxPool → Conv2D(16→32) → MaxPool → Dense(1568→64) → Dropout(0.3) → Dense(64→10)
- **Training**: 1500 samples, 2 epochs, Adam lr=0.002, He initialization
- **Results**:
  - Final training accuracy: **78.80%**
  - Test accuracy: **86.00%**
  - Test loss: 0.4174

### Key Findings

**Convergence Speed**:
- The larger model reached 86% test accuracy in just 2 epochs vs 5 epochs for the smaller model
- Despite fewer epochs, the larger model achieved comparable test performance

**Accuracy Comparison**:
- Smaller model: Better training accuracy (95.1% vs 78.8%) but similar test accuracy (87.5% vs 86.0%)
- Larger model shows signs of regularization working (dropout preventing overfitting)

**Architecture Impact**:
- Pooling layers in the larger model significantly improved efficiency and convergence speed
- Dropout regularization helped prevent overfitting despite the deeper network
- The larger model used better initialization (He vs Xavier) and slightly higher learning rate

**Training Efficiency**:
- Larger model trained faster per epoch despite more parameters (pooling reduces spatial dimensions)
- Both models achieved solid performance (>85% test accuracy) on the limited dataset

The larger model demonstrates better generalization with regularization, while the smaller model overfits slightly more but achieves higher peak training accuracy. The pooling layers provide significant computational efficiency gains.

