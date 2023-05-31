import torch


N_LAYERS = 4 # num of layers
EMB_DIM = 64
EPOCHS = 500
BATCH_SIZE = 2048
LR = 1e-3
EPOCHS_PER_EVAL = 20
EPOCHS_PER_LR_DECAY = 20
TOPK = [1, 5, 10, 20]
LAMBDA = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
