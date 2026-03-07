## Data paths ##
LOW_DRIFT_PATH = "data/processed/PFL1_low_drift_devices.csv"
PFL1_PATH      = "data/processed/PFL1_preprocessed.csv"

## Variables ##
INPUT_VARS       = ['Temperature', 'Salinity', 'z']   # encoder inputs (X)
TARGET_VARS      = ['Oxygen']                          # held-out reconstruction target (Y)
MIN_TARGET_PROBE = 8                                   # guaranteed floats with target coverage in probe set

## Data ##
DEPTH_STRIDE = 10   # take every Nth depth level per cast (reduces compute, retains structure)

## Split ##
TRAIN_FRAC = 0.70
TEST_FRAC  = 0.20
PROBE_FRAC = 0.10
SEED       = 42

## Model hyperparameters ##
LATENT_DIM     = 16         # dimension of latent profile vector p
ENCODER_HIDDEN = [64, 64]   # hidden layer sizes in encoder MLP
DECODER_HIDDEN = [64, 64]   # hidden layer sizes in decoder MLP
ODE_HIDDEN     = [64, 64]   # hidden layer sizes in ODE function f(p, lat, lon, t)

## Training ##
ENCODER_LR     = 1e-3
ENCODER_EPOCHS = 100
ODE_LR         = 1e-3
ODE_EPOCHS     = 100
BATCH_SIZE     = 32