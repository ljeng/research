from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
RUNS_SUBDIR = Path('runs')
WEB_RUNS_SUBDIR = RUNS_SUBDIR / 'web'
WEB_SUBDIR = 'web'
QUICK_RUN_NAME = 'smoke'
FULL_RUN_NAME = 'baseline'
DEVICE = 'CPU'

HOST = '127.0.0.1'
PORT = 8765
DEPENDENCIES = ('torch', 'torchvision', 'matplotlib')
PROCESS_START_METHOD = 'spawn'
RUN_ID_RANDOM_BYTES = 4
REQUEST_MAX_BYTES = 1 << 14
REQUEST_TIMEOUT_SECONDS = 5
STOP_TIMEOUT_SECONDS = 3
MONITOR_TIMEOUT_SECONDS = 5

IMAGE_SIZE = 28
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE
HIDDEN_SIZE = 1 << 7
CLASSES = ['T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot']
NUM_CLASSES = len(CLASSES)
TRAIN_SAMPLES = 1 << 16
VALIDATION_SAMPLES = 1 << 12
QUICK_TRAIN_SAMPLES = 1 << 10
QUICK_VALIDATION_SAMPLES = 1 << 8
QUICK_TEST_SAMPLES = 1 << 8
QUICK_EPOCHS = 1
QUICK_PRACTICE_DEFAULT = True

DEFAULT_CONFIG = {'epochs': 5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'seed': 42,
    'smoke': False}
DEFAULT_PREDICT_ONLY = False
INTEGER_LIMITS = {'epochs': (1, 100),
    'batch_size': (1, 1 << 10),
    'seed': (0, (1 << 31) - 1)}
MIN_LEARNING_RATE = 1e-9
MAX_LEARNING_RATE = 1

RUN_FILENAME = 'run.json'
CHECKPOINT_FILENAME = 'model.pt'
HISTORY_FILENAME = 'history.json'
TEST_METRICS_FILENAME = 'test-metrics.json'
PREDICTIONS_DATA_FILENAME = 'predictions.json'
CURVES_FILENAME = 'learning-curves.png'
PREDICTIONS_FILENAME = 'predictions.png'
ARTIFACTS = {'curves': CURVES_FILENAME,
    'predictions':
    PREDICTIONS_FILENAME,
    'checkpoint': CHECKPOINT_FILENAME}

PLOT_BACKEND = 'Agg'
PLOT_DPI = 1 << 7
PREDICTION_GRID = 3, 4
PREDICTION_COUNT = PREDICTION_GRID[0] * PREDICTION_GRID[1]
PREDICTION_FIGSIZE = 10, 8
CURVES_FIGSIZE = 10, 4
CURVES_GRID = 1, 2
CURVES_LINE_STYLE = 'o-'
PREDICTION_FONT_SIZE = 10
CORRECT_COLOR = 'darkgreen'
INCORRECT_COLOR = 'firebrick'
IMAGE_COLOR_MAP = 'gray'
PIXEL_MIN, PIXEL_MAX = 0, 1
GRID_ALPHA = 0.2

UI_CONFIG = {
    'defaults': DEFAULT_CONFIG,
    'quick_default': QUICK_PRACTICE_DEFAULT,
    'quick_epochs': QUICK_EPOCHS,
    'quick_train_samples': QUICK_TRAIN_SAMPLES,
    'train_samples': TRAIN_SAMPLES,
    'integer_limits': INTEGER_LIMITS,
    'learning_rate_min': MIN_LEARNING_RATE,
    'learning_rate_max': MAX_LEARNING_RATE,
    'poll_interval_ms': 1 << 10,
    'request_timeout_ms': 1 << 13,
}
