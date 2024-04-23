# Copyright 1999-2024 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NULL = 0

# creation
# tensor
SCALAR = 1
TENSOR_DATA_SOURCE = 2
TENSOR_ONES = 3
TENSOR_ONES_LIKE = 4
TENSOR_ZEROS = 5
TENSOR_ZEROS_LIKE = 6
TENSOR_EMPTY = 7
TENSOR_EMPTY_LIKE = 8
TENSOR_FULL = 9
TENSOR_FULL_LIKE = 25
TENSOR_ARANGE = 10
TENSOR_INDICES = 11
TENSOR_DIAG = 12
TENSOR_EYE = 13
TENSOR_LINSPACE = 14
TENSOR_TRIU = 15
TENSOR_TRIL = 16
# external storage
TENSOR_FROM_TILEDB = 18
TENSOR_STORE_TILEDB = 19
TENSOR_STORE_TILEDB_CONSOLIDATE = 20
TENSOR_FROM_DATAFRAME = 22
TENSOR_FROM_HDF5 = 27
TENSOR_STORE_HDF5 = 28
TENSOR_FROM_ZARR = 29
TENSOR_STORE_ZARR = 32

# dataframe
DATAFRAME_DATA_SOURCE = 17
DATAFRAME_FROM_TENSOR = 21
DATAFRAME_FROM_RECORDS = 24
# series
SERIES_DATA_SOURCE = 23
SERIES_FROM_TENSOR = 26
SERIES_FROM_INDEX = 39
# index
INDEX_DATA_SOURCE = 33
DATE_RANGE = 34
TIMEDELTA_RANGE = 35
CHECK_MONOTONIC = 38
# misc
MEMORY_USAGE = 36
REBALANCE = 37

# GPU
TO_GPU = 30
TO_CPU = 31

# random
RAND_RAND = 41
RAND_RANDN = 42
RAND_RANDINT = 43
RAND_RANDOM_INTEGERS = 44
RAND_RANDOM_SAMPLE = 45
RAND_RANDOM = 46
RAND_RANF = 47
RAND_SAMPLE = 48
RAND_BYTES = 49

# random distribution
RAND_BETA = 50
RAND_BINOMIAL = 51
RAND_CHISQUARE = 52
RAND_CHOICE = 53
RAND_DIRICHLET = 54
RAND_EXPONENTIAL = 55
RAND_F = 56
RAND_GAMMA = 57
RAND_GEOMETRIC = 58
RAND_GUMBEL = 59
RAND_HYPERGEOMETRIC = 60
RAND_LAPLACE = 61
RAND_LOGISTIC = 62
RAND_LOGNORMAL = 63
RAND_LOGSERIES = 64
RAND_MULTINOMIAL = 65
RAND_MULTIVARIATE_NORMAL = 66
RAND_NEGATIVE_BINOMIAL = 67
RAND_NONCENTRAL_CHISQURE = 68
RAND_NONCENTRAL_F = 69
RAND_NORMAL = 70
RAND_PARETO = 71
RAND_PERMUTATION = 72
RAND_POSSION = 73
RAND_POWER = 74
RAND_RAYLEIGH = 75
RAND_SHUFFLE = 76
RAND_STANDARD_CAUCHY = 77
RAND_STANDARD_EXPONENTIAL = 78
RAND_STANDARD_GAMMMA = 79
RAND_STANDARD_NORMAL = 80
RAND_STANDARD_T = 81
RAND_TOMAXINT = 82
RAND_TRIANGULAR = 83
RAND_UNIFORM = 84
RAND_VONMISES = 85
RAND_WALD = 86
RAND_WEIBULL = 87
RAND_ZIPF = 88
PERMUTATION = 89
UNIQUE = 90

# ufunc
ADD = 101
SUB = 102
MUL = 103
DIV = 104
TRUEDIV = 105
FLOORDIV = 106
POW = 107
MOD = 108
FMOD = 109
LOGADDEXP = 110
LOGADDEXP2 = 111
NEGATIVE = 112
POSITIVE = 113
ABSOLUTE = 114
FABS = 115
ABS = 116
RINT = 117
SIGN = 118
CONJ = 119
EXP = 120
EXP2 = 121
LOG = 122
LOG2 = 123
LOG10 = 124
EXPM1 = 125
LOG1P = 126
SQRT = 127
SQUARE = 128
CBRT = 129
RECIPROCAL = 130
EQ = 131
NE = 132
LT = 133
LE = 134
GT = 135
GE = 136
SIN = 137
COS = 138
TAN = 139
ARCSIN = 140
ARCCOS = 141
ARCTAN = 142
ARCTAN2 = 143
HYPOT = 144
SINH = 145
COSH = 146
TANH = 147
ARCSINH = 148
ARCCOSH = 149
ARCTANH = 150
DEG2RAD = 151
RAD2DEG = 152
BITAND = 153
BITOR = 154
BITXOR = 155
INVERT = 156
LSHIFT = 157
RSHIFT = 158
AND = 159
OR = 160
XOR = 161
NOT = 162
MAXIMUM = 163
MINIMUM = 164
AROUND = 165
FLOAT_POWER = 166
FMAX = 167
FMIN = 168
ISFINITE = 169
ISINF = 170
ISNAN = 171
SIGNBIT = 172
COPYSIGN = 173
NEXTAFTER = 174
SPACING = 175
LDEXP = 176
FREXP = 177
MODF = 178
FLOOR = 179
CEIL = 180
TRUNC = 181
DEGREES = 182
RADIANS = 183
CLIP = 184
ISREAL = 185
ISCOMPLEX = 186
REAL = 187
IMAG = 188
FIX = 189
I0 = 190
SINC = 191
NAN_TO_NUM = 192
ISCLOSE = 193
DIVMOD = 194
ANGLE = 195
SET_REAL = 196
SET_IMAG = 197

# special
SPECIAL = 200

# spatial
PDIST = 231
CDIST = 232
SQUAREFORM = 233

# tree operator
TREE_ADD = 251
TREE_MULTIPLY = 252
TREE_OR = 253

# reduction
CUMSUM = 301
CUMPROD = 302
PROD = 303
SUM = 304
MAX = 305
MIN = 306
ALL = 307
ANY = 308
MEAN = 309
ARGMAX = 310
ARGMIN = 311
NANSUM = 312
NANMAX = 313
NANMIN = 314
NANPROD = 315
NANMEAN = 316
NANARGMAX = 317
NANARGMIN = 318
COUNT_NONZERO = 319
MOMENT = 320
NANMOMENT = 321
VAR = 322
STD = 323
NANVAR = 324
NANSTD = 325
NANCUMSUM = 326
NANCUMPROD = 327
COUNT = 343
CUMMAX = 344
CUMMIN = 345
CUMCOUNT = 346
CORR = 347
REDUCTION_SIZE = 348
CUSTOM_REDUCTION = 349
SKEW = 350
KURTOSIS = 351
SEM = 352
STR_CONCAT = 353
MAD = 354

# tensor operator
RESHAPE = 401
SLICE = 402
INDEX = 403
INDEXSETVALUE = 404
CONCATENATE = 405
RECHUNK = 406
ASTYPE = 407
TRANSPOSE = 408
SWAPAXES = 409
BROADCAST_TO = 410
STACK = 411
WHERE = 412
CHOOSE = 413
NONZERO = 414
ARGWHERE = 415
UNRAVEL_INDEX = 416
RAVEL_MULTI_INDEX = 417
ARRAY_SPLIT = 418
SQUEEZE = 419
DIGITIZE = 420
REPEAT = 421
COPYTO = 422
ISIN = 423
SEARCHSORTED = 428
SORT = 429
HISTOGRAM = 430
HISTOGRAM_BIN_EDGES = 431
PARTITION = 432
QUANTILE = 440
FILL_DIAGONAL = 441
NORMALIZE = 442
TOPK = 443
TRAPZ = 444
GET_SHAPE = 445
BINCOUNT = 446
# fancy index, distributed phase is a shuffle operation that
# the fancy indexes will be distributed to the left chunks
# the concat phase will concat back the indexed left chunks and index
# according to the original fancy index order
FANCY_INDEX_DISTRIBUTE = 424
FANCY_INDEX_CONCAT = 425

# linear algebra
TENSORDOT = 501
DOT = 502
MATMUL = 503
CHOLESKY = 510
QR = 511
SVD = 512
LU = 513
SOLVE_TRIANGULAR = 520
INV = 521
NORM = 530

# fft
FFT = 601
IFFT = 602
FFT2 = 603
IFFT2 = 604
FFTN = 605
IFFTN = 606
RFFT = 607
IRFFT = 608
RFFT2 = 609
IRFFT2 = 610
RFFTN = 611
IRFFTN = 612
HFFT = 613
IHFFT = 614
FFTFREQ = 615
FFTFREQ_CHUNK = 616
RFFTFREQ = 617
FFTSHIFT = 618
IFFTSHIFT = 619

# einsum
EINSUM = 630

# sparse creation
SPARSE_MATRIX_DATA_SOURCE = 701
DENSE_TO_SPARSE = 702
SPARSE_TO_DENSE = 703

# DataFrame
MAP = 710
DESCRIBE = 712
FILL_NA = 713
AGGREGATE = 714
STRING_METHOD = 715
DATETIME_METHOD = 716
APPLY = 717
TRANSFORM = 718
CHECK_NA = 719
DROP_NA = 720
NUNIQUE = 721
CUT = 722
SHIFT = 723
DIFF = 724
VALUE_COUNTS = 725
TO_DATETIME = 726
DATAFRAME_DROP = 727
DROP_DUPLICATES = 728
MELT = 729
RENAME = 731
INSERT = 732
MAP_CHUNK = 733
CARTESIAN_CHUNK = 734
EXPLODE = 735
REPLACE = 736
RENAME_AXIS = 737
DATAFRAME_EVAL = 738
DUPLICATED = 739
DELETE = 740
ALIGN = 741

FUSE = 801

# table like input for tensor
TABLE_COO = 1003
# store tensor as coo format
STORE_COO = 1004

# shuffle
SHUFFLE_PROXY = 2001
DATAFRAME_INDEX_ALIGN = 2004

# indexing
DATAFRAME_SET_INDEX = 2020
DATAFRAME_SET_AXIS = 730
DATAFRAME_ILOC_GETITEM = 2021
DATAFRAME_ILOC_SETITEM = 2022
DATAFRAME_LOC_GETITEM = 2023
DATAFRAME_LOC_SETITEM = 2024

# merge
DATAFRAME_MERGE = 2010
DATAFRAME_SHUFFLE_MERGE_ALIGN = 2011

# bloom filter
DATAFRAME_BLOOM_FILTER = 2014

# append
APPEND = 2015

# reset index
RESET_INDEX = 2028
# reindex
REINDEX = 2029

# groupby
GROUPBY = 2030
GROUPBY_AGG = 2033
GROUPBY_CONCAT = 2034
GROUPBY_HEAD = 2035
GROUPBY_SAMPLE_ILOC = 2036
GROUPBY_SORT_REGULAR_SAMPLE = 2037
GROUPBY_SORT_PIVOT = 2038
GROUPBY_SORT_SHUFFLE = 2039

# parallel sorting by regular sampling
PSRS_SORT_REGULAR_SMAPLE = 2040
PSRS_CONCAT_PIVOT = 2041
PSRS_SHUFFLE = 2042
PSRS_ALIGN = 2043
# partition
CALC_PARTITIONS_INFO = 2046
PARTITION_MERGED = 2047

# dataframe sort
SORT_VALUES = 2050
SORT_INDEX = 2051

# window
ROLLING_AGG = 2060
EXPANDING_AGG = 2061
EWM_AGG = 2062

# source & store
READ_CSV = 2100
TO_CSV = 2101
READ_PARQUET = 2103
TO_PARQUET = 2104
READ_SQL = 2105
TO_SQL = 2108
READ_RAYDATASET = 2109
READ_MLDATASET = 2106
READ_ODPS_TABLE = 20111
TO_ODPS_TABLE = 20112
READ_ODPS_VOLUME = 20113
TO_ODPS_VOLUME = 20114
READ_ODPS_QUERY = 20115

TO_CSV_STAT = 2102

# standardize range index
STANDARDIZE_RANGE_INDEX = 2107

# successors exclusive
SUCCESSORS_EXCLUSIVE = 2002

# read images
IMREAD = 2110

# machine learning

# pairwise distances
PAIRWISE_EUCLIDEAN_DISTANCES = 2200
PAIRWISE_MANHATTAN_DISTANCES = 2201
PAIRWISE_COSINE_DISTANCES = 2202
PAIRWISE_HAVERSINE_DISTANCES = 2203
PAIRWISE_DISTANCES_TOPK = 2204

# nearest neighbors
KD_TREE_TRAIN = 2230
KD_TREE_QUERY = 2231
BALL_TREE_TRAIN = 2232
BALL_TREE_QUERY = 2233
FAISS_BUILD_INDEX = 2234
FAISS_TRAIN_SAMPLED_INDEX = 2235
FAISS_QUERY = 2236
PROXIMA_SIMPLE_BUILDER = 2238
PROXIMA_SIMPLE_SEARCHER = 2239
KNEIGHBORS_GRAPH = 2237

# cluster
KMEANS_PLUS_PLUS_INIT = 2250
KMEANS_SCALABLE_PLUS_PLUS_INIT = 2251
KMEANS_ELKAN_INIT_BOUNDS = 2252
KMEANS_ELKAN_UPDATE = 2253
KMEANS_ELKAN_POSTPROCESS = 2254
KMEANS_LLOYD_UPDATE = 2255
KMEANS_LLOYD_POSTPROCESS = 2256
KMEANS_INERTIA = 2257
KMEANS_RELOCASTE_EMPTY_CLUSTERS = 2258

# XGBoost
XGBOOST_TRAIN = 3001
XGBOOST_PREDICT = 3002
TO_DMATRIX = 3003
START_TRACKER = 3004

# LightGBM
LGBM_TRAIN = 3020
LGBM_PREDICT = 3021
LGBM_ALIGN = 3022

# TensorFlow
RUN_TENSORFLOW = 3010

# PyTorch
RUN_PYTORCH = 3011

# statsmodels
STATSMODELS_TRAIN = 3012
STATSMODELS_PREDICT = 3013

# learn
# checks
CHECK_NON_NEGATIVE = 3300
# classifier check targets
CHECK_TARGETS = 3301
ASSERT_ALL_FINITE = 3302
# multilabel
IS_MULTILABEL = 3303
# get type
TYPE_OF_TARGET = 3304
# classification
ACCURACY_SCORE = 3305
# port detection
COLLECT_PORTS = 3306
# unique labels
UNIQUE_LABELS = 3307
# preprocessing
LABEL_BINARIZE = 3308
# ensemble: blockwise
BLOCKWISE_ENSEMBLE_FIT = 3309
BLOCKWISE_ENSEMBLE_PREDICT = 3310
# ensemble: bagging
BAGGING_SHUFFLE_SAMPLE = 3400
BAGGING_SHUFFLE_REINDEX = 3401
BAGGING_FIT = 3402
BAGGING_PREDICTION = 3403

# Remote Functions and class
REMOTE_FUNCATION = 5001
RUN_SCRIPT = 5002

CHOLESKY_FUSE = 999988

# MaxFrame-dedicated functions
DATAFRAME_RESHUFFLE = 10001

# MaxFrame internal operators
DATAFRAME_PROJECTION_SAME_INDEX_MERGE = 100001
GROUPBY_AGGR_SAME_INDEX_MERGE = 100002
DATAFRAME_ILOC_GET_AND_RENAME_ITEM = 100003

# fetches
FETCH_SHUFFLE = 999998
FETCH = 999999


_val_to_dict = dict()
for _var_name, _var_val in globals().copy().items():
    if not isinstance(_var_val, int):
        continue
    if _var_val in _val_to_dict:  # pragma: no cover
        raise ImportError(
            f"Cannot import opcode: {_var_name} and "
            f"{_val_to_dict[_var_val]} collides with value {_var_val}"
        )
    _val_to_dict[_var_val] = _var_name
del _val_to_dict, _var_name, _var_val
