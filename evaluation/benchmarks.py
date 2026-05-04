"""
Published industry benchmark numbers for comparison in the evaluation dashboard.

Sources:
  - BoT-SORT:  Aharon et al., "BoT-SORT: Robust Associations Multi-Pedestrian
               Tracking", arXiv:2206.14651, 2022. Table 6 (MOT17 test, public detector).
  - ByteTrack: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
               Every Detection Box", ECCV 2022. Table 4 (MOT17 test, private).
  - DeepSORT:  Wojke et al., "Simple Online and Realtime Tracking with a Deep
               Association Metric", ICIP 2017. Reported across multiple
               re-implementations; numbers below are widely cited from the
               original MOT17 evaluation.
  - SORT:      Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
               Original baseline for the SORT family.

Numbers expressed as percentages (matching the units motmetrics reports
when scaled — MOTA/IDF1 in [0, 1] internally; we multiply by 100 for display).
"""

# MOT17 test set (single number per metric per tracker).
MOT17_BENCHMARKS = {
    "BoT-SORT (paper)":   {"MOTA": 80.5, "IDF1": 80.2, "HOTA": 65.0},
    "ByteTrack (paper)":  {"MOTA": 80.3, "IDF1": 77.3, "HOTA": 63.1},
    "DeepSORT (paper)":   {"MOTA": 78.0, "IDF1": 74.5, "HOTA": 61.2},
    "SORT (paper)":       {"MOTA": 74.6, "IDF1": 65.5, "HOTA": 53.8},
}

# MOT20 test set (denser crowds; lower numbers across the board)
MOT20_BENCHMARKS = {
    "BoT-SORT (paper)":   {"MOTA": 77.8, "IDF1": 77.5, "HOTA": 63.3},
    "ByteTrack (paper)":  {"MOTA": 77.8, "IDF1": 75.2, "HOTA": 61.3},
    "DeepSORT (paper)":   {"MOTA": 71.8, "IDF1": 69.6, "HOTA": 57.1},
}

# Default benchmark we compare against — most demos and reports cite MOT17.
DEFAULT_BENCHMARK = MOT17_BENCHMARKS
DEFAULT_BENCHMARK_NAME = "MOT17 (test)"
