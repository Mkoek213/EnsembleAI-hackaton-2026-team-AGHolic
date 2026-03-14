# Predictive Grid Management for Heat Pump Networks

## Background & Context

The mass electrification of heating in Poland is a cornerstone of the energy transition. However, for Distribution System Operators (DSOs), the rapid influx of heat pumps creates significant challenges. Precise forecasting of aggregated power demand is essential to prevent grid overloads and ensure system stability. This challenge focuses on high-resolution data analysis to predict seasonal grid strain.

## Problem Statement

The goal of this challenge is to develop a high-precision predictive system that utilizes historical data from 2024 to forecast the total electrical load for the summer-autumn of 2025. This is an extrapolation task; participants must build models capable of projecting learned behaviors into a new year.

- Timeframe: Forecast May-October 2025 (6 months per device).
- Task Nature: This is an extrapolative challenge. Models must be designed to project demand beyond the provided historical training scope, accounting for shifting patterns in device behavior.
- Objective: Architect a system that generates precise per-device monthly load forecasts.

## Data Specification

The dataset provided is of high resolution, requiring efficient data handling:

- Frequency: Data is recorded at 5-minute intervals.
- Data Management: Due to the substantial volume of data, teams are expected to "slice" or subsample the dataset for efficient training and model development.
- No Synthesis: The dataset consists of real-world telemetry; there is no requirement for generating additional synthetic data.

Download: `ensemble_task_3_data_oct24-oct25.tar.gz`

## Dataset Splits

| Subset | Period | Months | Notes |
| --- | --- | --- | --- |
| Train | Oct 2024 - Apr 2025 | 7 months | Full feature set |
| Validation | May 2025 - Jun 2025 | 2 months | Partial feature set; `x2` withheld |
| Test | Jul 2025 - Oct 2025 | 4 months | Partial feature set; `x2` withheld |

## Feature Reference

### `data.csv`

Time-series telemetry, one row per device per 5-minute interval.

| Column | Feature | Description |
| --- | --- | --- |
| `deviceId` | Device ID | Unique identifier for each unit. |
| `Timedate` | Timestamp | UTC timestamp of the 5-minute reading. |
| `t1` | External temperature | Outdoor ambient temperature; min-max normalised. |
| `t2` | Internal temperature | Indoor ambient temperature; min-max normalised. |
| `t3` | Source HEX temperature 1 | Temperature 1 on the source-side heat exchanger; min-max normalised. |
| `t4` | Source HEX temperature 2 | Temperature 2 on the source-side heat exchanger; min-max normalised. |
| `t5` | Load HEX temperature 1 | Temperature 1 on the receive-side heat exchanger; min-max normalised. |
| `t6` | Load HEX temperature 2 | Temperature 2 on the receive-side heat exchanger; min-max normalised. |
| `t7` | DHW tank temperature | Domestic hot water storage tank temperature; min-max normalised. |
| `t8` | Cooling circuit temperature 1 | Temperature 1 of the cooling circuit; min-max normalised. |
| `t9` | CH buffer temperature | Central heating buffer tank temperature; min-max normalised. |
| `t10` | Source HEX air temperature 1 | Air temperature 1 at the source-side heat exchanger; min-max normalised. |
| `t11` | Source HEX air temperature 2 | Air temperature 2 at the source-side heat exchanger; min-max normalised. |
| `t12` | Load HEX temperature 3 | Temperature 3 on the receiver-side heat exchanger; min-max normalised. |
| `t13` | Load HEX temperature 4 | Temperature 4 on the receiver-side heat exchanger; min-max normalised. |
| `x1` | Operating frequency | Compressor/device operating frequency. |
| `x2` | Grid load indicator | Individual electrical grid load index (prediction target). |
| `x3` | Heating curve type | Type of heating curve configuration used by the device. |
| `deviceType` | Device type | Integer-encoded device category. |

### `devices.csv`

Static per-device metadata.

| Column | Feature | Description |
| --- | --- | --- |
| `deviceId` | Device ID | Unique identifier for each unit (join key). |
| `latitude` | Latitude | Geographic latitude of the device installation site. |
| `longitude` | Longitude | Geographic longitude of the device installation site. |

## Prediction Target

For each device `d` and forecast month `m`, predict the average `x2` over all 5-minute readings within that month:

```text
target_{d,m} = (1 / N_{d,m}) * sum_{i=1}^{N_{d,m}} x2^{(d,m,i)}
```

where `N_{d,m}` is the total number of readings for device `d` in month `m`.

## Submission Format

Submit a single CSV file covering both validation and test months - one row per device per month:

```csv
deviceId,year,month,prediction
000cc3cb...,2025,5,0.18908
000cc3cb...,2025,6,0.18908
000cc3cb...,2025,7,0.21034
...
```

| Column | Type | Description |
| --- | --- | --- |
| `deviceId` | `str` | Device identifier (matches training data). |
| `year` | `int` | Year of the forecast month. |
| `month` | `int` | Month number (5-10 for May-October 2025). |
| `prediction` | `float` | Predicted average `x2` for that device and month. |

## Evaluation Metrics

Submissions are evaluated using MAE (Mean Absolute Error):

```text
MAE = (1 / n) * sum_{i=1}^{n} |y_i - yhat_i|
```

where `n` is the total number of predictions. Lower values indicate better predictions, and a score of `0` corresponds to a perfect prediction.

## Leaderboard vs. Final Score

| Score | Months used | Weights |
| --- | --- | --- |
| Leaderboard (visible) | Validation only (May - Jun 2025) | - |
| Final score | Validation + Test (May - Oct 2025) | `2/6` validation, `4/6` test |

In order to be included in the final leaderboard, a reproducible code implementation for generating the submission should be provided, in the form of a link to a public GitHub repository.

## Additional Information

The original PDF includes a map showing the regional coverage areas of the four largest electricity distribution companies in Poland: Enea, Energa, PGE, and Tauron.

Source: `wnp.pl`
