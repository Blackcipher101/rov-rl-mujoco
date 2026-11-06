| density | current (const/amp m/s) | controller | success | collision | timeout | time to goal [s] | path ratio | min clearance mean / worst [m] | thrust impulse [N s] |
|---|---|---|---|---|---|---|---|---|---|
| low | 0 (0.00/0.00) | baseline | 1.00 | 0.00 | 0.00 | 44.2 | 1.11 | 1.18 / 0.54 | 5449 |
| low | 0 (0.00/0.00) | rl | 0.90 | 0.10 | 0.00 | 39.8 | 1.01 | 0.68 / -0.02 | 4760 |
| low | 2 (0.20/0.06) | baseline | 1.00 | 0.00 | 0.00 | 45.6 | 1.09 | 1.17 / 0.42 | 6304 |
| low | 2 (0.20/0.06) | rl | 0.90 | 0.10 | 0.00 | 42.8 | 1.01 | 0.77 / -0.02 | 5783 |
| low | 3 (0.30/0.09) | baseline | 1.00 | 0.00 | 0.00 | 48.5 | 1.10 | 1.10 / 0.38 | 6737 |
| low | 3 (0.30/0.09) | rl | 0.85 | 0.15 | 0.00 | 49.9 | 1.01 | 0.68 / -0.01 | 6721 |
| medium | 0 (0.00/0.00) | baseline | 1.00 | 0.00 | 0.00 | 44.9 | 1.21 | 1.09 / 0.47 | 5535 |
| medium | 0 (0.00/0.00) | rl | 0.95 | 0.05 | 0.00 | 38.6 | 1.03 | 0.35 / -0.00 | 4548 |
| medium | 2 (0.20/0.06) | baseline | 1.00 | 0.00 | 0.00 | 48.0 | 1.34 | 1.10 / 0.33 | 6094 |
| medium | 2 (0.20/0.06) | rl | 0.80 | 0.20 | 0.00 | 40.5 | 1.07 | 0.22 / -0.01 | 5113 |
| medium | 3 (0.30/0.09) | baseline | 0.95 | 0.00 | 0.05 | 57.3 | 1.35 | 0.75 / 0.24 | 7785 |
| medium | 3 (0.30/0.09) | rl | 0.65 | 0.35 | 0.00 | 44.8 | 1.09 | 0.25 / -0.02 | 5929 |
| high | 0 (0.00/0.00) | baseline | 1.00 | 0.00 | 0.00 | 53.6 | 1.34 | 0.90 / 0.42 | 6507 |
| high | 0 (0.00/0.00) | rl | 0.75 | 0.25 | 0.00 | 43.0 | 1.07 | 0.11 / -0.01 | 4956 |
| high | 2 (0.20/0.06) | baseline | 1.00 | 0.00 | 0.00 | 56.0 | 1.35 | 0.75 / 0.39 | 7188 |
| high | 2 (0.20/0.06) | rl | 0.45 | 0.55 | 0.00 | 46.8 | 1.08 | 0.08 / -0.01 | 5990 |
| high | 3 (0.30/0.09) | baseline | 0.95 | 0.00 | 0.05 | 58.6 | 1.35 | 0.75 / 0.31 | 8240 |
| high | 3 (0.30/0.09) | rl | 0.80 | 0.20 | 0.00 | 52.5 | 1.11 | 0.19 / -0.01 | 7152 |
| all | all | baseline | 0.99 | 0.00 | 0.01 | 50.7 | 1.25 | 0.98 / 0.24 | 6633 |
| all | all | rl | 0.78 | 0.22 | 0.00 | 44.0 | 1.05 | 0.37 / -0.02 | 5617 |

success / collision / timeout: fractions of all episodes (timeout includes tilt failures). Time to goal, path ratio (path length / straight start-goal distance) and thrust impulse: mean over successful episodes. Min clearance: true gap between the vehicle bounding sphere and the nearest obstacle or the seabed, per episode; mean and worst over all episodes.
