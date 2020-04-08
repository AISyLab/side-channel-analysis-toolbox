# Side-channel Attack Project
<!---
[![build status](https://gitlab.ewi.tudelft.nl/TI2806/2018-2019/CS/cp19-cs-11/cs-11/badges/development/pipeline.svg)](https://gitlab.ewi.tudelft.nl/TI2806/2018-2019/CS/cp19-cs-11/cs-11/commits/development)
[![coverage report](https://gitlab.ewi.tudelft.nl/TI2806/2018-2019/CS/cp19-cs-11/cs-11/badges/development/coverage.svg)](https://gitlab.ewi.tudelft.nl/TI2806/2018-2019/CS/cp19-cs-11/cs-11/commits/development)
--->

This is a project in which side-channel attacks are researched and developed.

## Features

The software includes a number of features.

Side-channel attacks included in current version:
* Template attack
* Pooled template attack
* Stochastic attack
* Mutual information analysis

Analysis tools included in current version:
* Normalized inter class variance analysis
* Pearson correlation analysis

Other features include:
* Progressbar
* Logging of sub result

## Getting started

<!---

This section should contain installation, testing, and running instructions for people who want to get started with the project.

- These instructions should work on a clean system.
- These instructions should work without having to install an IDE.
- You can specify that the user should have a certain operating system.

--->

### Requirements
This project needs to following to run:
* Python 3.7.x
* A CUDA enabled grapics card with Compute Capability >= 3.5
    - You can check the CC of your card here: https://developer.nvidia.com/cuda-gpus
* The CUDA SDK (CUDA 10.x)

### Installing packages
In a terminal, run
```bash
python setup.py install
```
in the root directory (`cs-11/` most likely) to install most of the required dependencies.

### Running
Python can't find our modules by itself, so we need to tell Python where they are.
The command below will append the current directory (which should be the project root directory)
to the `PYTHONPATH` environment variable.

On Linux or if using a Linux shell on Windows:
````bash
export PYTHONPATH=$(pwd)
````
On Windows' `cmd`:
```
set PYTHONPATH=%cd%
```

Then you can run the program:
```bash
python sca
```

### Running tests
To verify the correctness of your installation, the test suite can be executed with the following command:
```bash
python setup.py test
```

## Documentation

The side-channel attack (sca) program allows the following side-channel attacks to be run:
* Mutual Information Analysis Attack (mia)
* Stochastic Attack (sa)
* (Pooled) Template Attack (ta)
* Online Correlation Power Analysis (cpa-online)

These simple attacks can be run using the following commands

```bash
python sca mia
python sca sa
python sca ta
python sca cpa-online
```

### Flags for the attacks
| Flag | Info | Default | Compatibility |
|------|------|---------|---------------|
| --traces-file, -tf | The path for the traces to be used | Default data set | ALL |
| --keys-file, -kf | The path for the keys to be used | Default data set | ALL |
| --plain-file, -pf | The path for the plain to be used | Default data set | ALL |
| --subkey | The subkey to find; must be in the range [0-15]. When leaving this option out, the whole key is calculated. | MIA: 0, SA: ALL, TA: ALL, CPA-online: ALL | ALL |
| --traces, -t | The number of traces to run the attack with. | MIA: 1000, SA: 4000, TA: 30, CPA-online: 100 | ALL |
| --num_attack-traces, -a | The number of traces which will be attacked | MIA: 1000, SA: 30, TA: - | MIA, SA, TA |
| --round, -r | The round to attack on. | 1 | MIA, SA, TA |
| --num-features, -f | Specify the number of features | - | MIA, SA |
| --op-substitution | Attack the SubBytes operation. | 0 |  MIA, SA, TA |
| --op-shift-rows | Attack the ShiftRows operation. | 0 |  MIA, SA, TA |
| --op-mix-columns | Attack the MixColumns operation. | 0 |  MIA, SA, TA |
| --op-add-round-key | Attack the AddRoundKey operation. | 0 |  MIA, SA, TA |
| --fs-none | Don\'t use feature selection | SOST | MIA, SA |
| --fs-pearson | Use pearson for feature selection | SOST | MIA, SA |
| --fs-sost | Use SOST for feature selection | SOST | MIA, SA |
| --fs-nicv | Use Nicv for feature selection | SOST | MIA, SA |
| --leakage-model, -lm | The leakage model used, the default is the hamming weights model, enable the flag to switch to the intermediate values model. | True |  MIA, SA, TA |
| --debug-mode, -d | 'Enables debug mode, a mode in which more detailed information about the execution of the attack is printed and logged | False |  MIA, SA, TA |
| --points-of-interest, -i | The number of points of interest. | 5 | TA |
| --pooled, -p | Perform pooled template attack instead of normal template attack. | False | TA |
| --spacing, -s | The spacing between the points of interest. | 1 | TA |

### Examples
Running just the simple mia attack on the default traces will not always yield the same subkey guess.
This behaviour is caused by the number of traces that are used to run the attack. Increasing the number of traces will give more consistent results

```bash
python sca mia --traces 2000
```

Running an attack with custom numpy datasets can be done as follows

```bash
python sca ta -tf data/custom_traces.npy -kf data/custom_keys.npy -pf data/custom_plain.npy
```

It is possible to attack the AES implementation on different encryption rounds as well as the operation to attack.

This example shows how to attack AES on the shift-rows operation on round three using stochastic attack.
```bash
python sca sa --op-shift-rows --round 3
```

The program provides multiple feature selection tools to use. The program also allows you to run an attack without any feature selection tool.
However, not using a feature selection tool will sometimes increase the computation time drastically.

This example shows how to use the pearson feature selection instead of the default SOST feature selection.
```bash
python sca mia --fs-pearson
```

### References

If you use this code, please consider citing:

    @misc{SCAT,
      author = {Wolfgang Bubberman, Sengim Karayalcin, Matthias Meester, Olaf Braakman, and Stjepan Picek},
      title  = {{Side-channel Analysis Toolbox}},
      note   = {{\url{https://github.com/AISyLab/side-channel-analysis-toolbox}}}
      year   = {2020}
    }

<!-- comment to break blocks -->
