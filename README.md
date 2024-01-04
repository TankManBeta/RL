# Preface

Personal implementation of some classical reinforcement learning algorithms.

# Structure

Each algorithm has its own directory. Typically, within the directory, there are files and sub-directories. The `code` directory is designated for saving source code. Within the `code` directory, `main.py` is the primary file that initiates the algorithm, while `models.py` contains the models utilized in the respective algorithm. Furthermore, `utils.py` comprises of customized functions such as `run_one_episode`, `train`, and `evaluate`. The `results` directory is utilized for storing results, encompassing checkpoints, log files, and evaluation outcomes.

# Usage

Anyone can run the code as long as they have correctly installed the required dependencies specified in the `requirements.txt` file.

# Notice

+ As the author is new to RL, there may be some errors in the implementation.
+ To make things simpler, the author did not account for error handling.
+ The dependencies in the requirements are redundant, and many of them are not used. The users are recommended to install the dependencies according to the reported errors.