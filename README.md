# Metodi_Monte_Carlo
A repository following the "Metodi Monte Carlo" course at UNIPI


## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/jpsi-scan.git
cd jpsi-scan
```

Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
Run the main simulation:
```python
python main.py
```

This will:
1. Generate pseudo-data at predefined scan points using a Monte-Carlo simulation
2. Compute theoretical curves with and without ISR corrections
3. Show and save plots

## Command line options
You can override default parameters directly from the command line. For example:
```python
python main.py --n_mc 50000 --points 2000 --no-isr --seed 1234
```

Available options:
--n-mc (int): Number of MC samples per scan point

--no-isr: Disable ISR effects in simulation

--n-escan-points (int): Number of theory points for smooth curves

--seed (int): Random seed for reproducibility

--help, -h: Show help message and exit



[MIT](https://choosealicense.com/licenses/mit/)
