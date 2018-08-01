import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(in_directory):
	df = pd.read_json(in_directory, lines=True)
	print(df)

if __name__ == "__main__":
	in_directory = sys.argv[1]
	main(in_directory)