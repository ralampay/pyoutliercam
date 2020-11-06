import pandas as pd
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.stats import iqr

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from utils import create_histogram
from utils import htb
from utils import fetch_threshold

class OccAutoencoder:
    def __init__(self, autoencoder, n_samples=1000, s_dimension_count=2, relative_frequency=0.1):
        self.autoencoder        = autoencoder
        self.n_samples          = n_samples
        self.s_dimension_count  = s_dimension_count
        self.relative_frequency = relative_frequency

    def predict(self, data):
        predictions = []

        reconstructed_data  = self.autoencoder.predict(data)
        error_values        = np.power((reconstructed_data - data), 2)

        self.predicted_mean_sq_errors = np.mean(error_values, axis=1)

        for e in self.predicted_mean_sq_errors:
            if e > self.optimal_threshold:
                predictions.append(-1)
            else:
                predictions.append(1)

        return predictions

    def sample_tails(self, mu, sig, n=1000):
        samples = np.array(sorted(np.random.normal(mu, sig, n)))

        q1, q3 = np.percentile(samples, [25, 75])
        iqr = q3 - q1

        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)

        return np.concatenate((samples[samples < lower_bound], samples[samples > upper_bound]))

    def build(self, data):
        self.original_data      = data
        self.encoded_data       = self.autoencoder.encode(data)
        self.reconstructed_data = self.autoencoder.decode(self.encoded_data)

        self.df_encoded_data        = pd.DataFrame(data=self.encoded_data)
        self.df_encoded_data_mean   = self.df_encoded_data.mean(axis=0)
        self.df_encoded_data_std    = self.df_encoded_data.std(axis=0)

        self.stochastic_dimensions  = random.sample(range(len(self.df_encoded_data.columns)), self.s_dimension_count)
        self.num_to_synthesize      = round(len(data) * self.relative_frequency)

        if self.num_to_synthesize > 0:
            self.synthetic_latent_data  = ((self.df_encoded_data.sample(self.num_to_synthesize)).reset_index(drop=True)).copy()
        else:
            self.synthetic_latent_data  = self.df_encoded_data


        if self.n_samples > 0:
            for index, row in self.synthetic_latent_data.iterrows():
                for d in self.stochastic_dimensions:
                    tail_values = self.sample_tails(self.df_encoded_data_mean.values[d], self.df_encoded_data_std.values[d], self.n_samples)

                    if len(tail_values) == 0:
                        outlier_v   = self.synthetic_latent_data.at[index, d]
                    else:
                        outlier_v   = random.choice(tail_values)

                    self.synthetic_latent_data.at[index, d] = outlier_v

        # Reconstruct using frozen weights
        self.synthetic_data     = self.autoencoder.decode(self.synthetic_latent_data.values)
        self.df_synthetic_data  = pd.DataFrame(data=self.synthetic_data)

        self.synthetic_data_with_labels     = np.append(self.synthetic_data, np.ones((len(self.synthetic_data), 1)), axis=1)
        self.reconstructed_data_with_labels = np.append(self.reconstructed_data, np.zeros((len(self.reconstructed_data), 1)), axis=1)

        # Reconstructed synthetic data
        self.reconstructed_synthetic    = self.autoencoder.predict(self.synthetic_data)

        self.X      = np.concatenate((self.original_data, self.synthetic_data))
        self.Y      = np.concatenate((self.reconstructed_data, self.reconstructed_synthetic))
        self.errors         = np.power(self.X - self.Y, 2)
        self.mean_sq_errors = np.mean(self.errors, axis=1)

        # Calculate the number of bins according to Freedman-Diaconis rule
        bin_width       = 2 * iqr(self.mean_sq_errors) / np.power(len(self.errors), (1/3))
        num_bins        = (np.max(self.mean_sq_errors) - np.min(self.mean_sq_errors)) / bin_width

        self.hist, self.bins = create_histogram(self.mean_sq_errors, num_bins=num_bins, step=bin_width)
        self.occurences      = [float(x) for x in self.hist.tolist()]    # Convert to float data type

        breaks = htb(self.hist)
        self.possible_thresholds = []

        for b in breaks:
            t = fetch_threshold(self.bins, self.hist, b)
            self.possible_thresholds.append(t)

        self.optimal_threshold = max(self.possible_thresholds)

        # Create labels for histogram rendering
        self.labels = []
        for i in range(len(self.bins) - 1):
            self.labels.append(str(round(self.bins[i], 4)) + "-" + str(round(self.bins[i + 1], 4)))

    def render(self):
        plt.rcParams['figure.figsize'] = [10, 8]
        plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower

        index = np.arange(len(self.labels))
        barlist = plt.bar(index, self.occurences)

        # Draw a red line to indicate position of threshold
        #plt.axvline(x=self.optimal_threshold, c='r')
        for i in range(len(self.bins) - 1):
            if self.optimal_threshold > self.bins[i] and self.optimal_threshold < self.bins[i + 1]:
                barlist[i].set_color('r')

        plt.xlabel('Error')
        plt.ylabel('Occurences')
        plt.xticks(index, self.labels, fontsize=5)
        plt.title('Histogram of Residual Errors')
        plt.grid(False)
        plt.show()
