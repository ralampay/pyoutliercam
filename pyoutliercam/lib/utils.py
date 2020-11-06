import numpy as np

def htb(data):
    outp = []

    def htb_inner(data):
        """
        Inner ht breaks function for recursively computing the break points.
        """
        data_length = float(len(data))

        data_mean = 0
        if data_length > 0:
            data_mean   = sum(data) / data_length
        head = [_ for _ in data if _ > data_mean]
        outp.append(data_mean)

        while len(head) > 1 and len(head) / data_length < 0.40:
            return htb_inner(head)

    htb_inner(data)

    return outp

def fetch_threshold(bins, counts, break_point):
    index       = 0
    latest_min  = 99999999
    threshold   = -1

    for i in range(len(counts)):
        if abs(counts[i] - break_point) <= latest_min:
            latest_min = abs(counts[i] - break_point)
            index = i
            threshold = ((bins[i + 1] - bins[i]) / 2) + bins[i]
    
    return threshold

def create_histogram(data, num_bins=100, step=-1):
    min_bin = np.min(data)
    max_bin = np.max(data) + min_bin

    if step < 0:
        step    = (max_bin - min_bin) / num_bins

    bins    = np.arange(min_bin, max_bin, step)

    (hist, bins) = np.histogram(data, bins=bins)

    return (hist,bins)
