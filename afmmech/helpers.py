from scipy.special import dawsn
from scipy.interpolate import splrep, splev
import pandas as pd
import numpy as np
from scipy.optimize import  minimize
import os
import pickle
from igor.binarywave import load as loadibw

def next_path(path_pattern):
    """
    https://stackoverflow.com/questions/17984809/how-do-i-create-an-incrementing-filename-in-python
    Finds the next free path in an sequentially named list of files
    e.g. path_pattern = 'file-%s.txt':
    file-1.txt
    file-2.txt
    file-3.txt
    Runs in log(n) time where n is the number of existing files in sequence
    """
    i = 1

    # First do an exponential search
    while os.path.exists(path_pattern % i):
        i = i * 2

    # Result lies somewhere in the interval (i/2..i]
    # We call this interval (a..b] and narrow it down until a + 1 = b
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b

def get_files(directory, req_ext=None):
    '''
    gets all the files in the given directory
    :param directory: str directory from which you want to load files from
    :param req_ext: optional str required tc_data extension
    :return: list of str names of the files in the given directory
    '''
    if req_ext is None:
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    else:
        return [os.path.join(directory, f) for f in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, f)) and req_ext in f]
        

def get_folders(directory):
    '''
    gets all the folders in the given directory
    :param directory: str directory from which you want the sub-directories
    :return: list of str names of the sub-directories
    '''
    return [f.path for f in os.scandir(directory) if f.is_dir()]

# appropriating some functions from from https://github.com/N-Parsons/ibw-extractor
def from_repr(s):
    """Get an int or float from its representation as a string"""
    # Strip any outside whitespace
    s = s.strip()
    # "NaN" and "inf" can be converted to floats, but we don't want this
    # because it breaks in Mathematica!
    if s[1:].isalpha():  # [1:] removes any sign
        rep = s
    else:
        try:
            rep = int(s)
        except ValueError:
            try:
                rep = float(s)
            except ValueError:
                rep = s
    return rep


def fill_blanks(lst):
    """Convert a list (or tuple) to a 2 element tuple"""
    try:
        return (lst[0], from_repr(lst[1]))
    except IndexError:
        return (lst[0], "")


def flatten(lst):
    """Completely flatten an arbitrarily-deep list"""
    return list(_flatten(lst))


def _flatten(lst):
    """Generator for flattening arbitrarily-deep lists"""
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        elif item not in (None, "", b''):
            yield item


def process_notes(notes):
    """Splits a byte string into an dict"""
    # Decode to UTF-8, split at carriage-return, and strip whitespace
    note_list = list(map(str.strip, notes.decode(errors='ignore').split("\r")))
    note_dict = dict(map(fill_blanks, [p.split(":") for p in note_list]))

    # Remove the empty string key if it exists
    try:
        del note_dict[""]
    except KeyError:
        pass
    return note_dict


def ibw2dict(filename):
    """Extract the contents of an *ibw to a dict"""
    data = loadibw(filename)
    wave = data['wave']

    # Get the labels and tidy them up into a list
    labels = list(map(bytes.decode,
                      flatten(wave['labels'])))

    # Get the notes and process them into a dict
    notes = process_notes(wave['note'])

    # Get the data numpy array and convert to a simple list
    wData = np.nan_to_num(wave['wData']).tolist()

    # Get the filename from the file - warn if it differs
    fname = wave['wave_header']['bname'].decode()
    input_fname = os.path.splitext(os.path.basename(filename))[0]
    if input_fname != fname:
        print("Warning: stored filename differs from input file name")
        print("Input filename: {}".format(input_fname))
        print("Stored filename: {}".format(str(fname) + " (.ibw)"))

    return {"filename": fname, "labels": labels, "notes": notes, "data": wData}


def ibw2df(filename):
    data = ibw2dict(filename)
    headers = data['labels']
    return pd.DataFrame(data['data'], columns=headers)


def load(file_path):
    """
    Load data from a file using either ibw loader numpy, pandas, or pickle.

    Parameters
    ----------
    file_path : str
        The name or path of the file to load.

    Returns
    -------
    data : numpy.ndarray or pandas.DataFrame or object
        The data loaded from the file.

    Raises
    ------
    ValueError
        If the file extension is not recognized or if there is an error loading the data.
    """
    try:
        ext = os.path.splitext(file_path)[1]

        if ext in ('.xlsx', '.csv', '.txt'):
            # Load data from an Excel or CSV file using pandas
            if ext == '.xlsx':
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # Drop any index columns
            df.reset_index(drop=True, inplace=True)

            # Estimate headers if not available
            if df.columns.duplicated().any():
                df = pd.read_csv(file_path, header=None)
                df.columns = [f"Column{i}" for i in range(1, len(df.columns) + 1)]

            # Return the data as a DataFrame
            return df
        
        elif ext == '.ibw':
           # Load data from an ibw file using ibw2df
           return ibw2df(file_path)

        elif ext == '.pkl':
            # Load data from a pickle file using pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data

    except Exception as e:
        raise ValueError(f"Error loading data from file '{file_path}': {e}")


def save(data, file_path, file_type, overwrite=False, lock=False):
    """
    Save data to a file in a specified format.

    Parameters
    ----------
    data : numpy.ndarray or pandas.DataFrame or object
        The data to save.
    file_path : str
        The file path to save the data to.
    file_type : str
        The file type to use for saving the data.
    overwrite : bool, optional
        Whether to allow overwriting existing files (default is False).
    lock : bool, optional
        if true, no overwriting will be allowed, neither will a new file be created (default is False).

    Raises
    ------
    ValueError
        If the file type is not recognized, if the data cannot be saved in the specified format,
        or if there is an error saving the data.
    """
    try:
        ext = os.path.splitext(file_path)[1]

        # If the file already exists and overwrite is False, get the next free file path
        if os.path.exists(file_path) and not overwrite:
            if lock:
                raise ValueError(f"File '{file_path}' already exists")
            file_path = next_path(file_path)

        if file_type == 'pkl':
            # Save data as a pickle file using pickle
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

        elif file_type in ('csv', 'xlsx', 'txt'):
            # Save data as a CSV, Excel, or text file using pandas
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            if file_type == 'csv':
                data.to_csv(file_path, index=False)
            elif file_type == 'xlsx':
                data.to_excel(file_path, index=False)
            elif file_type == 'txt':
                data.to_csv(file_path, index=False, sep='\t')

        else:
            # Raise an error if the file type is not recognized
            raise ValueError(f"Unrecognized file type '{file_type}'")

    except TypeError as e:
        raise ValueError(f"Error saving data to file '{file_path}': {e}. Please check that the data can be saved in the specified format.")
    except Exception as e:
        raise ValueError(f"Error saving data to file '{file_path}': {e}")
    

# format data
def format_df(df, k, sampling_frequency):
    force = df['Defl'].values.copy() * k
    inden = (df['ZSnsr'] - df['Defl']).values.copy()
    tip_pos = (df['Defl'] - df['ZSnsr']).values.copy()  #          CHECK IF THIS IS ACCURATE
    dt = 1 / sampling_frequency
    time = np.arange(0, tip_pos.size) * dt
    return pd.DataFrame({'force': force, 'indentation': inden, 'tip_position': tip_pos, 'base_position': df['ZSnsr'].values.copy(), 'time': time})

# smooth
def smooth_sma(elements, percent=0.05):    # VERIFY IF CORRECT IMPLEMENTATION OF CONVOLVE FILTERING
    if percent > 0.2:
        print(f'{percent} is a large smoothing percentage!')
    window_size = int(percent * len(elements))
    if window_size == 0:
        window_size = 1
    return np.convolve(elements, np.ones(window_size) / window_size, 'valid')

# pick trigger point
def get_trigger_point_index(df, guess_percent=0.8):
    j_initial = int(df.shape[0] * guess_percent)
    return np.argmax(df.force.values[:j_initial])    

# pick contact point - basic
def get_contact_point_index_basic(df, trigger_point_index):
    return np.argmin(df.force.values[: trigger_point_index])

# gui to pick contact point

# code to take contact point and cut the repulsive part of the data
def cut_repulsive_data(df, contact_point_index, trigger_point_index):
    return df[contact_point_index: trigger_point_index]

# shift data
def shift_repulsive_data(df):
    return df - df.iloc[0]

# force - indentation models
# hertz
def force_hertz(params, indentation, R, nu=0.5):
    E = params
    a = np.sqrt(R) * 4 / (3 * (1 - nu ** 2))
    return a * E * abs(indentation) ** 1.5
# convolution sls
def force_conv(params, indentation, t, R, nu=0.5):
    Eg, Ee, Tau = params
    a = np.sqrt(R) * 4 / (3 * (1 - nu ** 2))
    dt = t[1] - t[0]
    return a * (Eg * abs(indentation) ** 1.5 - (Eg - Ee) / Tau * np.convolve(np.exp(-t / Tau), abs(indentation) ** 1.5, 'full')[: t.size] * dt)
    
# ramp
def get_speed(df):
    return np.diff(df.indentation.values.copy()) / np.diff(df.time.values.copy())

def get_speed_spline(df, smoothing=1):
    y = df.indentation.values.copy()
    x = df.time.values.copy()
    spline = splrep(x, y, s=smoothing)
    return np.diff(splev(x, spline)) / np.diff(x)

def force_ramp(params, indentation, speed, t, R, nu=0.5):
    Eg, Ee, Tau = params
    a = speed ** 1.5 * np.sqrt(R) * 4 / (3 * (1 - nu ** 2))
    elastic_part = Ee * t ** 1.5
    second_part = np.sqrt(Tau) * dawsn(np.sqrt(t / Tau))
    first_part = np.sqrt(t)
    return a * (elastic_part + 1.5 * (Eg - Ee) * (first_part - second_part))

def indentation_creep_sls(params, force, t, R, nu=0.5):
    alpha = 4 * np.sqrt(R) / (3 * (1 - nu ** 2))  # contact mechanics term
    Eg, Ee, Tau = params
    # get force magnitude
    force_magnitude = np.mean(force)
    return (force_magnitude / alpha * (1 / Ee + (1 / Eg - 1 / Ee) * np.exp(-t * Ee / (Tau * Eg)))) ** (2 / 3)

# frequency - sls
def q_sls(params, w):
    Eg, Ee, Tau = params
    return Eg - (Eg - Ee) / (1 + Tau * 1j * w)

# frequency - sls absolute modulus
def q_abs_sls(params, w):
    return np.abs(q_sls(params, w))


# fitting models to data
def viscoelastic_constraint(params, x, y):
    Eg, Ee, Tau = params
    return (Ee - Eg) ** 2 * (Ee > Eg) * 1e9 + np.sum(params ** 2 * (params < 0)) * 1e9

def l2_obj(params, y, x, y_function, constraint_function=None, **kwargs):
    y_predicted = y_function(params, x, **kwargs)
    constraint = constraint_function(params, x, y) if constraint_function else 0
    return np.mean((y - y_predicted) ** 2) + constraint
    
def fit_function_to_data(x0, y, x, y_function, constraint_function=None, **kwargs):
    res = minimize(lambda p: l2_obj(p, y, x, y_function, constraint_function, **kwargs), x0, method='Nelder-Mead', tol=1e-12, options={'maxfev': 1e4, 'maxiter': 1e4})
    return res.x

def l2_obj_complex(params, y, x, y_function, constraint_function=None, **kwargs):
    y_predicted = y_function(params, x, **kwargs)
    constraint = constraint_function(params, x, y) if constraint_function else 0
    l2_real = np.mean((np.real(y - y_predicted)) ** 2)
    l2_imag = np.mean((np.imag(y - y_predicted)) ** 2)
    return l2_real + l2_imag + constraint    

def fit_complex_function_to_data(x0, y, x, y_function, constraint_function=None, **kwargs):
    res = minimize(lambda p: l2_obj_complex(p, y, x, y_function, constraint_function, **kwargs), x0, method='Nelder-Mead', tol=1e-12, options={'maxfev': 1e4, 'maxiter': 1e4})
    return res.x

# fourier processing
def get_fft(signal):
    return np.fft.fftshift(np.fft.fft(signal))

def get_fft_frequency(signal, sampling_frequency):
    return np.linspace(-1, 1, signal.size) * sampling_frequency / 2

# modified fourier processing - similar to laplace
def get_r(x):
    '''
    get the time decay constant needed for a given signal
    :param x: some signal (listlike)
    :return: optimal time decay constant (float)
    '''
    first_nonzero = np.nonzero(x)[0][0]
    return abs(x[-1] / x[first_nonzero]) ** (1 / (x.size - first_nonzero))


def mdft(x, r, length=None):
    '''
    perform the modified discrete fourier transform of a signal at a given radial distance
    :param x: some signal (listlike)
    :param r: radial distance defining the circle on which the modified fourier transform will be calculated (float)
    (r=1.0 gives a discrete fourier transform)
    :param length: number of elements in the transformed signal (int)
    (default is None, which leaves length=len(x); recommended to set equal to the length of shortest signal in batch)
    :return: modified discrete fourier transform of x at r (numpy array)
    '''
    n = np.arange(0, x.size, 1)
    return np.fft.fftshift(np.fft.fft(x * r ** -n, n=length))


# pick contact point - hertz fit
def offset_polynom(params, x, polynomial_order=2):
    y_offset, x_offset, slope = params
    d = x_offset - x
    return slope * (abs(d) - d) ** polynomial_order + y_offset

def offset_polynom_constraint(params, x, y):
    m = 1e9
    y_offset, x_offset, slope = params
    penalty = 0
    # y_offset
    penalty += ((y_offset - min(y)) ** 2 * (y_offset < min(y)) + (y_offset - max(y)) ** 2 * (y_offset > max(y))) * m
    # x_offset
    penalty += ((x_offset - min(x)) ** 2 * (x_offset < min(x)) + (x_offset - max(x)) ** 2 * (x_offset > max(x))) * m
    # slope
    min_slope = (max(y) - min(y)) / (max(x) - min(x))
    penalty += ((slope - min_slope) ** 2 * (slope < min_slope))
    return penalty

def get_contact_point_index_fit(df, trigger_point_index, n_tries=5):
    # pre-process
    # cut out trigger point
    x = df.indentation.values.copy()[:trigger_point_index]
    y = df.force.values.copy()[:trigger_point_index]
    # shift
    x -= x[0]
    y -= y[0]
    # normalize
    x /= max(x)
    y /= max(y)

    # define bounds
    min_slope = (max(y) - min(y)) / (max(x) - min(x))
    bounds = ((min(y), max(y)), (min(x), max(x)))

    params = []
    error_count = 0
    for _ in range(n_tries):
        # guess loop
        x0 = [np.random.uniform(low=bounds[0][0], high=bounds[0][1]),
            np.random.uniform(low=bounds[1][0], high=bounds[1][1]),
            min_slope]
        try:
            params.append(fit_function_to_data(x0, y, x, offset_polynom, offset_polynom_constraint))
        except:
            error_count += 1
    if error_count == n_tries:
        print(f'the code failed to find a contact point after {n_tries} attempts, there might be a problem with the data')
        return None
    y_offset, x_offset, slope = np.mean(params, axis=0)
    # get the index of the contact point
    return np.argmin((x - x_offset) ** 2)