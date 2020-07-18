"""
Required Packages: the ones mentioned in the PythonSetupNotes
"""
import librosa
import pandas as pd
import numpy as np
import scipy as sp
import warnings
warnings.filterwarnings('ignore')

from audiot.audio_features import AudioFeatures, calc_log_mel_energy_features
from audiot.audio_signal import AudioSignal
from audiot.audio_labels import load_labels, clean_overlapping_labels


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def train_cough_detector(audio_files_list, label_files_list):
    """
    Dummy example of how your train_cough_detector function should operate.  This function just
    "trains" a threshold that will classify any feature vector whose sum is above that threshold as
    a cough (which is not a very good way to do it, but is simple for the purposes of
    this example).
    Returns:
        Object:  An object representing the trained cough detector model.  This can be any type of
            object you want, but should match what your run_cough_detector function expects to be
            passed in as the first parameter (named cough_detector_model in this file).  For this
            dummy example, it returns a tuple containing the threshold and the percentile value
            used to set that threshold based on the coughs seen in the training data: (threshold,
            percentile_to_use_as_threshold).
    """
    # tested rbf,linear,poly and sigmoid kernel on svm
    tuning_parameters = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    scores = ['precision', 'recall']
    print("Loading features...")
    num_files = int(len(data))
    X = np.zeros((2401*num_files,13))# test num_files test files
    y = np.zeros((2401*num_files,1))
    i = 0
    for d in data.values():
        x_path, y_path = d["features"], d["labels"]
        signal = AudioSignal.from_file(x_path)
        label = load_labels(y_path)
        features = calc_log_mel_energy_features(signal)
        features.event_names = ["cough"]
        features.match_labels(label)
        X[2401*i:2401*(i+1),:] = features.features
        y[2401*i:2401*(i+1)] = features.true_events
        i = i+1
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=555)
    cough_detector = {}
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        clf = GridSearchCV(
            SVC(), tuning_parameters, scoring='%s_macro' % score)

        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print(clf.best_params_)
        y_true, y_pred = y_test, clf.predict(x_test) ## this line
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true,y_pred)
        F_05 = ((1+0.5**2) * precision * recall) / (0.5**2 * precision + recall)
        print("precision",precision, "  recall:",recall, " F 0.5:",F_05)
        cough_detector[F_05] = clf
        print("Detailed classification report:")
        print(classification_report(y_true, y_pred))
        print()
    # Return the classifier which has the highest F0.5 score
    return cough_detector[max(cough_detector.keys())]

def run_cough_detector(cough_detector, audio_file):
    """
    Args:
        in my couph detector I have a list of dictionaries contains SVM parameters, C, gamma, and kernel

        audio_file (Path): A pathlib.Path object containing the path to the audio recording that
            the cough detector should be run on.  You'll need to load this file, calculate whatever
            features you are using from it (along with the time windows that they span), classify
            them as coughs or not, then build and return a labels_dataframe that matches format
            you get when you read in an Audacity label file using audio_labels.load_labels().

    Returns:
        DataFrame: A pandas DataFrame with an "onset" column specifying the beginning time of each
            cough detected in the file (in seconds), an "offset" column specifying the ending time
            of each cough detected in the file (also in seconds), and an "event_label" column
            containing the label for each detected event ("cough" in this case).
    """
    X = np.zeros((2401,13))
    signal = AudioSignal.from_file(audio_file)
    features = calc_log_mel_energy_features(signal)
    features_summed = np.sum(features.features, axis=1)
    features.event_names = ["cough"]
    X[0:2401,:] = features.features
    y_pred = y_pred = [bool(c) for c in cough_detector.predict(X)]

    labels_dataframe = pd.DataFrame(
        {
            "onset": pd.Series(features.frame_start_times[y_pred]),
            "offset": pd.Series(features.frame_end_times[y_pred]),
            "event_label": pd.Series("cough", index=range(np.sum(y_pred))),
        }
    )
    labels_dataframe = clean_overlapping_labels(labels_dataframe)
    return labels_dataframe
