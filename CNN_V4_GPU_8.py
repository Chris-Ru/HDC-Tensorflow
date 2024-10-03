
import pandas as pd
from sklearn.metrics import classification_report
# from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import confusion_matrix

# Create an alias for np.object
np.object = object
np.bool = bool
np.int = int
np.float = float

def evaluate_model_performance(test_labels, predictions,delay, step_size,initial_osc_index):
    to_consider=int(delay/step_size)
    print(predictions[initial_osc_index:initial_osc_index+to_consider+1])
    predicted = np.count_nonzero(predictions[initial_osc_index:initial_osc_index+to_consider+1]==1)
    print("predicted true positives",predicted)
    corrected = np.count_nonzero(test_labels[initial_osc_index:initial_osc_index+to_consider+1]==1)
    print(test_labels[initial_osc_index:initial_osc_index+to_consider+1])
    print("predicted false positives",corrected)
    return predicted/corrected,to_consider
'''def evaluate_model_performance_fp(test_labels, predictions,initial_osc_index):
    to_consider = len(predictions[:initial_osc_index]
    print("total fpr count", to_consider)
    predicted = predictions[:initial_osc_index].count(0)
    print("predcited flase positives",predicted)
    corrected = np.count_nonzero(test_labels[:initial_osc_index]==0)
    print("actual false positives",corrected)
    print(predicted/corrected,"probability of TN")
    print("probability of detecting the false positives", 1-(predicted/corrected))
    return predicted/corrected,to_consider, predicted,corrected, 1-(predicted/corrected)'''
def evaluate_model_performance_fp(test_labels, predictions, initial_osc_index):
    to_consider = len(predictions[:initial_osc_index])
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # False Positive Rate
    return fpr , to_consider
def calculate_probability_from_tp(tpr, N, s):
    if N > 0:
        r = tpr
        N_s = (N - s) + 1
        P = 1 - (1 - r**s)**N_s
        return P
    return 0
def calculate_probability_from_fp(fpr, N, s):
    if N > 0:
        r = fpr
        N_s = (N - s) + 1
        P = 1 - (1 - r**s)**N_s
        return P
    return 0

import numpy as np

def generate_and_separate_ngrams(data, n, overlap_percentage, fault_start, fault_end, delay):
    step_size = round(n * (1 - overlap_percentage))
    ngrams = []
    labels = []
    oscillation_ngrams_count = 0 
    consider_delay = fault_start +( delay/ 1000000)  # Convert microseconds delay to seconds and add to fault_start
    print("Consider delay:", consider_delay)
    first_delay_ngram_index = None  # Initialize to None to find the first ngram that includes the delay
    initial_osc_index = None
    initial_array=[]
    delay_array=[]
    print(fault_start<consider_delay)
    for i in range(0, len(data) - n + 1, step_size):
        time_ngram = data['Time'].iloc[i:i + n].tolist()
        
        if any(fault_start <= time <= fault_end for time in time_ngram):
            label = 1
            oscillation_ngrams_count += 1
            if initial_osc_index is None:
                initial_osc_index=len(ngrams)

        else:
            label = 0
        
        current_ngram = [data[col].iloc[i:i + n].tolist() for col in data.columns if col != 'Time']
        combined_ngram = list(zip(*current_ngram))
        ngrams.append(combined_ngram)
        labels.append(label)
    #print("iitial array", initial_array)
    #print("delay array", delay_array)
    return np.array(ngrams), np.array(labels), initial_osc_index
# Main script
def main():
    # Load data
    import numpy as np
    import pandas as pd

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    import numpy as np
    from sklearn.metrics import confusion_matrix

    training = pd.read_csv("20sec_9-9.15_10mSec.csv", header=None, usecols=[0, 4], names=['Time','Current'])
    training = training[(training['Time'] >= 0.5) & (training['Time'] <= 9.15)].sort_values(by='Time')
    
    testing = pd.read_csv("NewDataSet4.csv", header=None, usecols=[0, 4], names=['Time','Current'])
    testing = testing[(testing['Time'] >= 0.5) & (testing['Time'] <= 9.1)].sort_values(by='Time')

    data=[]
    # Parameters for NGram and detection
    # Number of consecutive NGrams for oscillation detection
    for d in [500]: #500, 800
      for s in [3,5,8]:#3, 5, 8, 10
        for i in [300, 250, 200, 150, 100, 70, 50, 20, 10]: #300, 250, 200, 150, 100, 70, 50, 20, 10 
            for j in [.95, 0.85, 0.75, 0.50]:   #0.95, 0.85, 0.75, 0.50
                # Generate NGrams and labels
                fault_start, fault_end = 9.03, 9.1
                train_fault_start, train_fault_end = 9.0, 9.15
                delay=d
                s = s  
                step_size = round(i * (1 -j))
                ngrams, labels , _= generate_and_separate_ngrams(training, i, j, train_fault_start, train_fault_end, delay)
                test_ngrams, test_label,initial_osc_index= generate_and_separate_ngrams(testing, i, j, fault_start, fault_end, delay)
                train_len=len(ngrams)
                test_len= len(test_ngrams)
                print(ngrams.shape)
                print(test_ngrams.shape)
                #print("osc count",N)
                print("first oscllaiton index",initial_osc_index)

                # Transform NGrams
                # Flatten the NGrams before scaling
                ngrams_flat = np.array(ngrams).reshape(len(ngrams), -1)
                test_ngrams_flat = np.array(test_ngrams).reshape(len(test_ngrams), -1)

                # Scale data
                scaler = StandardScaler()
                scaler.fit(ngrams_flat)  # Fit only on training data
                X_train_scaled = scaler.transform(ngrams_flat)
                X_test_scaled = scaler.transform(test_ngrams_flat)

                # sm = SMOTE()
                # X_train_scaled, labels = sm.fit_resample(X_train_scaled, labels)

                # Check the shape of the scaled data
                print(f"Shape of scaled training data: {X_train_scaled.shape}")
                print(f"Shape of scaled testing data: {X_test_scaled.shape}")
                
                import time
                from tensorflow import keras
                from keras import layers
                from keras import regularizers


                model = keras.Sequential()
                model.add(layers.Conv1D(filters= 64, kernel_size=3, activation='relu', input_shape=(i, 1), padding='same'))
                model.add(layers.Conv1D(filters= 64, kernel_size=3, activation='relu', padding='same'))
                model.add(layers.Conv1D(filters= 64, kernel_size=3, strides = 2 ,activation='relu', padding='same'))
                model.add(layers.Conv1D(filters= 64, kernel_size=3, strides = 2 ,activation='relu', padding='same'))


                model.add(layers.Dropout(0.5))
                model.add(layers.Flatten())
                model.add(layers.Dense(100, activation='relu'))
                model.add(layers.Dropout(0.3))
                model.add(layers.Dense(1, activation='sigmoid'))

                model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])

                model.summary()
                batch_train = 512
                batch_test = 1
                import time

                start=time.time()

                history = model.fit(X_train_scaled , labels, epochs=10, batch_size=batch_train)
                train_time=time.time()-start

                from sklearn.metrics import classification_report
                import time
                # Get the model predictions for the train set
                train_pred = model.predict(X_train_scaled, batch_size=batch_train)
                train_pred = np.round(train_pred).astype(int)

                # Print the classification report
                print("Train")
                target_names = ['normal', 'oscillation']
                print(classification_report(labels, train_pred, target_names = target_names))


                print("test")
                start=time.time()
                test_pred = model.predict(X_test_scaled, batch_size=batch_test)
                test_time=time.time()-start

                test_pred = np.round(test_pred).astype(int)

                # Print the classification report
                target_names = ['normal', 'oscillation']
                print(classification_report(test_label, test_pred, target_names = target_names))
                report = classification_report(test_label, test_pred, target_names = target_names, output_dict=True)
                macro_avg_f1 =report['macro avg']['f1-score']
                tpr, N= evaluate_model_performance(test_label,test_pred, delay,step_size, initial_osc_index)  # From your existing evaluation function
                
                print("tpr",tpr)
                calculated_probability_TP = calculate_probability_from_tp(tpr,N, s)
                print(f"we are {calculated_probability_TP} confidernt that we would detect oscillation in critical wibndow")
                fpr,N_fp= evaluate_model_performance_fp(test_label,test_pred,initial_osc_index)  
                calculated_probability_FP = calculate_probability_from_fp(fpr, N_fp,s)
                print(f"probability of false positives{calculated_probability_FP}")
                
                data.append({
                'Ngram': i,
                "s": s,
                'Overlap': j,
                'delay':d,
                'train length': train_len,
                'test_len': test_len,
                'Train Time': train_time,
                'test time': test_time,
                'F1 score': macro_avg_f1,
                #'support vetors': sv,
                #'model size in disk': f"{model_size:.3f} MB",
                #'model size in bits': f"{megabytes:.3f} MB",
                'Train time per sample': (train_time/train_len)*1000000,
                'test time per sample':  (test_time/test_len)*1000000,
                'N for TP': N,
                'tpr':tpr,
                'TP_confidence_probability':calculated_probability_TP,
                'N for FP':N_fp,
                'fpr':fpr,
                'FP_confidence_probability':calculated_probability_FP})

                import pandas as pd
                results_df = pd.DataFrame(data)

                # Save to Excel if needed
                results_df.to_excel("CNN_V4_gpu001_2x2gpu.xlsx", index=False)






if __name__ == "__main__":

  main()