import tensorflow.keras as keras
from ray.tune import track
from ray.tune import SyncConfig
import numpy as np
np.random.seed(0)

import tensorflow as tf
print(tf.__version__)
try:
    tf.get_logger().setLevel('INFO')
except Exception as exc:
    print(exc)
import warnings
warnings.simplefilter("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import ray
from ray import tune
import time
import inspect
import pandas as pd


'''
class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras.
    
    The callback is invoked every epoch.
    """

    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        track.log(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))
    
'''   
class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras.
    
    The callback is invoked every epoch.
    """
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()
    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        if "acc" in logs:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs["acc"])
        else:
            tune.report(keras_info=logs, val_loss=logs['val_loss'], mean_accuracy=logs.get("accuracy"))
        time.sleep(0.1)

from keras import backend as K
from sklearn.metrics import cohen_kappa_score
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon())) 

def create_model(learning_rate, dense_1, dense_2):
    assert learning_rate > 0 and dense_1 > 0 and dense_2 > 0, "Did you set the right configuration?"
    model = Sequential()
    model.add(Dense(int(dense_1), input_shape=(898,), activation='relu', name='fc1'))
    model.add(Dense(int(dense_2), activation='relu', name='fc2'))
    model.add(Dense(19, activation='softmax', name='output'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    return model
        
def tune_ASNM(config):  
    model = create_model(learning_rate=config["lr"], dense_1=config["dense_1"], dense_2=config["dense_2"])  # TODO: Change me.
    checkpoint_callback = ModelCheckpoint(
        "model.h5", monitor='loss', save_best_only=True, save_freq=2)

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [checkpoint_callback, TuneReporterCallback()]
    task_dataset = pickle.load(open('/app/task_dataset.pkl', "rb"))
    X_train = task_dataset[0]
    Y_train = task_dataset[1]
    X_test = task_dataset[2]
    Y_test = task_dataset[3]
    # Train the model
    hist = model.fit(
        X_train, Y_train, 
        validation_data=(X_test, Y_test),
        verbose=0, 
        batch_size=100, 
        epochs=100, 
        callbacks=callbacks)
    for key in hist.history:
        print(key)

# Random and uniform sampling for hypertune
def random_search(task_data, task_id=0):
    import numpy as np; np.random.seed(5)  
    hyperparameter_space = {
        "lr": tune.loguniform(0.001, 0.1),  
        "dense_1": tune.uniform(50, 150),
        "dense_2": tune.uniform(20, 100),
    }  
    num_samples = 10 
    ####################################################################################################
    ################ This is just a validation function for tutorial purposes only. ####################
    HP_KEYS = ["lr", "dense_1", "dense_2"]
    assert all(key in hyperparameter_space for key in HP_KEYS), (
        "The hyperparameter space is not fully designated. It must include all of {}".format(HP_KEYS))
    ######################################################################################################

    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
            ignore_reinit_error=True
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init(ignore_reinit_error=True)
    analysis = tune.run(
        tune_ASNM, 
        name="Random_ASNM_task"+str(task_id),
        verbose=1, 
        config=hyperparameter_space,
        num_samples=num_samples,
        sync_config=SyncConfig(upload_dir="gs://rkarn-28d6244ed4c54337-outputs")
    time.sleep(1)

    assert len(analysis.trials) > 2, "Did you set the correct number of samples?"

    # Obtain the directory where the best model is saved.
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_accuracy", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_accuracy', mode='max'), 
          'lr:', analysis.get_best_config(metric='keras_info/val_accuracy', mode='max')['lr'], 'dense_1:', analysis.get_best_config(metric='keras_info/val_accuracy', mode='max')['dense_1'], 'dense_2:', analysis.get_best_config(metric='keras_info/val_accuracy', mode='max')['dense_2']        )
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    learning_rate = analysis.get_best_config(metric='keras_info/val_accuracy', mode='max')['lr']
    optimizer = Adam(lr=learning_rate)
    tuned_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/val_accuracy', mode='max'))

#PBT population based sampling 
def mutation_pbtsearch(task_data, task_id=0):
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune.utils import validate_save_restore
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        perturbation_interval=20,
        hyperparam_mutations={
            # distribution for resampling
            "lr": lambda: np.random.uniform(0.0001, 1),
            # allow perturbations within this set of categorical values
            "dense_1": [40, 60, 100], "dense_2": [30, 50, 70], 
        }
    )

    
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis = tune.run(
        tune_ASNM,
        name="PBT_ASNM_task"+str(task_id),
        scheduler=scheduler,
        reuse_actors=True,
        verbose=1,
        stop={
            "training_iteration": 100,
        },
        num_samples=10,

        # PBT starts by training many neural networks in parallel with random hyperparameters. 
        config={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        })
    time.sleep(1)
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_accuracy", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_accuracy', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/val_accuracy', mode='max'))

#ASHA Schedular
def ASHA_search(task_data, task_id=0):
    from ray.tune.schedulers import ASHAScheduler
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    custom_scheduler = ASHAScheduler(
        metric='mean_accuracy',
        mode="max",
        reduction_factor = 2,
        grace_period=1)# TODO: Add a ASHA as custom scheduler here
    hyperparameter_space={
        "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    
    analysis = tune.run(
        tune_ASNM, 
        scheduler=custom_scheduler, 
        config=hyperparameter_space, 
        verbose=1,
        num_samples=10,
        #resources_per_trial={"cpu":4},
        name="ASHA_ASNM_task"+str(task_id)  # This is used to specify the logging directory.
    )
    time.sleep(1)
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/val_acc", mode="max")
    print('Best model:',analysis.get_best_trial(metric='keras_info/val_acc', mode='max'), 
          analysis.get_best_config(metric='keras_info/val_acc', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))


#HyperOpt Search 
def hyperopt_search(task_data, task_id=0):
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.hyperopt import HyperOptSearch

    
    search_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    current_best_params = [{
    'lr': 0.01,
    'dense_1': 100,
    'dense_2': 50,
    }]
    scheduler = AsyncHyperBandScheduler()
    
    algo = HyperOptSearch(points_to_evaluate=current_best_params)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis =tune.run(tune_ASNM,
        name="hyperopt_search"+str(task_id), verbose = 1,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=10, 
        metric="mean_accuracy",
        mode="max",
        config=search_space,
        stop={"training_iteration": 150})
    time.sleep(1)
    #from ray.tune import Analysis as analysis
    #analysis = ray.tune.Analysis('/root/ray_results/BayesOptSearch_ASNM') 
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/accuracy", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))

def BayesOptSearch(task_data, task_id=0):
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.suggest.bayesopt import BayesOptSearch
    
    search_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    scheduler = AsyncHyperBandScheduler()
    
    algo = ConcurrencyLimiter(BayesOptSearch(utility_kwargs={
        "kind": "ucb",
        "kappa": 2.5,
        "xi": 0.0
        }, metric = 'mean_accuracy', mode = 'max'), 
        max_concurrent=4)
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis =tune.run(tune_ASNM,
        name="BayesOptSearch_ASNM_task"+str(task_id), verbose = 1,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=100, 
        metric="mean_accuracy",
        mode="max",
        config=search_space,
        stop={"training_iteration": 150})
    time.sleep(1)
    
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/accuracy", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))


def NeverGradSearch(task_data, task_id=0):
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.nevergrad import NevergradSearch
    import nevergrad as ng
    
    search_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    scheduler = AsyncHyperBandScheduler()
    
    algo = NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne,
        # space=space,  # If you want to set the space manually
    )
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis =tune.run(tune_ASNM,
        name="NeverGradSearch"+str(task_id), verbose = 1,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=10, 
        metric="mean_accuracy",
        mode="max",
        config=search_space,
        stop={"training_iteration": 150})
    time.sleep(1)
    
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/accuracy", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))

def OptunaSearch(task_data, task_id=0):
    from ray.tune.suggest import ConcurrencyLimiter
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.tune.suggest.optuna import OptunaSearch
    
    search_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    scheduler = AsyncHyperBandScheduler()
    
    algo = OptunaSearch(metric="mean_accuracy",
        mode="max")
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis =tune.run(tune_ASNM,
        name="OptunaSearch"+str(task_id), verbose = 1,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=10, 
        metric="mean_accuracy",
        mode="max",
        config=search_space,
        stop={"training_iteration": 150})
    time.sleep(1)
    
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/accuracy", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))

def ZOOptSearch(task_data, task_id=0):
    from ray.tune.suggest.zoopt import ZOOptSearch
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from zoopt import ValueType  # noqa: F401
    
    search_space={
            "lr": tune.uniform(0.001, 1),
            "dense_1": tune.uniform(50, 150), "dense_2": tune.uniform(20, 100),
        }
    scheduler = AsyncHyperBandScheduler()
    num_samples = 10
    zoopt_search_config = {
        "parallel_num": 4,
    }

    algo = ZOOptSearch(
        algo="Asracos",  # only support ASRacos currently
        budget=num_samples,
        # dim_dict=space,  # If you want to set the space yourself
        **zoopt_search_config)

    if "REDIS_PASSWORD" in os.environ:
        ray.init(
            address=os.environ.get("RAY_SERVER", "auto"),
            _redis_password=os.environ.get("REDIS_PASSWORD", ""),
        )
    else:
        # according to the docs local_mode, if true, forces serial execution which is meant for debugging
        # unfortunately, it also allows requests for resources such as GPUs to subsequently ignore them without
        # any error or warning
        ray.init()
    analysis =tune.run(tune_ASNM,
        name="ZOOptSearch"+str(task_id), verbose = 1,
        scheduler=scheduler,
        search_alg=algo,
        num_samples=num_samples, 
        metric="mean_accuracy",
        mode="max",
        config=search_space,
        stop={"training_iteration": 150})
    time.sleep(1)
    #from ray.tune import Analysis as analysis
    #analysis = ray.tune.Analysis('/root/ray_results/BayesOptSearch_ASNM') 
    print("You can use any of the following columns to get the best model: \n{}.".format(
        [k for k in analysis.dataframe() if k.startswith("keras_info")]))
    print("=" * 10)
    logdir = analysis.get_best_logdir("keras_info/accuracy", mode="max")
    print('Best model:', analysis.get_best_config(metric='keras_info/accuracy', mode='max'))
    # We saved the model as `model.h5` in the logdir of the trial.
    from tensorflow.keras.models import load_model
    tuned_model = load_model(logdir + "/model.h5", custom_objects =  {'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    tuned_model.summary()
    X_test = task_data[2]
    Y_test = task_data[3]
    tuned_loss, tuned_accuracy, f1_score, precision, recall = tuned_model.evaluate(X_test, Y_test, verbose=0)
    print("Loss is {:0.4f}".format(tuned_loss))
    print("Tuned accuracy is {:0.4f}".format(tuned_accuracy))
    print ('F1-score = {0}'.format(f1_score))
    print ('Precision = {0}'.format(precision))
    print ('Recall = {0}'.format(recall))
    return(analysis.get_best_config(metric='keras_info/accuracy', mode='max'))

import pickle
import pdb
import time
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        last_model_stats = Model_Perf_save
        for i,lay in enumerate(model.layers):
            last_model_size = last_model_stats['shape'][-1][2*i+1][0]
            layer_weights = lay.get_weights()
            layer_weights[0][:last_model_stats['weights'][-1][2*i].shape[0],:last_model_size] = last_model_stats['weights'][-1][2*i]
            layer_weights[1][:last_model_size] = last_model_stats['weights'][-1][2*i+1]
            model.layers[i].set_weights(layer_weights)      

    def on_batch_end(self, batch, logs={}):
        last_model_stats = Model_Perf_save
        for i,lay in enumerate(model.layers):
            last_model_size = last_model_stats['shape'][-1][2*i+1][0]
            layer_weights = lay.get_weights()
            #pdb.set_trace()
            layer_weights[0][:last_model_stats['weights'][-1][2*i].shape[0],:last_model_size] = last_model_stats['weights'][-1][2*i]
            layer_weights[1][:last_model_size] = last_model_stats['weights'][-1][2*i+1]
            model.layers[i].set_weights(layer_weights)
        
def create_task(data_path):
    data = pickle.load(open(data_path, "rb"))
    return data

def measure_CPU_Mem():
    import psutil
    CPU_usage_dump = []
    Mem_usage_dump = []  
    while True:
        CPU_usage_dump.append(psutil.cpu_percent())
        Mem_usage_dump.append(psutil.virtual_memory().percent)
        f_cpu=open("CPU_used.txt", "wb")
        f_mem=open("Mem_used.txt", "wb")
        pickle.dump(CPU_usage_dump, f_cpu) 
        pickle.dump(Mem_usage_dump, f_mem)     
        f_cpu.close()
        f_mem.close()
        time.sleep(1)

import multiprocessing
task_list = create_task('reduced_asnm_nbpo_tasks.pkl')
num_tasks=5
Model_Perf_save = {}
Model_Perf_save['tr_acc'] = []
Model_Perf_save['val_acc'] = []
Model_Perf_save['precision'] = []
Model_Perf_save['recall'] = []
Model_Perf_save['f1_score'] = []
Model_Perf_save['shape'] = []
Model_Perf_save['weights'] = []
Model_Perf_save['learn_rate'] = []
for search_algo in [random_search,
                    #mutation_pbtsearch,
                    #ASHA_search,
                    #BayesOptSearch,
                    #NeverGradSearch,
                    #OptunaSearch,
                    #ZOOptSearch,
                    #hyperopt_search
                    ]:
    cpu_mem_collection = multiprocessing.Process(target=measure_CPU_Mem)
    cpu_mem_collection.start()
    start_time = time.time()
    for task_id in range(0,num_tasks):
        f = open('/app/task_dataset.pkl', 'wb')
        pickle.dump(task_list[task_id], f)
        f.close()
        hyper_param = search_algo(task_list[task_id], task_id)
        if task_id == 0:
            model = create_model(learning_rate=hyper_param["lr"], dense_1=hyper_param["dense_1"], dense_2=hyper_param["dense_2"])
            #call one of the search algorithm
            history = model.fit(task_list[task_id][0], task_list[task_id][1],
                  batch_size=128, epochs=10, verbose=0,
                  validation_data=(task_list[task_id][2], task_list[task_id][3]))
        else:
            dense_1 = hyper_param["dense_1"] + Model_Perf_save['shape'][-1][1][0]
            dense_2 = hyper_param["dense_2"] + Model_Perf_save['shape'][-1][3][0]
            model = create_model(learning_rate=hyper_param["lr"], dense_1=dense_1, dense_2=dense_2)
            history = model.fit(task_list[task_id][0], task_list[task_id][1],
                  batch_size=100, epochs=100, verbose=0,
                  validation_data=(task_list[task_id][2], task_list[task_id][3]), callbacks  = [LossHistory()])
        loss_and_metrics = model.evaluate(task_list[task_id][0], task_list[task_id][1], verbose=2)
        Model_Perf_save['tr_acc'].append(loss_and_metrics[1])
        loss_and_metrics = model.evaluate(task_list[task_id][4], task_list[task_id][5], verbose=2)
        Model_Perf_save['val_acc'].append(loss_and_metrics[1])
        Model_Perf_save['f1_score'].append(loss_and_metrics[2])
        Model_Perf_save['precision'].append(loss_and_metrics[3])
        Model_Perf_save['recall'].append(loss_and_metrics[4])
        Model_Perf_save['shape'].append([i.shape for i in model.get_weights()])
        Model_Perf_save['learn_rate'].append(hyper_param["lr"])
        Model_Perf_save['weights'].append(model.get_weights()) 
    end_time = time.time()
    print('Search algorithm {} took {}.'.format(search_algo.__name__, end_time - start_time))
    
    f=open("time_taken.txt", "a+")
    f.write('Time taken for algo {} is {}. \n'.format(search_algo.__name__, end_time-start_time))


    f_cpu=open("CPU_used.txt", "rb")
    f_mem=open("Mem_used.txt", "rb")
    cpu_usage = pickle.load(f_cpu)
    mem_usage = pickle.load(f_mem)
    f_cpu.close()
    f_mem.close()
    cpu_mem_collection.terminate()
    cpu_mem_collection.join()
    f.write('CPU used is {}. \n'.format(np.mean(cpu_usage)))
    f.write('Memory used is {}. \n \n'.format(np.mean(mem_usage)))
    f.close()

for j,search_algo in enumerate([
                    random_search,
                    #mutation_pbtsearch,
                    #ASHA_search,
                    #BayesOptSearch,
                    #NeverGradSearch,
                    #OptunaSearch,
                    #ZOOptSearch,
                    #hyperopt_search
                    ]):
    print('Search Algorithm: {0}'.format(search_algo.__name__))
    print('Training accuracy for tasks:',Model_Perf_save['tr_acc'][j*num_tasks:j*num_tasks+num_tasks])
    print('Validation accuracy for tasks:',Model_Perf_save['val_acc'][j*num_tasks:j*num_tasks+num_tasks])
    print('Precision:',Model_Perf_save['precision'][j*num_tasks:j*num_tasks+num_tasks])
    print('Recall:',Model_Perf_save['recall'][j*num_tasks:j*num_tasks+num_tasks])
    print('F1 score:',Model_Perf_save['f1_score'][j*num_tasks:j*num_tasks+num_tasks])
    print('Nodes in hidden layers (1,2): ',[(i[0][1],i[2][1])for i in Model_Perf_save['shape']][j*num_tasks:j*num_tasks+num_tasks])
    print('Learning rates: ', Model_Perf_save['learn_rate'][j*num_tasks:j*num_tasks+num_tasks])
    print('------------------------')

