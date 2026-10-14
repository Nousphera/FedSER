##############################################################################
import os
import sys
# Import Project Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
##############################################################################

import gc
import tensorflow as tf
from typing import cast
from model.model import SelfTrainingNetwork
from data.data_builder import DataBuilder
from flwr.client import NumPyClient as Client
from flwr.client import start_numpy_client as start_client

# Hide Keras retracing warnings
tf.get_logger().setLevel('ERROR')

class AudioClient(Client):

    def __init__(self, server_address, num_clients, client_id,
                dataset_name ,dataset_dir, utilize_percentages=[0.1,1.0],
                dist_type='Random', dist_mean=0.5, dist_var=0.25,
                batch_size=64, num_mels=64, verbose=2, split_file='', seed=2021):

        # Client Parameters
        self.server_address = server_address
        self.client_id = client_id
        self.batch_size = batch_size
        self.num_mels = num_mels
        self.verbose = verbose

        # Self-training Parameters
        self.beta = 0.5
        self.confidence_limits = [0.5,0.9]
        self.temperature_scaling= 1.0

        # Load Clients Data
        self.train_ds, self.dev_ds, self.num_batches, self.num_classes = DataBuilder.train_dev_ds(
            dataset_name=dataset_name, dataset_dir=dataset_dir, 
            num_clients=num_clients, client=self.client_id,
            dist_type=dist_type, dist_mean=dist_mean, dist_var=dist_var,
            utilize_percentages=utilize_percentages, augmentation=None, validation=False, 
            n_mels=num_mels, batch_size=batch_size, seed=seed, split=split_file)
        self.num_examples_train = self.num_batches*batch_size if self.train_ds else 0
        self.input_shape= self.train_ds[0].element_spec[0].shape[1:] #(None,num_mels,1)

        # Local Variables Initialize
        self.cur_train_round, self.cur_evaluate_round = 0,0
        self.weights = SelfTrainingNetwork(input_shape=self.input_shape, num_classes=self.num_classes, confidence_limits=self.confidence_limits, beta=self.beta).get_init_weights(input_shape=self.input_shape, num_classes=self.num_classes) 
        self.history = {'loss': [], 'accuracy': []}

        tf.keras.backend.clear_session()

    def __call__(self,introduce=False):
        try:
            if introduce:
                print(self)
            start_client(self.server_address, self)
            print("Client Shutdown.")
        except RuntimeError as e:
            print(e)
        return 0

    def __str__(self):
        return "This is client "+str(self.client_id)+" with train dataset of "+str(self.num_examples_train)+" elements." 

    def get_parameters(self):  
        """Return the current local model parameters."""
        return self.weights

    def fit(self, parameters, config):

        # Update training round
        self.cur_train_round+=1
        self.weights = parameters
        ################################################################################################################
        # Run Training Proccess
        # 1.Create model
        model = SelfTrainingNetwork(input_shape=self.input_shape, 
                                    num_classes=self.num_classes, 
                                    dropout_rate=float(config["dropout_rate"]), 
                                    confidence_limits=self.confidence_limits, 
                                    beta=self.beta)
        # 2.Compile model
        model.compile(optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])), 
                    loss=tf.nn.sparse_softmax_cross_entropy_with_logits, 
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        # 3.Set weights
        model.set_weights(parameters)
        # 4.Train
        history = model.fit(train_data=self.train_ds, epochs=int(config["local_train_epochs"]), 
                            verbose=self.verbose, fit_details= {'current_round': self.cur_train_round, 
                                                                'total_rounds': int(config["total_rounds"]), 
                                                                'num_batches': self.num_batches, 
                                                                'temperature_scaling': self.temperature_scaling})
        ################################################################################################################
        # Store Progress
        self.history['loss'] += history.history["loss"]
        self.history['accuracy'] += history.history["sparse_categorical_accuracy"]
        # Print Results
        print('Client '+str(self.client_id)+' finished '+str(self.cur_train_round)+'th round of training with loss/acc : '+str(round(history.history["loss"][0],4))+'/'+str(round(history.history["sparse_categorical_accuracy"][0],4)))
        # Length of train data
        train_set_length = int(config["batch_size"])*self.num_examples_train
        # Clear Memory
        tf.keras.backend.clear_session()
        gc.collect()
        # Return : weights, train examples used, communicated parameters back to server
        return model.get_weights(), train_set_length, {"cur_train_round": self.cur_train_round}

    # Note: Evaluation is used to update server on clients loss/accuracy results.
    # TODO: Move this in the return of train function.
    def evaluate(self, parameters, config):

        # Check if results are available
        if len(self.history['accuracy']) and len(self.history['loss']):
            # Get current loss/acc. during training
            acc,loss = float(self.history['accuracy'][-1]), float(self.history['loss'][-1])
        else:
            acc,loss = 0,0
        # Return: loss, num_test_samples, dictionary
        return loss, 1, {"accuracy": acc}