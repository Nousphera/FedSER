##############################################################################
import os
import sys
# Import Project Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # Max message size on comms 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
##############################################################################

import gc
import time
import tensorflow as tf
# Flower
from flwr.common import (EvaluateIns, Parameters)
from flwr.server import Server, start_server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import DefaultStrategy, FedAvg
# Custom Modules
from model.model import SelfTrainingNetwork
from utils.history import History
from utils.plotter import ProgressPlot

class AudioServer:
    
    def __init__(self,
                flwr_evalution_step, flwr_sample_fraction, flwr_min_sample_size, flwr_min_num_clients, flwr_rounds,
                model_num_classes, model_num_mel_bins, model_lr, model_batch_size, model_dropout_rate, model_epochs, model_verbose, model_ds_test,
                enabler_save_model, enabler_save_figures, enabler_save_logs,
                name_model_file, name_figure_file, name_dataset):

        # Flower Parameters
        self.evalution_step = flwr_evalution_step
        self.sample_fraction = flwr_sample_fraction
        self.min_sample_size = flwr_min_sample_size
        self.min_num_clients = flwr_min_num_clients
        self.rounds = flwr_rounds

        # Model Parameters
        self.num_classes = model_num_classes
        self.num_mel_bins = model_num_mel_bins
        self.lr = model_lr
        self.batch_size = model_batch_size
        self.dropout_rate = model_dropout_rate
        self.epochs = model_epochs
        self.verbose = model_verbose
        self.ds_test = model_ds_test
        self.input_shape = self.ds_test.element_spec[0].shape[2:]

        # Logger Enablers
        self.save_model = enabler_save_model
        self.save_figures = enabler_save_figures
        self.save_logs = enabler_save_logs
        self.monitor_clients = False

        # Local Variables Counters and Variables
        self.name_model_file = name_model_file
        self.train_metrics_file = '/'.join(self.name_model_file.split('/')[:6])+'/train_metrics/'+((self.name_model_file.split('.')[0]).split('/')[-1])+'.txt'
        self.current_round = 0
        self.strategy = None
        self.final_accuracy = 0.0
        self.round_time = time.time()

        # Objects
        self.plotter = ProgressPlot(figure_file=name_figure_file, num_clients=flwr_min_num_clients, dataset_name=name_dataset, train_step=model_epochs)
        self.client_manager = SimpleClientManager()
        self.history = History()

        # Clear data from previous sessions.
        tf.keras.backend.clear_session()

    def set_strategy(self, name):

        if name == "FedAvg":
            self.strategy = FedAvg(fraction_fit=self.sample_fraction, min_fit_clients=self.min_sample_size, min_available_clients=self.min_num_clients,
                    on_fit_config_fn=self.get_on_fit_config_fn(), fraction_eval=0, min_eval_clients=0,
                    on_evaluate_config_fn=None, eval_fn=self.get_eval_fn(ds_test=self.ds_test), accept_failures=True)
        else:
            self.strategy = DefaultStrategy(fraction_fit=self.sample_fraction, min_fit_clients=self.min_sample_size, min_available_clients=self.min_num_clients,
                    on_fit_config_fn=self.get_on_fit_config_fn(), fraction_eval=0, min_eval_clients=0,
                    on_evaluate_config_fn=None, eval_fn=self.get_eval_fn(ds_test=self.ds_test), accept_failures=True)

    def server_start(self, server_address):

        # Run server
        if self.strategy==None:
            self.set_strategy(None)

        start_server(server_address=server_address, server = Server(client_manager=self.client_manager, strategy=self.strategy),
                    config={"num_rounds": self.rounds}, strategy=self.strategy, grpc_max_message_length= GRPC_MAX_MESSAGE_LENGTH)

    def get_on_fit_config_fn(self):

        # Return a function which returns training configurations for clients.
        #def fit_config(rounds=self.rounds, epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.lr, dropout_rate=self.dropout_rate):
        def fit_config(rnd=self.current_round):

            if self.current_round!=1:
                print('\nTraining round completed in '+str(round(time.time()-self.round_time,2))+ ' seconds.')

            print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'+
                    '               Server started '+str(self.current_round)+'th round of training.\n'+
                    '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')

            # Update round start time
            self.round_time = time.time()
            return {"total_rounds": str(self.rounds), "current_round": str(self.current_round), "local_train_epochs": str(self.epochs),
                    "batch_size": str(self.batch_size), "learning_rate": str(self.lr), "dropout_rate" : str(self.dropout_rate)}

        return fit_config

    def get_eval_fn(self, ds_test):

        # Return an evaluation function for centralized evaluation.
        def evaluate(weights):

            # Initialize Variables
            loss, acc = 0,0

            # Update Current Round
            self.current_round += 1

            if self.monitor_clients and self.current_round!=1:
                self.monitor_clients_progress()
                
                if (self.current_round-1)==self.rounds and self.save_logs:
                    with open(self.train_metrics_file,'w') as f:
                        f.write( 'Average Train Loss\n'+str(self.history.get_loss_distributed()) + '\nAverage Train Accuracy\n'+str(self.history.get_accuracy_distributed()))

            # Perform Evaluation every eval_step rounds
            if ((self.current_round-1) % self.evalution_step == 0):

                # Load Model and Wrap into Time Distributed Layer
                model = SelfTrainingNetwork.get_evaluation_network(input_shape=self.input_shape, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
                model.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

                # Set model Weights
                model.set_weights(weights)

                # Perform Evaluation
                loss, acc = model.evaluate(ds_test, verbose=self.verbose)

                # Add Results to History
                self.history.add_loss_centralized(rnd=self.current_round-1, loss=loss)
                self.history.add_accuracy_centralized(rnd=self.current_round-1, acc=acc)
                self.final_accuracy = acc
                
                # Save Figure 
                if self.save_figures:
                    self.plotter.evaluate_plot(loss=self.history.get_loss_centralized(), accuracy=self.history.get_accuracy_centralized())

                # Save Model
                if self.save_model:
                    self.store_model(weights)    

                # Clear Memory
                gc.collect()
                tf.keras.backend.clear_session()

            return float(loss), {"accuracy": float(acc)}

        return evaluate

    def monitor_clients_progress(self):

        # Get Mean Results from Clients
        loss, accuracy = self.get_mean_loss_accuracy()

        # Add Results to History
        self.history.add_loss_distributed(rnd=self.current_round, loss=loss)
        self.history.add_accuracy_distributed(rnd=self.current_round, acc=accuracy)

        # Plot Results
        if self.save_figures:
            self.plotter.train_plot(loss=self.history.get_loss_distributed(), accuracy=self.history.get_accuracy_distributed())

    def get_mean_loss_accuracy(self):

        # Initialize Variables
        loss, accuracy = 0,0

        # Create list of Clients
        clients = self.client_manager.sample(self.client_manager.num_available())
        # Iterate Through Clients to get loss, accuracy
        for idx,client in enumerate(clients):
            results = client.evaluate(EvaluateIns(Parameters(None,tensor_type=""),{"return_history": str(True)}))
            #print('Client '+str(idx)+' results : ('+str(results.loss)+' , '+str(results.accuracy)+')')
            loss += results.loss
            accuracy += results.accuracy

        # Calculate mean - Avoid division with 0
        loss = loss/len(clients) if len(clients)>0 else 0
        accuracy = accuracy/len(clients) if len(clients)>0 else 0

        return loss, accuracy

    def store_model(self,weights):

        # Load Model
        model = SelfTrainingNetwork.get_network(input_shape=self.input_shape, num_classes=self.num_classes, dropout_rate=self.dropout_rate)
        model.set_weights(weights)
        # Save model
        model.save(self.name_model_file)

        return 0

    def get_accuracy(self):
        return self.final_accuracy