##############################################################################
import os
import sys
# Import Project Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##############################################################################
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import itertools
from model.model_utils import ElementwiseMultiply, History

class SelfTrainingNetwork(tf.keras.Model):

    def __init__(self, input_shape=(None,64,1), num_classes=4, dropout_rate=0.1, l2_rate=0.0001, confidence_limits=[0.5,0.9], beta=0.5):

        super(SelfTrainingNetwork, self).__init__()
        self.model = SelfTrainingNetwork.get_network(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate, l2_rate=l2_rate)        
        #self.input_shape = input_shape
        self.current_confidence = 0.0
        self.confidence_limits = confidence_limits
        self.beta = beta
        self.history = History()

    # Original code taken from Supervised Model
    @staticmethod
    def _conv_block(input_shape, num_features, l2_rate=0.0001, dropout_rate=0.1 , add_max_pool = True):

        inputs = tf.keras.layers.Input(shape=input_shape)
        x_t = tf.keras.layers.Conv2D(num_features, (1, 4), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(inputs)
        x_t = tfa.layers.GroupNormalization(groups=4)(x_t)
        x_t = tf.keras.layers.Activation("relu")(x_t)
        
        x_f = tf.keras.layers.Conv2D(num_features, (4, 1), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(inputs)
        x_f = tfa.layers.GroupNormalization(groups=4)(x_f)
        x_f = tf.keras.layers.Activation("relu")(x_f)
        
        x = tf.keras.layers.Concatenate(axis=-1)([x_t, x_f])
        
        x = tf.keras.layers.Conv2D(num_features, (1, 1), padding = "same", kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(x)
        x = tfa.layers.GroupNormalization(groups=4)(x)
        x = tf.keras.layers.Activation("relu")(x)

        if add_max_pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.SpatialDropout2D(dropout_rate)(x)

        return tf.keras.Model(inputs, x)

    @staticmethod
    def _stc_block(input_shape=(None,64,1), r=2):

        inputs = tf.keras.layers.Input(shape=input_shape, name="STC_input") # (B,C,T,F)
        # Convert to channel first
        x = tf.keras.layers.Permute(dims=[3,2,1])(inputs) # (B,C,F,T)
        # Spectral-Temporal Attention
        max_x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='same', data_format='channels_first')(x)
        avg_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1,1), padding='same', data_format='channels_first')(x)
        x_st = tf.keras.layers.Concatenate(axis=1)([max_x,avg_x])
        x_s = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, input_shape[0]), padding = "valid", data_format='channels_first')(x_st)
        x_t = tf.keras.layers.Conv2D(filters=1, kernel_size=(input_shape[1], 1), padding = "valid", data_format='channels_first')(x_st)
        x_st = tf.keras.layers.Multiply()([x_s,x_t])
        # Channel Attention
        x_c = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(x)
        x_c = tf.keras.layers.Dense(units=int(input_shape[2]/r))(x_c)
        x_c = tf.keras.layers.Dense(units=input_shape[2], activation='sigmoid')(x_c)        
        x_c = tf.keras.layers.Reshape(target_shape=(-1,1,1))(x_c)
        # Combined  STC Attention
        x_stc = tf.keras.layers.Multiply()([x_c,x_st])
        x_stc = tf.keras.layers.Softmax()(x_stc)        
        # Convert back to channel last
        outputs = tf.keras.layers.Permute(dims=[3,2,1])(x_stc) # (B,T,F,C)

        return tf.keras.Model(inputs, outputs, name="STC_attention")

    @staticmethod
    def get_network(input_shape=(None,64,1), num_classes=4, dropout_rate=0.1, l2_rate=0.0001):

        # Input Layer
        inputs = tf.keras.layers.Input(shape=input_shape, name="audio_input")

        # Convolution Layers (4 Blocks)
        x = SelfTrainingNetwork._conv_block(input_shape=input_shape, num_features=24,  l2_rate=l2_rate, dropout_rate=dropout_rate, add_max_pool=True)(inputs)
        x = SelfTrainingNetwork._conv_block(input_shape=x.shape[1:], num_features=32,  l2_rate=l2_rate, dropout_rate=dropout_rate, add_max_pool=True)(x)
        x = SelfTrainingNetwork._conv_block(input_shape=x.shape[1:], num_features=64,  l2_rate=l2_rate, dropout_rate=dropout_rate, add_max_pool=True)(x)
        x = SelfTrainingNetwork._conv_block(input_shape=x.shape[1:], num_features=128, l2_rate=l2_rate, dropout_rate=dropout_rate, add_max_pool=True)(x)
        # STC-Attention Layer
        scores = SelfTrainingNetwork._stc_block(input_shape=x.shape[1:], r=2)(x)
        weigthed_x = ElementwiseMultiply()([x,scores])
        x = tf.keras.layers.Add()([x,weigthed_x])
        # FC Layer
        x = tf.keras.layers.GlobalMaxPool2D()(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)

        return tf.keras.Model(inputs, outputs, name="audio_classifier")

    @staticmethod
    def get_evaluation_network(input_shape=(None,64,1), num_classes=4 , dropout_rate=0.1, l2_rate=0.0001):

        model = SelfTrainingNetwork.get_network(input_shape=input_shape, num_classes=num_classes, dropout_rate=dropout_rate, l2_rate=l2_rate)
        input = tf.keras.layers.Input(shape=(None,)+input_shape)
        output = tf.reduce_mean(tf.keras.layers.TimeDistributed(model)(input), axis=1)
        return tf.keras.Model(input, output)

    @staticmethod
    def get_init_weights(input_shape=(None,64,1), num_classes=4, dropout_rate=0.1, l2_rate=0.0001):

        return SelfTrainingNetwork.get_network(input_shape=input_shape, num_classes=num_classes,
                                    dropout_rate=dropout_rate, l2_rate=l2_rate).get_weights()

    # Additional Code to utilize self-training on unlabeled data.
    @staticmethod
    def cosine_confidence_scheduler(cur_step, total_steps, confidence_limits=[0.5,0.9]):
      return (confidence_limits[1] - (confidence_limits[1] - confidence_limits[0]) * 0.5 * (1 + np.cos((cur_step-1) / (total_steps) * np.pi)))
    
        #     t = t_max - (t_max-t_min)*0.5*(1+math.cos((C/R)*g*math.pi))
        # t = (t_max-t_min)*0.5*(1+math.cos((C/R)*g*math.pi))
    def compile(self, optimizer, loss=tf.nn.sparse_softmax_cross_entropy_with_logits, metrics=['sparse_categorical_accuracy']):
        super(SelfTrainingNetwork, self).compile(optimizer=optimizer, 
                                                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                                metrics=metrics)
        self.optimizer_fn = optimizer
        self.loss_fn = loss

    def fit(self, train_data, epochs, verbose=0, fit_details = {'current_round': 1, 'total_rounds': 100, 'num_batches': None, 'temperature_scaling': 1}):
        
        # Initialize
        l_dataset,u_dataset = train_data
        self.history.on_train_begin()

        for epoch in range(epochs):

            # Set Epoch Variables
            start_it, train_loss = time.time(), 0.0
            self.current_confidence = SelfTrainingNetwork.cosine_confidence_scheduler(cur_step=fit_details['current_round'],
                                                                                    total_steps=fit_details['total_rounds'],
                                                                                    confidence_limits=self.confidence_limits)
            #print('Round '+str(fit_details['current_round'])+' out of '+str(fit_details['total_rounds'])+': Confidence Threshold is set to '+str(self.current_confidence))

            progbar = tf.keras.utils.Progbar(target=fit_details['num_batches']-1, width=30, 
                                            verbose=verbose if verbose != 2 else 0,
                                            interval=0.05, stateful_metrics=None,
                                            unit_name='step')

            # Print Epoch details in verbose 1
            if verbose==1:
                tf.print('Epoch %d/%d' %(epoch+1,epochs))

            # Run train step on each batch
            for step,((l_batch_x,l_batch_y),(u_batch_x,u_batch_y)) in enumerate(zip(itertools.cycle(l_dataset),u_dataset)):
                # Merge batches
                batch_x, batch_y = tf.concat([l_batch_x, u_batch_x], 0), tf.concat([l_batch_y, u_batch_y], 0)
                # Train on 'merged batch'
                loss = self.train_step(x=batch_x, y=batch_y, temperature_scaling=fit_details['temperature_scaling'])
                # Update Progressbar
                progbar.update(current=step, values=[('loss', float(loss['loss']))])
                # Accumulate loss
                train_loss += float(loss['loss'])

            # Store result in history
            self.history.on_epoch_end(epoch=epoch, logs={'loss': train_loss/fit_details['num_batches'], 'sparse_categorical_accuracy': float(0.0)})

            # Print Epoch results in verbose 2
            if verbose==2:    
                tf.print('Epoch %d/%d\n%d/%d - %.1fs - loss: %.4f - sparse_categorical_accuracy: %.4f' %(epoch+1,epochs,step,step,time.time()-start_it,train_loss/fit_details['num_batches'],float(0.0)))

        return self.history

    @tf.function#(experimental_relax_shapes=True)
    def train_step(self, x, y, temperature_scaling=1):
        
        with tf.GradientTape() as tape:
            #####################################################################################################################
            # Get logits and labels
            logits,labels = self.model(x, training=True), y
            # Create an array with bool(T/F) for all point that do not have -1 in labels!
            labeled_mask, unlabeled_mask = tf.not_equal(-1, labels), tf.cast(tf.equal(-1, labels), dtype=tf.float32)
            # Get label/unlabeled split
            num_labeled, num_unlabeled = tf.math.reduce_sum(tf.cast(labeled_mask,dtype=tf.float32)), tf.math.reduce_sum(unlabeled_mask)
            # Get labeled loss
            masked_logits, masked_labels = tf.boolean_mask(logits, labeled_mask), tf.boolean_mask(labels, labeled_mask)
            loss_labeled = tf.reduce_sum(self.loss_fn(labels=masked_labels, logits=masked_logits)) / num_labeled            
            #####################################################################################################################
            # Calculate Prediction Confidence
            pseudo_labels = tf.stop_gradient(input=tf.nn.softmax(logits/temperature_scaling))
            # Calculate labels for unlabeled data 
            pseudo_mask = tf.cast(tf.reduce_max(pseudo_labels, axis=1) >= self.current_confidence, dtype=tf.float32)
            # Get unlabeled loss
            loss_unlabeled = tf.reduce_sum((tf.cast(self.loss_fn(labels=tf.argmax(pseudo_labels, axis=1), logits=logits), dtype=tf.float32) * pseudo_mask) * unlabeled_mask) / num_unlabeled
            #####################################################################################################################
            # Compute total loss
            total_loss = tf.add_n( [((1-self.beta) * loss_labeled) + (self.beta * loss_unlabeled)] + self.model.losses)
            #####################################################################################################################

        # Update model parameters
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer_fn.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": total_loss, "labelled_loss":loss_labeled, "unlabelled_loss":loss_unlabeled}

