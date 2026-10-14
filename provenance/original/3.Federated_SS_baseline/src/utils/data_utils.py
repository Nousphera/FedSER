##############################################################################
import os
import sys
# Import Project Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##############################################################################

import random
import itertools
import tensorflow as tf
import numpy as np
import collections
import pandas as pd
from sklearn.model_selection import train_test_split

"Class containing methods to manipulate tf.data"
class DataUtils:
    
    ##################################################################
    ########### Functions that return dataset useful info. ###########
    ##################################################################
    '''Function that returns number of sampels in dataset.'''
    @staticmethod
    def get_num_samples(dataset):
        return len(dataset[0])

    '''Function that returns mean/var/min/max/error of a distribution given samples and clients.'''
    @staticmethod
    def get_statistics(num_samples, num_clients, var):

        # Mean & Var
        mean = num_samples/num_clients
        var = int(var*mean)
        # Min-Max Values
        minimum,maximum = mean-var, int(mean+var)
        # Errors
        error = num_samples-minimum*num_clients # values to be distributed
        round_error = (error-int(error)) + ((mean-int(mean))*num_clients)

        return mean,var,(int(minimum),maximum),(int(error),round_error)

    '''Function that returns classes distribution in dataset.'''
    @staticmethod
    def get_class_distribution(dataset):
        return collections.Counter(dataset[1])

    '''Function that split datasets samples into partitions.'''
    @staticmethod
    def split_class_samples(dataset, partitions=1):

        total = len(dataset[0])
        size = total // partitions
        rest = total % partitions
        ranges = []

        assert size!=0 and partitions>0

        if rest:
            index = [x for x in range(0, total, size)]
            extra = [index[i] + i for i in range(rest + 1)] + [x + rest for x in index[rest+1:][:partitions-rest]]
            ranges = [(extra[i], extra[i+1]) for i in range(len(extra) - 1)]

        else:
            index = [x for x in range(0, total+1, size)]
            ranges = [(index[i], index[i+1]) for i in range(len(index) - 1)]

        return [(dataset[0][i:j], dataset[1][i:j]) for i, j in ranges]

    '''Function that returns dataset's samples corresponding to a specific class.'''
    @staticmethod
    def get_dataset_class_samples(dataset,class_number):

        dataset = list(zip(*[ (dataset[0][idx],label) for idx,label in enumerate(dataset[1]) if label==class_number]))
        return list(dataset[0]), list(dataset[1])

    '''Function that returns a subset of a dataset keeping the same class distribution.'''
    
    '''
    @staticmethod
    def get_subset(dataset, percentages, num_classes, seed=2021):

        # Get number of samples
        #num_samples = DataUtils.get_num_samples(dataset=dataset)
        # Get class distribution in dataset
        class_distribution = DataUtils.get_class_distribution(dataset=dataset)
        class_distribution = dict(sorted(class_distribution.items()))
        # Get number of samples per class in new subset
        class_samples = {k: int(percentages[0]*v) for k, v in class_distribution.items()}
        class_samples = dict(sorted(class_samples.items()))
        # Convert dataset to list
        dataset = list(zip(dataset[0], dataset[1]))
        # Split dataset samples per class
        dataset_per_class = [[s for s in dataset if s[1] == i] for i in range(num_classes)]            
        # Convert number of samples per class from dict to list
        class_num_samples = list(class_samples.values())
        # Check that all classes are present in train set
        assert len(class_num_samples) == len(dataset_per_class), "Train/Test Spliting Mismatch.\nNot all classes are present in the training set."
        # FIXED Random Seed
        random.seed(seed)
        # Get Subset of labelled samples - FIX Random SEED
        subset = list(itertools.chain(*[[dataset_per_class[i].pop(random.Random(seed).randrange(len(dataset_per_class[i]))) for _ in range(class_num_samples[i])] for i in range(num_classes)]))
        random.Random(seed).shuffle(subset)
        # Get Subset of unlabelled samples
        subset_u = list(itertools.chain(*dataset_per_class))
        random.Random(seed).shuffle(subset_u)
        # Keep a random percentage of the unlabeled dataset
        subset_u = subset_u[0:int((len(subset_u)-1)*percentages[1])] if percentages[1] < 1.0 else subset_u

        return (subset,len(subset)),(subset_u,len(subset_u))
    '''
    
    @staticmethod
    def get_subset(dataset, percentages, num_classes, speakers=['Random',None,'_'], seed=2021):

        if speakers[0] == 'Speaker' and percentages[0]<1.0: # Create speaker,files,labels structure
            df = pd.DataFrame([(f,f.split('/')[-1].split(speakers[-1])[0],l) for f,l in zip(dataset[0], dataset[1])], columns =['Files', 'Speakers', 'Labels'])
            train, test, _, _ = train_test_split(df,df[['Speakers', 'Labels']], train_size=percentages[0], random_state=seed, stratify=df[['Speakers', 'Labels']])
            subset, subset_u = [tuple(x) for x in train[['Files','Labels']].values], [tuple(x) for x in test[['Files','Labels']].values]
        else:
            #df = pd.DataFrame([(f,l) for f,l in zip(dataset[0], dataset[1])], columns =['Files', 'Labels'])
            #train, test, _, _ = train_test_split(df,df[['Labels']], train_size=percentages[0], random_state=seed, stratify=df[['Labels']])
            #subset, subset_u = [tuple(x) for x in train[['Files','Labels']].values],[tuple(x) for x in test[['Files','Labels']].values]
            
            # Get class distribution in dataset
            class_distribution = DataUtils.get_class_distribution(dataset=dataset)
            class_distribution = dict(sorted(class_distribution.items()))
            # Get number of samples per class in new subset
            class_samples = {k: int(percentages[0]*v) for k, v in class_distribution.items()}
            class_samples = dict(sorted(class_samples.items()))
            # Convert dataset to list
            dataset = list(zip(dataset[0], dataset[1]))
            # Split dataset samples per class
            dataset_per_class = [[s for s in dataset if s[1] == i] for i in range(num_classes)]            
            # Convert number of samples per class from dict to list
            class_num_samples = list(class_samples.values())
            # Check that all classes are present in train set
            assert len(class_num_samples) == len(dataset_per_class), "Train/Test Spliting Mismatch.\nNot all classes are present in the training set."
            # FIXED Random Seed
            random.seed(seed)
            # Get Subset of labelled samples - FIX Random SEED
            subset = list(itertools.chain(*[[dataset_per_class[i].pop(random.Random(seed).randrange(len(dataset_per_class[i]))) for _ in range(class_num_samples[i])] for i in range(num_classes)]))
            random.Random(seed).shuffle(subset)
            # Get Subset of unlabelled samples
            subset_u = list(itertools.chain(*dataset_per_class))
            random.Random(seed).shuffle(subset_u)
            # Keep a random percentage of the unlabeled dataset
            subset_u = subset_u[0:int((len(subset_u)-1)*percentages[1])] if percentages[1] < 1.0 else subset_u
        return (subset,len(subset)),(subset_u,len(subset_u))

    ##################################################################
    ########### Functions that distribute tf.data objects. ###########
    ##################################################################

    '''Function that split dataset to clients varying number of samples.'''
    @staticmethod
    def distribute_per_samples(dataset, num_clients, variance=0.25, tfds=False, seed=2021):
        
        # Convert to list
        dataset = tuple(list(t) for t in zip(*dataset))
        # Get number of samples
        num_samples = DataUtils.get_num_samples(dataset=dataset)
        # Get statistics of dataset
        mean, var, limits, errors = DataUtils.get_statistics(num_samples=num_samples, num_clients=num_clients, var=variance)
        # Get samples distribution
        distribution = DataUtils.create_distribution(num_clients=num_clients, num_samples=num_samples, remain_samples=errors[0], minimum=limits[0], maximum=limits[1], round_error=errors[1], seed=seed)
        # Extract Samples and Labels from Dataset
        samples, labels = DataUtils.convert_to_tuple(dataset) if tfds else dataset
        # Shuffle Dataset
        dataset = DataUtils.shuffle_dataset(dataset=dataset, seed=seed)
        # Create Iterators for samples & labels
        iter_samples, iter_labels = iter(samples), iter(labels)
        # Slice Dataset to Clients Partisions
        for i in range(num_clients):
            yield list(itertools.islice(iter_samples, distribution[i])), list(itertools.islice(iter_labels, distribution[i]))

    '''Function that split dataset to clients varying number of classes.'''
    @staticmethod
    def distribute_per_class(dataset, num_clients, num_classes, class_variance=0.0, tfds=False, seed=2021):

        # Convert to list
        dataset = tuple(list(t) for t in zip(*dataset))
        # Test for overlap
        (overlap, _num_classes_, _num_clients_) = (True,num_clients, num_classes) if num_classes < num_clients else (False,num_classes,num_clients)
        # Get statistics of dataset
        mean, var, limits, errors = DataUtils.get_statistics(num_samples=_num_classes_, num_clients=_num_clients_, var=class_variance)
        # Get classes distribution
        distribution =  DataUtils.create_distribution(num_clients=_num_clients_, num_samples=_num_classes_, remain_samples=errors[0], minimum=limits[0], maximum=limits[1], round_error=errors[1], seed=seed)
        # Fix random seed
        random.seed(seed)

        # Calculate corresponding class for each client.
        if overlap:
            # Clients will contain only one class and clients classes will overlap
            distribution = list(itertools.chain(*[[idx]*clients for idx,clients in enumerate(distribution)]))
            random.shuffle(distribution)
        else:
            classes = [i for i in range(_num_classes_)]
            random.shuffle(classes)
            _iter_ = iter(classes)
            distribute_fun = lambda: (yield from (list(itertools.islice(_iter_, distribution[i])) for i in range(_num_clients_)))
            distribution = [i for i in distribute_fun()]

        # Convert class number to class samples
        class_datasets = [DataUtils.get_dataset_class_samples(dataset=dataset, class_number=class_number) for class_number in range(num_classes)]

        # Distribute dataset to clients according to distribution
        if overlap:
            # Find number of clients that share dataset.
            dataset_use = dict(sorted(collections.Counter(distribution).items()))
            # Split classes into partitions
            class_datasets = [ class_datasets[c] if dataset_use[c]==1 else DataUtils.split_class_samples(class_datasets[c],dataset_use[c]) for c in range(num_classes) ]
            # Keep track of splitted partitions
            parts = list(itertools.chain.from_iterable([[i for i in range(len(c))] for c in class_datasets if type(c) is list]))
            # Fill datasets with examples from class distribution.
            _iter_ = iter(parts)
            datasets = [ class_datasets[i] if dataset_use[i]==1 else class_datasets[i][next(_iter_)] for i in distribution]
            # Shuffle each dataset.
            datasets = [DataUtils.shuffle_dataset(dataset) for dataset in datasets]
        else:
            # Fill datasets with examples from class distribution.
            datasets = [[class_datasets[class_number] for class_number in i] for i in distribution]
            # Merge clients datasets with more than one class.
            datasets = [np.hstack((datasets[i])).tolist() for i in range(len(datasets))]
            # Convert to tuple of lists.
            datasets = [tuple(l) for l in datasets]
            # Shuffle each dataset.
            datasets = [DataUtils.shuffle_dataset(dataset) for dataset in datasets]

        # Convert labels from str back to int32.
        datasets = [ [dataset[0],[int(label) for label in dataset[1]]] for dataset in datasets]

        return datasets

    '''Function that split dataset to clients based on speaker id.'''
    def distribute_per_speaker(dataset, num_clients, speakers, delimeter):

        # Get list of speakers
        speakers = list(filter(None, speakers.value.split(' ')))
        # Create one hot encode of speakers names
        one_hot_speakers = {k: v for k, v in dict(enumerate(speakers)).items()}
        # Convert to lists of tuples
        dataset = tuple(list(t) for t in zip(*dataset))
        # Split (files,labels) per speaker and store in list of tuples (One tuple per speaker contain pairs of (files,labels)) 
        datasets = [[(file,label) for file,label in zip(dataset[0],dataset[1]) if file.split('/')[-1].split(delimeter)[0] == one_hot_speakers[s]] for s in range(len(speakers))]
        # Convert back to list of lists : [[[Speaker_1_files],[Speaker_1_labels]],,...]
        datasets = [list(zip(*dataset)) for dataset in datasets if len(dataset)!=0]
        datasets = [ (list(dataset[0]),list(dataset[1])) for dataset in datasets]
        # Ensure sufficients partitions exist for all clients
        assert num_clients <= len(datasets),'Number of Clients exceeds the number of speakers ('+str(len(datasets))+')'

        return datasets

    '''Function that split dataset to clients varying number of classes.'''
    @staticmethod
    def distribute_per_class_with_class_limit(dataset, num_clients, num_classes, dist_mean=0.5, dist_var=0.25, seed=2021):

        # Convert to list
        dataset = tuple(list(t) for t in zip(*dataset))
        # Fix Random Seed
        random.seed(seed)
        np.random.seed(seed)
        # Create Class Probability Distribution
        probability_distribution = np.random.normal(int(dist_mean*num_classes), np.sqrt(dist_var) , num_clients)
        probability_distribution = [int(p) for p in probability_distribution]
        # Create class distribution across clients
        distributions = [list(np.random.choice(a=num_classes, size=p, replace=False)) for p in probability_distribution]
        # Find number of clients that share dataset.
        dataset_use = dict(sorted(collections.Counter(list(itertools.chain(*distributions))).items()))
        # Make sure that all classes are used. Otherwise change seed. 
        assert len(dataset_use)==num_classes, "Not all classes were selected. Choose a different seed!"
        # Split classes into clients partitions
        class_datasets = [DataUtils.get_dataset_class_samples(dataset=dataset, class_number=class_number) for class_number in range(num_classes)]
        class_datasets = [class_datasets[c] if dataset_use[c]==1 else DataUtils.split_class_samples(class_datasets[c],dataset_use[c]) for c in range(num_classes)]
        # Keep track of splitted partitions        
        parts = [list(np.arange(i)) for i in list(dataset_use.values())]    
        # Fill datasets with examples from class distribution.
        datasets = [[ class_datasets[j] if (len(parts[j])==1 and type(class_datasets[j])==tuple) else class_datasets[j][parts[j].pop()] for j in dis] for dis in distributions]
        # Flatten clients datasets to ([samples],[labels]) format.
        datasets = [ [list(itertools.chain(*list(zip(*d))[0])),list(itertools.chain(*list(zip(*d))[1]))] for d in datasets ]
        # Shuffle each dataset.
        datasets = [DataUtils.shuffle_dataset(d) for d in datasets]
        
        # Convert labels from str back to int32.
        #datasets = [ [dataset[0],[int(label) for label in dataset[1]]] for dataset in datasets]

        return datasets

    ##################################################################
    ################# Functions that create objects. #################
    ##################################################################

    '''Function that create a samples distribution given the number of clients.'''
    @staticmethod
    def create_distribution(num_clients, num_samples, remain_samples, minimum, maximum, round_error, seed=2021):

        # Fill distribution with minimum number of samples
        distribution = [minimum]*num_clients
        # Initialize Seed
        random.seed(seed)
        # Distribute remaining samples.
        while remain_samples > 0:
            idx = random.randint(0, len(distribution)-1)
            distribution[idx], remain_samples = (distribution[idx]+1, remain_samples-1) if distribution[idx] < maximum else (distribution[idx], remain_samples)
        # Add error due to rounding to random positions.
        for i in range(round(round_error)):
            idx = random.randint(0, len(distribution)-1)
            distribution[idx] = distribution[idx]+1 if sum(distribution)+1<=num_samples else distribution[idx]

        return distribution

    # TODO: Deprecated implementation. Remove in final version.
    '''Function that calculates tfds split configuration.'''
    @staticmethod
    def create_tfds_split(dataset):

        split=''
        for sample_id,_ in zip(dataset[0],dataset[1]):
            split+=('+train['+str(sample_id)+':'+str(sample_id+1)+']')

        return split[1:]
    
    ##################################################################
    ############# Functions that convert tf.data objects. ############
    ##################################################################

    '''Function that converts labelled to unlabelled samples.'''
    @staticmethod
    def convert_to_unlabelled(dataset, unlabelled_class_label=-1):
        
        dataset = [(data[0], unlabelled_class_label) for data in dataset]

        return dataset

    # TODO: Deprecated implementation. Remove in final version.
    '''Function that converts tfds dataset to a dummy dataset with format (sample_idx,label) - Decrease memory footprint'''
    @staticmethod
    def convert_to_dummy(dataset):
        return tf.data.Dataset.zip((tf.data.Dataset.range(len(dataset)), dataset.map(lambda _,label: label)))

    # TODO: Deprecated implementation. Remove in final version.
    '''Function that covert a tf.dataset object to tuple of (samples,labels).'''
    @staticmethod
    def convert_to_tuple(dataset):

        data = [(file.numpy(), label.numpy())for file, label in data.take(-1)]
        return ([x[0] for x in data], [x[1] for x in data])

    '''Function that covert a list of (samples,labels) to tf.dataset object.'''
    @staticmethod
    def convert_to_dataset(dataset, output_shapes=(tf.TensorShape((None,)), tf.TensorShape(())), output_types=(tf.int32, tf.int32)):

        return tf.data.Dataset.from_generator(lambda: (sample for sample in list(zip(dataset[0], dataset[1]))), output_types=output_types, output_shapes=output_shapes)

    '''Function that merge two dataset in tfds forms.'''
    @staticmethod
    def shuffle_merge(dataset_a, dataset_b):

        zipped = tf.data.Dataset.zip((dataset_a, dataset_b))

        def concat_func(x_a, x_b):
            return tf.data.Dataset.from_tensors(x_a).concatenate(tf.data.Dataset.from_tensors(x_b))

        alternated = zipped.flat_map(concat_func)

        return alternated

    '''Function that shuffles the samples of a dataset.'''
    @staticmethod
    def shuffle_dataset(dataset,seed=2021):

        # FIXED seed
        random.seed(seed)
        # Convert to tuple of (sample,label) and shuffle
        dataset = list(zip(dataset[0],dataset[1]))
        random.shuffle(dataset)

        return tuple(list(t) for t in zip(*dataset))

    ##################################################################
    ###################### Deprecated Functions. #####################
    ##################################################################

    @staticmethod
    def to_percentage(table,samples):
        return [100*(x/samples) for x in table]  

    @staticmethod
    def random_table(samples, parts, variance, seed=2021):

        # Calculate Variance
        mean = int(samples/parts)
        r_mean = samples/parts
        variance = int(variance*r_mean)
        round_error = (samples-(r_mean-variance)*parts)-int(samples-(r_mean-variance)*parts) + (r_mean-mean)*parts

        # Initialize Table with Minimum Size
        table = [mean - variance] * parts

        # Fill partition randomly until #samples
        table = DataUtils.distribute_values(table=table, values=int(samples-(r_mean-variance)*parts) , error=round_error, total=samples, maximum=mean+variance, _seed_=2021)

        return table

    @staticmethod
    def distribute_values(table, values, error, total, maximum, _seed_=2021):

        # Initialize Seed
        random.seed(_seed_)

        # Distribute values
        while values > 0:
            idx = random.randint(0,len(table)-1)
            table[idx], values = (table[idx]+1 , values-1) if table[idx]<maximum else (table[idx], values)

        # Add error due to rounding to a random position 
        idx = random.randint(0,len(table)-1)
        table[idx] = table[idx]+round(error) if sum(table)+round(error) <= total else table[idx]

        return table

    @staticmethod
    def shuffle_data(data, seed=2021):

        random.Random(seed).shuffle(data)
        map(shuffle, data)

        return data
