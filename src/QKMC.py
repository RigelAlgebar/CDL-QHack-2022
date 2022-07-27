#!/usr/bin/env python
# coding: utf-8


import logging

import numpy as np
import random

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import QuantumInstance, algorithm_globals

#from qiskit.aqua.utils import split_dataset_to_data_and_labels
from qiskit_machine_learning.datasets import ad_hoc_data


#from qiskit.aqua.components.feature_maps.raw_feature_vector import RawFeatureVector
from qiskit_machine_learning.circuit.library import RawFeatureVector


#from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.circuit.library import ZZFeatureMap
from qiskit import Aer
from qiskit.providers import Backend, BackendV1, BackendV2

"""
This module implements the abstract base class for algorithm modules.

To create add-on algorithm modules subclass the QuantumAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from qiskit.providers.backend import Backend
from qiskit.utils import QuantumInstance, algorithm_globals
#from qiskit.aqua import aqua_globals, QuantumInstance, AquaError


class QuantumAlgorithm(ABC):
    """
    Base class for Quantum Algorithms.

    This method should initialize the module and
    use an exception if a component of the module is available.
    """
    @abstractmethod
    def __init__(self,
                 quantum_instance: Optional[
                     Union[QuantumInstance, Backend, Backend]]) -> None:
        self._quantum_instance = None
        if quantum_instance:
            self.quantum_instance = quantum_instance

    @property
    def random(self):
        """Return a numpy random."""
        return aqua_globals.random

    def run(self,
            quantum_instance: Optional[
                Union[QuantumInstance, Backend]] = None,
            **kwargs) -> Dict:
        """Execute the algorithm with selected backend.

        Args:
            quantum_instance: the experimental setting.
            kwargs (dict): kwargs
        Returns:
            dict: results of an algorithm.
        Raises:
            AquaError: If a quantum instance or backend has not been provided
        """
        if quantum_instance is None and self.quantum_instance is None:
            raise AquaError("A QuantumInstance or Backend "
                            "must be supplied to run the quantum algorithm.")
        if isinstance(quantum_instance, (Backend)):
            self.set_backend(quantum_instance, **kwargs)
        else:
            if quantum_instance is not None:
                self.quantum_instance = quantum_instance

        return self._run()


    @abstractmethod
    def _run(self) -> Dict:
        raise NotImplementedError()

    @property
    def quantum_instance(self) -> Optional[QuantumInstance]:
        """ Returns quantum instance. """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[QuantumInstance,
                                                       Backend]) -> None:
        """ Sets quantum instance. """
        if isinstance(quantum_instance, (Backend)):
            quantum_instance = QuantumInstance(quantum_instance)
        self._quantum_instance = quantum_instance

    def set_backend(self, backend: Union[Backend], **kwargs) -> None:
        """ Sets backend with configuration. """
        self.quantum_instance = QuantumInstance(backend)
        self.quantum_instance.set_config(**kwargs)


    @property
    def backend(self) -> Union[Backend]:
        """ Returns backend. """
        return self.quantum_instance.backend

    @backend.setter
    def backend(self, backend: Union[Backend]):
        """ Sets backend without additional configuration. """
        self.set_backend(backend)





class QKMC(QuantumAlgorithm):
    """
    Quantum K-means clustering algorithm.
    """
    
    CONFIGURATION = {
        'name': 'QKMC',
        'description': 'QKMC Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QKMC_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
        ],
    }

    BATCH_SIZE = 1000
    
    def __init__(self, feature_dim, is_quantum, backend, test_dataset, num_clusters=None):
        """
        K-means Clustering Classification Algorithm
        """
        super().__init__(backend)

        self.test_dataset = None
        self.num_classes = None
        
        self.feature_dim = feature_dim
        self.is_quantum = is_quantum
        
        if(num_clusters==None):
            self.setup_test_data(test_dataset)
        else:
            self.test_dataset = test_dataset
            self.num_clusters = num_clusters
        
        #self.backend = backend
    def setup_test_data(self, test_dataset):
        """
        """
        _, _, self.test_dataset, self.num_clusters = ad_hoc_data(training_size=10, test_size=30, n=self.feature_dim, gap=0.6)
        self.num_clusters = len(self.num_clusters)
        self.test_dataset = self.test_dataset[0]
    
    def _run(self):
        """
        Classify with k-means clustering algorithm
        """
        cluster_assignments = self.get_initial_clusters()
        stop = False
        count = 1
        while(not stop):
            if (count>6):
                print('Algorithm failed to converge: run again')
                return cluster_assignments
            cluster_assignments, stop = self.iterate(cluster_assignments)
            #print(count)
            #print(cluster_assignments)
            #print()
            count+=1
        return cluster_assignments
        
    def get_initial_clusters(self):
        """
        Randomly assign each datapoint to a cluster
        """

        cluster_assignments = {}
        cluster_arrays = []

        for i in range(self.num_clusters):
            cluster_arrays.append([])

        for i in self.test_dataset:
            cluster = random.randint(0, self.num_clusters-1)
            cluster_arrays[cluster].append(i)

        for i in range(len(cluster_arrays)):
            cluster_assignments.update({str(i): cluster_arrays[i]})
        #print(cluster_assignments)
        return cluster_assignments
    
    def iterate(self, cluster_assignments):
        stop = True
        new_cluster_assignments = {}
        centroids = []
        cluster_arrays = []

        for i in range(self.num_clusters):
            cluster_arrays.append([])
            centroids.append(QKMC.calculate_centroid(self.feature_dim, cluster_assignments[str(i)]))

        for i in self.test_dataset:
            closest_cluster = self.closest_cluster(i, centroids)
            cluster_arrays[closest_cluster].append(i)
            if(stop):
                old_cluster = 1000
                for j in range(len(cluster_assignments)):
                    for k in cluster_assignments[str(j)]:
                        if (k==i).all():
                            old_cluster = j
                if(old_cluster != closest_cluster):
                    stop=False
                    
        for i in range(len(cluster_arrays)):
            new_cluster_assignments.update({str(i): cluster_arrays[i]})
        return (new_cluster_assignments, stop)
    
    @staticmethod
    def calculate_centroid(feature_dim, cluster_array):
        """
        Calculate centroid of a cluster
        """
        if (len(cluster_array)==0):
            return np.zeros(feature_dim)
        centroid = []
        for i in range(feature_dim):
            featuremean = 0
            for j in range(len(cluster_array)):
                featuremean += cluster_array[j][i]
            featuremean /= len(cluster_array)
            centroid.append(featuremean)
        return centroid
    
    def closest_cluster(self, x, centroids):
        if (self.is_quantum):
            quant = QKMC.closest_cluster_quantum(self.backend, x, centroids)
            #classic = QKMC.closest_cluster_classical(x, centroids)
            #if (quant != classic):
                #print('Wrong')
            return quant
        else:
            return QKMC.closest_cluster_classical(x, centroids)
        
    @staticmethod
    def closest_cluster_classical(x, centroids):
        """
        Calculate closest centroid from a data point
        """
        distances = []
        for i in range(len(centroids)):
            distances.append(QKMC.classical_calculate_squared_distance(x, centroids[i]))
        return distances.index(min(distances))
    
    @staticmethod
    def closest_cluster_quantum(backend, x, centroids):
        """
        Calculate closest centroid from a data point
        """
        distances = []
        for i in range(len(centroids)):
            distances.append(QKMC.quantum_calculate_squared_distance(backend, x, centroids[i]))
        return distances.index(min(distances))
    
    @staticmethod
    def classical_calculate_squared_distance(x, y):
        squares = 0
        if(len(x) != len(y)):
            raise ValueError("x and y must be of same length")
        for i in range(len(x)):
            squares += (x[i]-y[i])**2
        return squares
    
    @staticmethod
    def quantum_calculate_squared_distance(backend, x, y):
        if(len(x) != len(y)):
            raise ValueError("x and y must be of same length")
            
        X = []
        Y = []
        
        for i in range(len(x)):
            if (x[i]<y[i]):
                X += [x[i]]
                Y += [y[i]]
            else:
                Y += [x[i]]
                X += [y[i]]
            
        X = np.array(x)
        Y = np.array(y)
        
        if (np.linalg.norm(Y) == 0):
            print('broken')
            return 0
        #feature vector converter
        c0 = ClassicalRegister(1)
        q0 = QuantumRegister(1)
        zerocircuit = QuantumCircuit(q0)
        zerocircuit.h(q0)
        
        fvc = RawFeatureVector(len(X))
        q1 = QuantumRegister(fvc.num_qubits)
        q2 = QuantumRegister(fvc.num_qubits)
        ketxcircuit = fvc.construct_circuit(X, qr=q1)
        ketycircuit = fvc.construct_circuit(Y, qr=q2)
        
        psicircuit = zerocircuit+ketxcircuit+ketycircuit
        for i in range(fvc.num_qubits):
            psicircuit.cswap(q0, q1[i], q2[i])
            
        psicircuit.barrier(q0, q1, q2)
        psicircuit.reset(q2)
        
        Z=0
        for i in range(len(X)):
            Z += X[i]**2+Y[i]**2
        
        fvc2 = RawFeatureVector(2)
        p1 = np.linalg.norm(X)
        p2 = -np.linalg.norm(Y)
        phi = np.array([p1, p2])
        phicircuit = fvc2.construct_circuit(phi, qr=q2)
        
        q3 = QuantumRegister(1)
        swapcircuit = psicircuit+phicircuit
        swapcircuit.add_register(q3)
        swapcircuit.add_register(c0)
        swapcircuit.h(q3)
        swapcircuit.cswap(q3, q0, q2[0])
        swapcircuit.h(q3)
        swapcircuit.measure(q3, c0)
        result = execute(swapcircuit, backend, shots=100000).result()
        squares = Z*((4*result.get_counts()['0']/100000.0)-2)
        #print('error ', abs(100*(squares - QKMC.classical_calculate_squared_distance(X, Y))/QKMC.classical_calculate_squared_distance(X, Y)), "%")
        return squares



