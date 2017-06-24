import os
import types
import jsonpickle

import tensorflow as tf
from abc import ABCMeta, abstractmethod


class AbstractModel(object):
    """Abstract class as a template for a TensorFlow model.
       
       References: Inspired by http://danijar.com/structuring-your-tensorflow-models/
    """
    __metaclass__ = ABCMeta
    
    def __init__(self, weight_decay=0.0):
        """Creates the base model instance that is shared accross all models.
           It allows to build multiple models using the same construction plan.
        Parameters
        ----------
        weight_decay: float, optional
            The weight decay regularization factor (lambda).
        """
        self._weight_decay = weight_decay
        
    @abstractmethod
    def inference(self, inputs, labels, is_training=True):
        """Builds the models inference.
           Note: Everytime this method function is called, a new
                 model instance in created.
        Parameters
        ----------
        inputs: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The inputs of the model.
        labels: 5D-Tensor of shape [batch_size, nstep, h, w, c]
            The target outputs of the model.
        is_training: Boolean, optional
            Flag inidcating training or eval mode. E.g. used for batch norm.
        """
        pass
    
    @abstractmethod
    def loss(self, predictions, labels):
        """Gets the loss of the model.
        Parameters
        ----------
        predictions: n-D Tensor
            The predictions of the model.
        labels: n-D Tensor
            The targets/labels.
        Returns
        ----------
        Returns the loss as a float.
        """
        pass
    
    def total_loss(self, loss, predictions, targets, device_scope=None):
        """Gets the total loss of the model including the regularization losses.
           Implemented as a lazy property.
        Parameters
        ----------
        loss: float32 Tensor
            The result of loss(predictions, targets) that should be included in this loss.
        predictions: n-D Tensor
            The predictions of the model.
        targets: n-D Tensor
            The targets/labels.
        device_scope: str or None, optional
            The tower name in case of multi-GPU runs.
        Returns
        ----------
        Returns the total loss as a float.
        """
        wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(wd_losses) > 0:
            reg_loss = tf.add_n(wd_losses, name="reg_loss")
            return tf.add(loss, reg_loss, name="loss_with_reg")
        else:
            # we have to wrap this with identity, because in in other case it
            # would rise a summary-writer error that this loss was already added
            return tf.identity(loss, name="loss_with_reg")
        
    @abstractmethod
    def validation(self, predictions, labels, device_scope=None):
        """Returns a dict of {title: scalar-Tensor, ...} that are evaluated during 
           validation and training.
           Note:
               All names must be a valid filename, as they are used in TensorBoard. 
        predictions: n-D Tensor
            The predictions of the model.
        labels: n-D Tensor
            The targets/labels.
        Returns
        ----------
        A dict of {title: scalar-Tensor, ...} to be executed in validation and testing.
        """
        pass
    
    def save(self, filepath):
        """Saves the model parameters to the specifiec path as JSON.
        Parameters
        ----------
        filepath: str
            The file path.
        """
        # check and create dirs
        if not os.path.exists(os.path.dirname(filepath)):
            subdirs = os.path.dirname(filepath)
            if subdirs is not None and subdirs != '':
                os.makedirs(subdirs)
        
        with open(filepath, 'wb') as f:
            json = jsonpickle.encode(self)
            f.write(json)
            
    def load(self, filepath):
        """Load the model parameters from the specifiec path as JSON.
        Parameters
        ----------
        filepath: str
            The file path.
        """
        with open(filepath, 'r') as f:
            json = f.read()
            model = jsonpickle.decode(json)
            self.__dict__.update(model.__dict__)
    
    def print_params(self):
        """Shows the model parameters."""
        params = self.__getstate__()
        
        def trim_prefix(text, prefix):
            return text[text.startswith(prefix) and len(prefix):]
        
        def to_string(value):
            if isinstance(value, types.FunctionType):
                return value.__name__.upper()
            return value

        print(">>> Model:")
        for name, value in params.iteritems():
            print("{:16}  ->  {}".format(trim_prefix(name, '_'), to_string(value)))

    @property
    def batch_size(self):
        """Gets the dynamic shape of the batch size."""
        return tf.shape(self._inputs)[0]
    
    @property
    def weight_decay(self):
        """Gets the regularization factor (lambda) for weight decay."""
        return self._weight_decay
