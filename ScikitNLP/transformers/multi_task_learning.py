import torch
from allennlp.models.model import Model
from allennlp.training.optimizers import Optimizer

class MTLModel(Model):
    def __init__(self):
        """
        Whatever you need to initialize your MTL model/architecture.
        """

    def forward(self,
                task_name: str,
                tensor_batch: torch.Tensor):
        """
        Defines the forward pass of the model. This function is designed
        to compute a loss function defined by the user.
        It should return

        Parameters
        ----------
        task_name: ``str``, required
            The name of the task for which to compute the forward pass.
        tensor_batch: ``torch.Tensor``, required
            An embedding representation of the input to pass through the model.

        Returns
        -------
        output_dict: ``Dict[str, torch.Tensor]``
            An output dictionary containing at least the computed loss for the task of interest.
        """
        raise NotImplementedError

    def get_metrics(self,
                    task_name: str):
        """
        Compute and update the metrics for the current task of interest.

        Parameters
        ----------
        task_name: ``str``, required
            The name of the current task of interest.
        Returns
        -------
        A dictionary of metrics.
        """
        raise NotImplementedError


class MultiTaskTrainer():
    # https://github.com/huggingface/hmtl/blob/master/hmtl/training/multi_task_trainer.py
    # To add successive regularization: it prevents the parameter updates from being too far from the parameters at the previous epoch by adding an L2 penalty on the loss
    def __init__(self, model: Model, task_list: List[Task])
        self._model = model
        self._task_list = task_list

        self._optimizers = {}
        for task in self._task_list:
            self._optimizers[task._name] = Optimizer() # Set the Optimizer you like.
            # Each task can have its own optimizer and own learning rate scheduler.


    def train(self, n_epochs: int = 50):
        ### Instantiate the training generators ###
        self._tr_generators = {}
        for task in self._task_list:
            data_iterator = task._data_iterator
            tr_generator = data_iterator(task._train_data, num_epochs = None)
            self._tr_generators[task._name] = tr_generator

        ### Begin Training ###
        self._model.train() # Set the model to train mode.
        for i in range(n_epochs):
            for _ in range(total_nb_training_batches):
                task_idx = choose_task()
                task = self._task_list[task_idx]
                task_name = task._name

                next_batch = next(self._tr_generators[task._name]) # Sample the next batch for the current task of interest.

                optimizer = self._optimizers[task._name] # Get the task-specific optimizer for the current task of interest.
                optimizer.zero_grad()

                output_dict = self._model.forward(task_name = task_name,
                                                  tensor_batch = batch) #Forward Pass

                loss = output_dict["loss"]
                loss.backward() # Backward Pass
