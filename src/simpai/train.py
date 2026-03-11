from typing import Callable, Optional, Self

from simpai import utils
if utils.is_notebook():
    import tqdm.notebook as tqdm
else:
    import tqdm

import torch
from typeguard import typechecked

from simpai import logger

class Trainer:
    """
    A flexible training framework for PyTorch models with progress tracking and checkpoint support.

    The Trainer class provides a structured approach to training PyTorch models with support for:
    - Progress bars with tqdm
    - Checkpoint saving and loading
    - Evaluation callbacks
    - KeyboardInterrupt handling with emergency checkpoint saving

    Attributes:
        model: The PyTorch model to train.
        train_dataloader: DataLoader providing training data.
        step_fn: Function called for each batch during training.
        eval_fn: Function called after each epoch for evaluation.
        epoch_idx_offset: Offset for epoch numbering (useful for resuming training).
    """
    @typechecked
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        user_data: dict = dict()
    ) -> None:
        """
        Initialize the Trainer with a model and training data loader.

        Args:
            model: The PyTorch model to be trained.
            train_dataloader: DataLoader providing training data batches.
        """
        self.model: torch.nn.Module = model
        self.train_dataloader: torch.utils.data.DataLoader = train_dataloader
        self.step_fn: Callable[[int, torch.nn.Module, dict, tqdm.tqdm, tuple[torch.Tensor, torch.Tensor]], torch.Tensor] | None = None
        self.eval_fn: Callable[[int, torch.nn.Module, dict, tqdm.tqdm], None] | None = None
        self.epoch_idx_offset: int = 0
        self.user_data: dict = user_data

    @typechecked
    def set_user_data(
        self,
        user_data: dict,
    ) -> None:
        self.user_data = user_data


    @typechecked
    def set_step(
        self, 
        step_fn: Callable[[int, torch.nn.Module, dict, tqdm.tqdm, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]
    ) -> Callable[[int, torch.nn.Module, dict, tqdm.tqdm, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Set the function to be called for each training step (batch).

        The step function should have the signature:
            step_fn(epoch_idx, model, tqdm.std.tqdm, data) -> loss

        Args:
            step_fn: Callable function to execute for each batch.
                Should return the loss value.

        Returns:
            The step_fn parameter (allows decorator-style usage).
        """
        self.step_fn = step_fn
        return step_fn
    
    @typechecked
    def set_eval(
        self,
        eval_fn: Callable[[int, torch.nn.Module, dict, tqdm.tqdm], None]
    ) -> Callable[[int, torch.nn.Module, dict, tqdm.tqdm], None]:
        """
        Set the function to be called after each epoch for evaluation.

        The eval function should have the signature:
            eval_fn(epoch_idx, model, tqdm.std.tqdm) -> None

        Args:
            eval_fn: Callable function to execute after each epoch.

        Returns:
            The eval_fn parameter (allows decorator-style usage).
        """
        self.eval_fn = eval_fn
        return eval_fn

    @typechecked
    def train(
        self, 
        epoch_num: int,
        interrupt_feedback: Optional[Callable[[int, Self], None]] = None,
    ) -> None:
        """
        Train the model for a specified number of epochs with progress tracking.

        Args:
            epoch_num: Number of epochs to train.
            save_checkpoint: Optional path to save the checkpoint after training.
                If None, no checkpoint is saved.
            optimizer: PyTorch optimizer to use for training.

        Raises:
            RuntimeError: If eval_fn, step_fn, or optimizer is not set.
            KeyboardInterrupt: Re-raised after optionally saving emergency checkpoint.
        """
        # Validate required components are set
        if self.eval_fn is None:
            raise RuntimeError('eval_fn is not set, please call set_eval() first!')
        if self.step_fn is None:
            raise RuntimeError('step_fn is not set, please call set_step() first!')

        # Import appropriate tqdm for notebook or terminal environment

        # Main training loop with epoch progress bar
        interrupt_idx: Optional[int] = None
        with tqdm.tqdm(
            range(epoch_num),
            desc = 'Total progress (epoch)',
            leave = True,
            position = 0
        ) as epoch_pbar:
            # Run initial evaluation
            self.eval_fn(self.epoch_idx_offset - 1, self.model, self.user_data, epoch_pbar)
            for idx in epoch_pbar:
                try:
                    epoch_idx = idx + self.epoch_idx_offset
                    # Set model to training mode
                    self.model.train()
                    # Inner loop with batch progress bar
                    with tqdm.tqdm(
                        total = len(self.train_dataloader.dataset),
                        desc = f'Progress of epoch {epoch_idx + 1}',
                        leave = False,
                        position = 1
                    ) as pbar:
                        pbar.set_postfix(status = 'Loading data')
                        loss = None
                        for data in self.train_dataloader:
                            if loss is None:
                                pbar.set_postfix(status = 'Computing')
                            else:
                                pbar.set_postfix(status = 'Computing', loss = f'{loss:.5g}')
                            # Execute training step
                            loss = self.step_fn(epoch_idx, self.model, self.user_data, pbar, data)
                            if isinstance(loss, torch.Tensor):
                                loss = loss.item()
                            pbar.update(self.train_dataloader.batch_size)
                            if isinstance(loss, float):
                                pbar.set_postfix(status = 'Loading data', loss = f'{loss:.5g}')
                            else:
                                pbar.set_postfix(status = 'Loading data')

                    # Run evaluation after each epoch
                    self.eval_fn(epoch_idx, self.model, self.user_data, epoch_pbar)
                except KeyboardInterrupt:
                    interrupt_idx = idx
                    break
        if interrupt_idx is not None:
            logger.error('Training interrupted by user.')
            if interrupt_feedback is not None:
                interrupt_feedback(interrupt_idx, self)
            raise KeyboardInterrupt
