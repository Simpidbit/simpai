from typing import final
import torch

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
    def __init__(self, model, train_dataloader):
        """
        Initialize the Trainer with a model and training data loader.

        Args:
            model: The PyTorch model to be trained.
            train_dataloader: DataLoader providing training data batches.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.step_fn = None
        self.eval_fn = None
        self.epoch_idx_offset = 0

    def _is_notebook(self) -> bool:
        """
        Check if the current environment is a Notebook (Jupyter/Colab/VS Code, etc.).

        Returns:
            bool: True if running in a notebook environment, False otherwise.
        """
        try:
            # get_ipython is a global function automatically injected in IPython/Jupyter environments
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            elif 'google.colab' in str(get_ipython().__class__):
                return True   # Google Colab
            else:
                return False  # Other type
        except NameError:
            return False      # Standard Python Interpreter (no get_ipython)

    def set_step(self, step_fn):
        """
        Set the function to be called for each training step (batch).

        The step function should have the signature:
            step_fn(epoch_idx, model, tqdm_module, data) -> loss

        Args:
            step_fn: Callable function to execute for each batch.
                Should return the loss value.

        Returns:
            The step_fn parameter (allows decorator-style usage).
        """
        self.step_fn = step_fn
        return step_fn
    
    def set_eval(self, eval_fn):
        """
        Set the function to be called after each epoch for evaluation.

        The eval function should have the signature:
            eval_fn(epoch_idx, model, tqdm_module) -> None

        Args:
            eval_fn: Callable function to execute after each epoch.

        Returns:
            The eval_fn parameter (allows decorator-style usage).
        """
        self.eval_fn = eval_fn
        return eval_fn

    def load_checkpoint(self, filepath: str, device: torch.device):
        """
        Load a training checkpoint and restore model state.

        Args:
            filepath: Path to the checkpoint file.
            device: PyTorch device to load the checkpoint on.

        Returns:
            optimizer_state_dict: The optimizer state dictionary from the checkpoint,
                which can be used to restore the optimizer state.
        """
        # Load checkpoint from file
        checkpoint = torch.load(filepath, map_location = device)
        # Restore model parameters
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Restore epoch offset for resuming training
        self.epoch_idx_offset = checkpoint['epoch_idx_offset']
        return checkpoint['optimizer_state_dict']

    def train(self, epoch_num, save_checkpoint = None, optimizer = None):
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
        if optimizer is None:
            raise RuntimeError('optimizer is not set!')

        # Import appropriate tqdm for notebook or terminal environment
        if self._is_notebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        # Main training loop with epoch progress bar
        with tqdm(
            range(epoch_num),
            desc = 'Total progress (epoch)',
            leave = True,
            position = 0
        ) as epoch_pbar:
            # Run initial evaluation
            self.eval_fn(self.epoch_idx_offset - 1, self.model, tqdm)
            try:
                for idx in epoch_pbar:
                    epoch_idx = idx + self.epoch_idx_offset
                    # Set model to training mode
                    self.model.train()
                    # Inner loop with batch progress bar
                    with tqdm(
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
                            loss = self.step_fn(epoch_idx, self.model, tqdm, data)
                            if isinstance(loss, torch.Tensor):
                                loss = loss.item()
                            pbar.update(self.train_dataloader.batch_size)
                            if isinstance(loss, float):
                                pbar.set_postfix(status = 'Loading data', loss = f'{loss:.5g}')
                            else:
                                pbar.set_postfix(status = 'Loading data')

                    # Run evaluation after each epoch
                    self.eval_fn(epoch_idx, self.model, tqdm)
            except KeyboardInterrupt:
                print('\nTraining interrupted by user.')

                # Handle emergency checkpoint on interruption
                if isinstance(save_checkpoint, str):
                    while True:
                        ask_save = input('Save emergency checkpoint? (y/n): ')
                        if ask_save == 'y':
                            print(f'Saving emergency checkpoint to {save_checkpoint}...')
                            checkpoint = {
                                'model_state_dict': self.model.state_dict(),
                                'epoch_idx_offset': epoch_num + self.epoch_idx_offset,
                                'optimizer_state_dict': optimizer.state_dict(),
                            }
                            torch.save(checkpoint, save_checkpoint)
                            print('Emergency checkpoint saved.')
                            break
                        elif ask_save == 'n':
                            break
                        else:
                            print('Input \"y\" or \"n\". Try again.')
                raise

        # Save final checkpoint if path provided
        if isinstance(save_checkpoint, str):
            print(f'Saving checkpoint to {save_checkpoint}...')
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch_idx_offset': epoch_num + self.epoch_idx_offset,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, save_checkpoint)
            print('Checkpoint has been saved.')
