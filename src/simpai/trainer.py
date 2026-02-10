import torch

class Trainer:
    def __init__(self, model, train_dataloader):
        self.model = model
        self.train_dataloader = train_dataloader
        self.step_fn = None
        self.eval_fn = None
        self.epoch_idx_offset = 0

    def __is_notebook(self) -> bool:
        """
        检查当前环境是否为 Notebook (Jupyter/Colab/VS Code等)。
        """
        try:
            # get_ipython 是在 IPython/Jupyter 环境中自动注入的全局函数
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
            return False      # Standard Python Interpreter (没有 get_ipython)

    def set_step(self, step_fn):
        self.step_fn = step_fn
        return step_fn
    
    def set_eval(self, eval_fn):
        self.eval_fn = eval_fn
        return eval_fn

    def load_checkpoint(self, filepath: str, device: torch.device):
        checkpoint = torch.load(filepath, map_location = device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch_idx_offset = checkpoint['epoch_idx_offset']
        return checkpoint['optimizer_state_dict']

    def train(self, epoch_num, save_checkpoint = None, optimizer = None):
        if self.__is_notebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        epoch_pbar = tqdm(
            range(epoch_num),
            desc = 'Total progress (epoch)',
            leave = True,
            position = 0
        )
        for idx in epoch_pbar:
            epoch_idx = idx + self.epoch_idx_offset
            self.model.train()
            with tqdm(
                total = self.train_dataloader.batch_size * len(self.train_dataloader),
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
                    loss = self.step_fn(epoch_idx, self.model, data)
                    if isinstance(loss, torch.Tensor):
                        loss = loss.item()
                    pbar.update(self.train_dataloader.batch_size)
                    if isinstance(loss, float):
                        pbar.set_postfix(status = 'Loading data', loss = f'{loss:.5g}')
                    else:
                        pbar.set_postfix(status = 'Loading data')

            self.eval_fn(epoch_idx, self.model, tqdm)
        epoch_pbar.close()

        if isinstance(save_checkpoint, str):
            print(f'Saving checkpoint to {save_checkpoint}...')
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'epoch_idx_offset': epoch_num + self.epoch_idx_offset,
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, save_checkpoint)
            print('Checkpoint has been saved.')
