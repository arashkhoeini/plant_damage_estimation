import torch


class DataPrefetcher(object):
    """
    Data prefetcher for efficient GPU loading during training.

    This class overlaps data loading with GPU computation by prefetching
    the next batch while the current batch is being processed.

    Args:
        loader (DataLoader): PyTorch DataLoader instance
        device (str or torch.device): Target device for data loading
        stop_after (int, optional): Stop after this many batches (for debugging)
    """

    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        """Return the number of batches in the loader."""
        return len(self.loader)

    def preload(self):
        """
        Preload the next batch to GPU memory asynchronously.

        This method loads the next batch from the dataloader and transfers
        it to GPU memory using a separate CUDA stream for non-blocking operation.
        """
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):  # type: ignore
            self.next_input = self.next_input.cuda(
                device=self.device, non_blocking=True
            )
            self.next_target = self.next_target.cuda(
                device=self.device, non_blocking=True
            )

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break
