import torch

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=0
    )
    return images, captions
