import torch

def collate_fn_model1(batch):
    images, captions, meta_dicts = zip(*batch)

    images = torch.stack(images, dim=0)
    captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=0
    )

    # stack metadata
    merged_meta = {}
    first_keys = meta_dicts[0].keys()
    for key in first_keys:
        merged_meta[key] = torch.stack([m[key] for m in meta_dicts], dim=0)

    return images, captions, merged_meta
