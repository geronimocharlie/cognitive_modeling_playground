import torch
from datasets import CubDataset, get_coco_loader
import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
from hyperparameters import *
import numpy as np

if data_mode == 'cub':
    # old specifications taken from the github of the paper
    train_loader = torch.utils.data.DataLoader(
        CubDataset(transform=transforms.Compose([normalize])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    num_classes = 200
    vocab_size = 5725 + 1 # not sure were the plus 1 is coming from


elif 'coco' in data_mode:
    # mode val for less images
    train_loader = get_coco_loader(mode="val", transform=transforms.Compose([transforms.ToTensor(), normalize]), batch_size=batch_size, num_workers=workers)
    num_classes = train_loader.dataset.num_classes + 1
    vocab_size = train_loader.dataset.vocab_size + 1 # the one is somehow crucial unsure why
    print("num of data")
    print(len(train_loader.dataset))

else:
    print("please specify data_mode as 'coco' or 'cub'")
    raise NotImplemented

# Note that the resize is already done in the encoder, so no need to do it here again
if load:
    # Load the model from checkpoints
    checkpoints = torch.load("checkpoint_j")
    encoder = checkpoints["encoder"]
    decoder = checkpoints["decoder"]
    decoder_optimizer = checkpoints["decoder_optimizer"]
    epoch = checkpoints["epoch"]
    decoder_lr = decoder_lr * pow(0.8, epoch // 5)
    for param_group in decoder_optimizer.param_groups:
        param_group["lr"] = decoder_lr
else:
    epoch = 0
    encoder = Encoder()
    decoder = DecoderWithAttention_justify(
        attention_dim=attention_dim,
        embed_dim=emb_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        num_classes=num_classes,
    )
    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr
    )

criterion = nn.CrossEntropyLoss()
encoder = encoder.to(device)
decoder = decoder.to(device)
criterion = criterion.to(device)

decoder.train()
encoder.train()

for epoch in range(epoch, numepochs):
    if epoch % 5 == 0 and epoch > 0:
        # For every 5 epochs, the lr is annealed by 0.8
        decoder_lr *= 0.8
        for param_group in decoder_optimizer.param_groups:
            param_group["lr"] = decoder_lr

    for i, (img, caption, caplen, class_k) in tqdm(
        enumerate(train_loader), desc="Batch"
        ):

        img = img.to(device)
        caption = caption.to(device)
        caplen = caplen.to(device)
        class_k = class_k.to(device)
        img = encoder(img)

        scores, caps_sorted, decode_lengths, sort_ind = decoder(
            img, caption, caplen, class_k
        )
        targets = caps_sorted[:, 1:]
        # Suitable format, so that loss can be applied. The scores had unwated padding, that is removed. Similarly for target
        # resulting size (sum over lenghts for each bathc, vocab_size) -> eg. bacth size 2, lenghts 13 and 21, vocab 5000 -> (34,5000)
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True
        ).data

        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True
        ).data  # [bacth sieze, max lenght] to [sum over lenghts]

        # A gradient descend step
        loss = criterion(scores, targets)  # add gating scalar, to this loss
        decoder_optimizer.zero_grad()
        loss.backward()
        decoder_optimizer.step()
        tqdm.write(f"Loss {loss.detach().cpu().numpy()}")

    if (epoch%save_after) == 0:
        print("saving")
        save(epoch, encoder, decoder, decoder_optimizer, data_mode)

