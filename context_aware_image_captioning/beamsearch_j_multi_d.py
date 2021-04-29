"""
Variant of  beamsearch justiy with multiple distractor classes.
"""

import torch
import torchvision.transforms as transforms
from models import *
import torchfile as tf
from imageio import imread
from PIL import Image
import torch.nn.functional as F
import sys
import numpy as np
import pickle
from hyperparameters import *



def beam_search_justify(
    encoder, decoder, image_path, class_t, class_ds, lambda_, vocab_size, beam_size
):

    k = beam_size

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    # todo replace resize with adaptive pooling
    img = Image.fromarray(img).resize(size=(256, 256))
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    img = torch.FloatTensor(img).to(device)
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # treating problem like k is batch_size

    class_embedding_t = decoder.class_embedding(
        torch.LongTensor([[class_t]]).to(device)
    ).expand(k, 1, 512)
    class_embedding_ds = [decoder.class_embedding(
        torch.LongTensor([[class_d]]).to(device)
    ).expand(k, 1, 512) for class_d in class_ds]

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(
        k, num_pixels, encoder_dim
    )  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[start]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    # complete_seqs_alpha = list()
    complete_seqs_scores = list()
    incomplete_seqs_scores = list()

    # Start decoding
    step = 1
    h_t, c_t = decoder.init_hidden_state(encoder_out, class_embedding_t)
    # list of hidden/cell state pairs for each distractor
    hc_d = [decoder.init_hidden_state(encoder_out, class_embedding_d) for class_embedding_d in class_embedding_ds]

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        print(f"step {step} previous words {k_prev_words.size()}")
        print(k_prev_words)
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        scores_t, h_t, c_t = decoder_forward(decoder, embeddings, class_embedding_t, encoder_out, h_t, c_t)

        scores_ds = []

        # aggregate distractor scores and new cell/hidden states
        for i, ((h_d, c_d), class_embedding_d) in enumerate(zip(hc_d, class_embedding_ds)):
            score, h, c = decoder_forward(decoder, embeddings, class_embedding_d, encoder_out, h_d, c_d)
            scores_ds.append(score)
            hc_d[i] = (h, c)

        # sum over distractor classes, averaged to alter scores
        sum_ds = np.sum(scores_ds, 0) / len(scores_ds)
        scores = scores_t - (1 - lambda_) * sum_ds
        scores = torch.as_tensor(scores)
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)


        # Add new words to sequences, alphas
        seqs = torch.cat(
            [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
        )  # (s, step+1)
        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [
            ind for ind, next_word in enumerate(next_word_inds) if next_word != end
        ]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            print("completed a sequence")
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]

        h_t = h_t[prev_word_inds[incomplete_inds]]
        c_t = c_t[prev_word_inds[incomplete_inds]]

        hc_d = [ (h_d[prev_word_inds[incomplete_inds]], c_d[prev_word_inds[incomplete_inds]]) for h_d, c_d in hc_d]

        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]

        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        class_embedding_t = class_embedding_t[incomplete_inds, :, :]
        class_embedding_ds = [class_embedding_d[incomplete_inds, :, :] for class_embedding_d in class_embedding_ds]

        # Break if things have been going on too long
        if step > 15:
            break
        step += 1

    print("number of complete seqs: ", len(complete_seqs_scores))
    if len(complete_seqs_scores)!=0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        # alphas = complete_seqs_alpha[i]
    else:
        incomplete_seqs_scores.extend(top_k_scores[incomplete_inds])
        print("len of incomplete scores", len(incomplete_seqs_scores))
        print(incomplete_seqs_scores)
        i = incomplete_seqs_scores.index(max(incomplete_seqs_scores))
        print(  "i",i)
        seqs = seqs.tolist()
        print("seqs", len(seqs))
        seq = seqs[i]

    return seq

def decoder_forward(decoder, embeddings, class_embedding, encoder_out, h, c):
    awe_d = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
    gate_d = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
    awe_d = gate_d * awe_d

    h, c = decoder.decode_step(
        torch.cat([embeddings, awe_d, class_embedding[:, 0, :]], dim=1),
        (h, c),
    )  # (s, decoder_dim)
    scores = decoder.fc(h)
    scores = F.log_softmax(scores, dim=1)
    return scores.detach().numpy(), h, c

if __name__ == "__main__":
    print("sys args ", sys.argv)

    assert checkpoint_j is not None, "missing checkpoint"
    checkpoints = torch.load(checkpoint_j)
    encoder = checkpoints["encoder"]
    decoder = checkpoints["decoder"]
    encoder.eval()
    decoder.eval()
    image_path = sys.argv[1]
    # second arg is target
    target = int(sys.argv[2])
    # all other args are distractors
    distractors = [int(c) for c in sys.argv[3:]]

    if data_mode == "cub":
        word_map = tf.load(
        "data_cub/cvpr2016_cub/vocab_c10.t7" ,
        force_8bytes_long=True,
        )
        word_map = {word_map[i]: i for i in word_map}
    elif data_mode == "coco":
        with open('data_coco/vocab.pkl', "rb") as f:
            vocab = pickle.load(f)
            word_map = vocab.idx2word


    vocab_size = len(word_map) + 1
    #print(word_map)

    seq = beam_search_justify(encoder, decoder, image_path, target, distractors, lamb, vocab_size, beam_size)
    for i in seq[1:]:
        print(word_map[int(i)])
    print("")

