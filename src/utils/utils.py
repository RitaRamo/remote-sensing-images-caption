from torch.nn.utils.rnn import pack_padded_sequence


def get_pack_padded_sequences(predictions, targets,  caption_lengths):
    predictions = pack_padded_sequence(
        predictions, caption_lengths, batch_first=True)
    targets = pack_padded_sequence(
        targets, caption_lengths, batch_first=True)

    return predictions.data, targets.data
