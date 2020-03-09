import torchvision
from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from models.abtract_model import AbstractEncoderDecoderModel
import torch.nn.functional as F
from embeddings.embeddings import get_embedding_layer


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14, enable_fine_tuning=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # TODO: mudar isto
        resnet = torchvision.models.resnet101(
            pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]

        self.model = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size))

        # Disable calculation of all gradients
        for p in self.model.parameters():
            p.requires_grad = False

        # Enable calculation of some gradients for fine tuning
        self.fine_tune(enable_fine_tuning)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(
            images)  # (batch_size, 2048, image_size/32, image_size/32)
        # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = self.adaptive_pool(out)
        # (batch_size, encoded_image_size, encoded_image_size, 2048)
        out = out.permute(0, 2, 3, 1)
        return out

    def fine_tune(self, enable_fine_tuning):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: Allow?
        """
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = enable_fine_tuning


class Decoder(nn.Module):
    """
    Decoder.
    """

    def __init__(self, decoder_dim,  embed_dim, embedding_type, vocab_size, token_to_id, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(Decoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = get_embedding_layer(
            embedding_type, embed_dim, vocab_size, token_to_id)

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(
            embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        # linear layer to find initial hidden state of LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # linear layer to find initial cell state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, word, encoder_out, decoder_hidden_state, decoder_cell_state):
        embeddings = self.embedding(word)

        decoder_hidden_state, decoder_cell_state = self.decode_step(
            embeddings, (decoder_hidden_state, decoder_cell_state)
        )

        scores = self.fc(self.dropout(decoder_hidden_state))

        return scores, decoder_hidden_state, decoder_cell_state


class BasicModel(AbstractEncoderDecoderModel):

    MODEL_DIRECTORY = "experiments/results/basic_model/"

    def __init__(self,
                 args,
                 vocab_size,
                 token_to_id,
                 id_to_token,
                 max_len,
                 device
                 ):
        super().__init__(args, vocab_size, token_to_id, id_to_token, max_len, device)

    def _initialize_encoder_and_decoder(self):

        self.decoder = Decoder(
            decoder_dim=self.args.decoder_dim,
            embedding_type=self.args.embedding_type,
            embed_dim=self.args.embed_dim,
            vocab_size=self.vocab_size,
            token_to_id=self.token_to_id,
            dropout=self.args.dropout
        )

        self.encoder = Encoder(enable_fine_tuning=self.args.fine_tune_encoder)
        self.decoder = self.decoder.to(self.device)
        self.encoder = self.encoder.to(self.device)

    def _define_loss_criteria(self):
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def _calculate_loss(self, encoder_out, caps, caption_lengths):
        batch_size = encoder_out.size(0)

        # Create tensors to hold word predicion scores
        all_predictions = torch.zeros(batch_size, max(
            caption_lengths), self.vocab_size).to(self.device)

        h, c = self.decoder.init_hidden_state(encoder_out)

        # Predict
        for t in range(max(caption_lengths)):
            # batchsizes of current time_step are the ones with lenght bigger than time-step (i.e have not fineshed yet)
            batch_size_t = sum([l > t for l in caption_lengths])

            predictions, h, c = self.decoder(
                caps[:batch_size_t, t], encoder_out[:batch_size_t], h[:batch_size_t], c[:batch_size_t])

            all_predictions[:batch_size_t, t, :] = predictions

        # Calculate loss
        targets = caps[:, 1:]  # targets doesnt have stark token
        # pack scores and target
        scores = pack_padded_sequence(
            all_predictions, caption_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, caption_lengths, batch_first=True)

        loss = self.criterion(scores.data, targets.data)

        return loss

    def generate_output_index(self, input_word, encoder_out, h, c):
        predictions, h, c = self.decoder(
            input_word, encoder_out, h, c)

        scores = F.log_softmax(predictions, dim=1)
        current_output_index = torch.argmax(scores, dim=1)

        return current_output_index, h, c
