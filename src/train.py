
import torch
import logging
from preprocess_data.tokens import convert_captions_to_Y
from create_data_files import PATH_RSICD, get_vocab_info, get_dataset
from datasets import CaptionDataset, TrialDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from preprocess_data.images import augment_image
from args_parser import get_args
# from models.attention.attention_model import Encoder, DecoderWithAttention
from models.attention.attention_with_padded import Encoder, DecoderWithAttention
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


PATH_DATASETS_RSICD = PATH_RSICD+"datasets/"

if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)

    args = get_args()
    logging.info(args.__dict__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s \nCount %i gpus",
                 device, torch.cuda.device_count())

    vocab_info = get_vocab_info(PATH_DATASETS_RSICD+"vocab_info.json")
    vocab_size, token_to_id, id_to_token, max_len = vocab_info[
        "vocab_size"], vocab_info["token_to_id"], vocab_info["id_to_token"], vocab_info["max_len"]
    logging.info("vocab size %s", vocab_size)

    train_dataset_args = (PATH_DATASETS_RSICD+"train.json",
                          PATH_RSICD+"raw_dataset/RSICD_images/",
                          "TRAIN",
                          max_len,
                          token_to_id
                          )

    val_dataset_args = (PATH_DATASETS_RSICD+"val.json",
                        PATH_RSICD+"raw_dataset/RSICD_images/",
                        "VAL",
                        max_len,
                        token_to_id)

    if args.augment_data:
        transform = transforms.Compose([
            augment_image(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                 std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                 std=[0.229, 0.224, 0.225])
        ])

    train_dataloader = DataLoader(
        CaptionDataset(*train_dataset_args, transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        CaptionDataset(*val_dataset_args, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Model parameters
    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5

    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=vocab_size,
                                   dropout=dropout)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=args.decoder_lr)
    encoder = Encoder()

    encoder.fine_tune(args.fine_tune_encoder)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=args.encoder_lr) if args.fine_tune_encoder else None

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    for i, (imgs, caps_input, caps_target, cap_len) in enumerate(train_dataloader):
        print("this i", i)
        print("img", imgs.size())
        print("capt size", caps_input.size())

        print("capt", caps_input)
        print("caps_tar", caps_target)
        print("caption len", cap_len)
        print("caption len size", cap_len.size())

        decoder.train()  # train mode (dropout and batchnorm is used)
        encoder.train()

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps_input.to(device)
        caplens = cap_len.to(device)

        # Forward prop.
        imgs = encoder(imgs)

        print("np shape encoder imgs", imgs.size())

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens)

        # print("this scores", scores.size())

        # preds = torch.max(scores.data, dim=2)
        # print("preds", preds)

        # # indices=tensor([[ 797,  987, 2543,  408, 1125, 1908, 1099, 1250,  353, 2604,  940, 2728,
        # #  1125, 2269,   77, 2118, 1125, 1455,  376, 1125],
        # # [1780,  330, 2297,  546, 2672, 1311,  987,  434,   98, 2118, 2455,  214,
        # #  1766,   98,  388, 2741, 2741, 2741, 2741, 2741]]))

        # targets = caps_sorted[:, 1:]
        # print("this are the targets", targets)

        # scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)

        # preds = torch.max(scores.data, dim=1)
        # print("preds agains", preds)

        # targets = pack_padded_sequence(
        #     targets, decode_lengths, batch_first=True)

        # print("this scores", scores.data.size())
        # print("this targets", targets.data.size())
        # # this scores torch.Size([35, 2742])
        # # this targets torch.Size([35])

        # print("new targets ", targets)

        # losses = 0
        # losses2 = []
        # for i in range(35):
        #     score = scores.data[i, :].unsqueeze(0)
        #     pred = torch.max(score, dim=1)[1]
        #     target = torch.LongTensor([targets.data[i]])
        #     print("score and target", pred, target)
        #     # print("shape score and targe", score.size(), target.size())

        #     loss = criterion(score, target)
        #     print("this is loss", loss)
        #     losses += loss
        #     losses2.append(loss.item())
        # print("this is final loss", losses/35)

        # #[8.048075675964355, 7.870279312133789, 7.890999794006348, 7.729227066040039, 8.021017074584961, 7.960415363311768, 7.881063938140869, 7.812114238739014, 7.605221748352051, 8.255047798156738, 8.201086044311523, 7.787632942199707, 7.919833660125732, 8.308751106262207, 7.854781627655029, 7.93643856048584, 7.538877487182617, 8.196657180786133, 7.66047477722168, 7.93613862991333, 7.782405853271484, 8.187858581542969, 7.99949836730957, 8.603503227233887, 8.212371826171875, 7.601338863372803, 8.035136222839355, 7.778920650482178, 7.826919078826904, 7.667997360229492, 7.810032844543457, 8.209424018859863, 8.074832916259766, 8.334678649902344, 7.999967575073242]

        # loss = criterion(scores.data, targets.data)
        # print("this loss", loss)  # 7.9583
        # print("final losses", losses2)

        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)

        preds = torch.max(scores, dim=2)
    #  indices=tensor([[ 797,  987, 2543,  408, 1125, 1908, 1099, 1250,  353, 2604,  940, 2728,
    #      1125, 2269,   77, 2118, 1125, 1455,  376, 1125,  617,  122,  868, 1250,
    #       868, 1099,  481,  592, 1020, 1250, 2503,  987, 2672, 2249, 1933],
    #     [1780,  330, 2297,  546, 2672, 1311,  987,  434,   98, 2118, 2455,  214,
    #      1766,   98,  388, 1084, 1556, 1311,  732, 1280, 2141,  115, 1780, 2728,
    #      2027,  372,  547, 2080,  547,  557, 1588, 1596,  137, 2375, 1236]]))

        print("these are preds", preds)
        print("scores before", scores.size())

        scores = scores.view(-1, 2742)
        print("target before", caps_target)

        caps_target = caps_target.view(-1)
        print("This are targets", caps_target)

        print("new scores data", scores.data.size())
        print("this are the real targets", caps_target.size())

        losses = 0
        losses2 = []
        for i in range(70):
            score = scores[i, :].unsqueeze(0)
            pred = torch.max(score, dim=1)[1]
            target = torch.LongTensor([caps_target[i]])
            print("score and target", pred, target)
            # print("shape score and targe", score.size(), target.size())

            loss = criterion(score, target)
            print("this is loss", loss)
            losses += loss
            losses2.append(loss.item())
        print("this is final loss", losses/35)  # 7.9471

        loss = criterion(scores, caps_target)
        print("this loss", loss)  # 7.9808
        print("final losses", losses2)

        break

    # lixo_dataloader = DataLoader(
    #     TrialDataset([1, 2, 3, 4, 5]),
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=1,
    #     pin_memory=True
    # )

    # for i, data in enumerate(lixo_dataloader):
    #     print("this i", i)
    #     print("data", data)

    # por imagem a funcar...
    # por transformers da imagem...
    # ver o caps len...
    # ver se consegues por augmentation (normalização e rodar,etc...)

    # train = get_dataset(PATH_DATASETS_RSICD+"train.json")
    # train_images_names, train_captions_of_tokens = train["images_names"], train["captions_tokens"]
    # logging.info("len train_images_names %s", len(train_images_names))

    # val = get_dataset(PATH_DATASETS_RSICD+"/val.json")
    # val_images_names, val_captions_of_tokens = val["images_names"], val["captions_tokens"]
    # logging.info("len val_images_names %s", len(val_images_names))

    # # TODO: fix this to consider only captins with len<35 (Filtering two lists simultaneously)
    # logging.info(
    #     "captions tokens -> captions ids, since the NN doesnot read tokens but rather numbers")
    # train_input_captions, train_target_captions = convert_captions_to_Y(
    #     train_captions_of_tokens, max_len, token_to_id)
    # val_input_captions, val_target_captions = convert_captions_to_Y(
    #     val_captions_of_tokens, max_len, token_to_id)

    # logging.info(
    #     "images names -> images vectors (respective representantion of an image)")
    # train_generator = None
    # val_generator = None

    # generator_args = (raw_dataset, args.image_model_type)
    # if args.fine_tuning:
    #     logging.info("Fine Tuning")

    #     if args.augment_data:
    #         logging.info("with augmented images")
    #         generator = FineTunedAugmentedGenerator(*generator_args)

    #     else:
    #         logging.info("without augmented images")
    #         generator = FineTunedSimpleGenerator(*generator_args)

    # else:
    #     logging.info("Feature extraction")

    #     if args.augment_data:
    #         logging.info("with augmented images")

    #         generator = FeaturesExtractedAugmentedGenerator(*generator_args)

    #     else:
    #         logging.info("without augmented images")

    #         generator = FeaturesExtractedSimpleGenerator(*generator_args)

    # logging.info("create generators for datasets (train and val)")

    # train_generator = generator.generate(
    #     train_images_names,
    #     train_input_captions,
    #     train_target_captions,
    #     vocab_size
    # )

    # val_generator = generator.generate(
    #     val_images_names,
    #     val_input_captions,
    #     val_target_captions,
    #     vocab_size
    # )

    # #TODO: SUFFLE

    # train_dataset = tf.data.Dataset.from_generator(
    #     lambda: train_generator,
    #     ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32),
    # ).batch(args.batch_size)

    # val_dataset = tf.data.Dataset.from_generator(
    #     lambda: val_generator,
    #     ({'input_1': tf.float32, 'input_2': tf.float32}, tf.float32)
    # ).batch(args.batch_size)

    # logging.info("create and run model")
    # logging.info("qual é o tamanho do input %s",
    #              generator.get_shape_of_input_image())

    # model_class = globals()[args.model_class_str]

    # model = model_class(
    #     args,
    #     vocab_size,
    #     max_len,
    #     token_to_id,
    #     id_to_token,
    #     generator.get_shape_of_input_image(),
    #     args.embedding_type,
    #     args.units
    # )

    # model.create()
    # model.summary()
    # model.build()
    # model.train(train_dataset, val_dataset,
    #             len(train_images_names), len(val_images_names))

    # model.save()
