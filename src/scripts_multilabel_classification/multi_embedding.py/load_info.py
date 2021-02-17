import os 
import torch
    
def _load_weights_from_checkpoint(checkpoint_path):

    if os.path.exists(checkpoint_path):
        print("checkpoint_path", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        print("checkpoint['epoch']", checkpoint['epoch'])
        print("checkpoint['val_loss']", checkpoint['val_loss'])

    else:
        logging.info(
            "No checkpoint")

if __name__ == "__main__":

    print("\nb5 TI")
    path='experiments/results/classification_efficientnet_rsicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)

    print("\nb5 new old com o codigo backup")
    path='experiments/results/classification_efficientnet_b5old_onersicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)

    print("\n b7oldseed")
    path='experiments/results/classification_efficientnet_b7oldseed_onersicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)

    print("\nb5 b5oldseed")
    path='experiments/results/classification_efficientnet_b5oldseed_onersicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)

    print("\nb5 classification_efficientnet_b5all_rsicd_embedding_caption_glove_smoothl1")
    path='experiments/results/classification_efficientnet_b5all_rsicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)

