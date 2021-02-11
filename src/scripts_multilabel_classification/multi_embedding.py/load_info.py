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

    print("\nPAPER300ef5allNEWS_newsplit_fine_attenscalepro (efficientb5_net_rsicd_allcaptions_emb_glove_smoothl1)")
    path="experiments/results/classification_efficientnet_b5all_rsicd_embedding_caption_glove_smoothl1.pth.tar"
    _load_weights_from_checkpoint(path)

    print("\nPAPER300ef5oneNEWS_newsplit_fine_attenscalepro")
    path="experiments/results/classification_efficientnet_b5one_rsicd_embedding_caption_glove_smoothl1.pth.tar"
    _load_weights_from_checkpoint(path)

    print("\PAPER300ef7oneOLD_fine_attenscaleprod_3comp_ef  (efficientb7oone_net_rsicd_caption_emb_glove_smoothl1)")
    path="experiments/results/classification_efficientnet_b7old_onersicd_embedding_caption_glove_smoothl1.pth.tar"
    _load_weights_from_checkpoint(path)

    print("\npaperr_newsplit_fine_attenscaleprod_1compr_eff")
    path='experiments/results/classification_efficientnet_b7_rsicd_embedding_caption_glove_smoothl1.pth.tar'
    _load_weights_from_checkpoint(path)