import logging
from utils.optimizer import adjust_learning_rate


class EarlyStopping():

    def __init__(
        self,
        epochs_limit_without_improvement,
        epochs_since_last_improvement,
        baseline, encoder_optimizer,
        decoder_optimizer,
        period_decay_lr=5
    ):
        self.epochs_limit_without_improvement = epochs_limit_without_improvement
        self.epochs_since_last_improvement = epochs_since_last_improvement
        self.best_loss = baseline
        self.stop_training = False
        self.improved = False
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.period_decay_lr = period_decay_lr

    def check_improvement(self, current_loss):

        if current_loss < self.best_loss:

            logging.info("melhorei a loss. Current %s %s",
                         current_loss, self.best_loss)

            self.best_loss = current_loss
            self.epochs_since_last_improvement = 0
            self.improved = True

        else:

            self.epochs_since_last_improvement += 1
            self.improved = False
            logging.info("Val without improvement. Not improving since %s epochs",
                         self.epochs_since_last_improvement)

            if self.epochs_since_last_improvement >= self.epochs_limit_without_improvement:
                logging.info("Early stopping")
                self.stop_training = True

            # Decay learning rate if there is no improvement for x consecutive epochs
            if self.epochs_since_last_improvement % self.period_decay_lr == 0:
                logging.info("Decay learning rate")
                adjust_learning_rate(self.decoder_optimizer, 0.8)
                if self.encoder_optimizer:
                    adjust_learning_rate(self.encoder_optimizer, 0.8)

    def get_number_of_epochs_without_improvement(self):
        return self.epochs_since_last_improvement

    def is_current_val_best(self):
        return self.improved

    def is_to_stop_training_early(self):
        return self.stop_training

   # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 20:
        #     break
        # if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
        #     adjust_learning_rate(decoder_optimizer, 0.8)
        #     if fine_tune_encoder:
        #         adjust_learning_rate(encoder_optimizer, 0.8)
