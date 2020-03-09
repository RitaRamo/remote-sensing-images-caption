import logging


class EarlyStopping():

    def __init__(self, epochs_limit_without_improvement, epochs_since_last_improvement, baseline):
        self.epochs_limit_without_improvement = epochs_limit_without_improvement
        self.epochs_since_last_improvement = epochs_since_last_improvement
        self.best_loss = baseline
        self.stop_training = False
        self.improved = False

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

    def get_number_of_epochs_without_improvement(self):
        return self.epochs_since_last_improvement

    def is_current_val_best(self):
        return self.improved

    def is_to_stop_training_early(self):
        return self.stop_training
