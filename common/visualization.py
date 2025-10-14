import matplotlib.pyplot as plt
from IPython.display import clear_output


class LossPlot:
    def __init__(self, title="Training Loss", xlabel="Epoch", ylabel="Loss"):
        self.losses = []
        self.val_losses = []
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = (8, 5)

    def update(self, train_loss, val_loss=None):
        """
        Add a new loss value and update the live plot.

        Args:
            train_loss (float): Training loss for the current epoch.
            val_loss (float, optional): Validation loss for the current epoch.
        """
        self.losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        else:
            self.val_losses.append(None)

        clear_output(wait=True)
        plt.figure(figsize=self.figsize)
        plt.plot(self.losses, label="Train Loss", color="b")

        if any(v is not None for v in self.val_losses):
            plt.plot(
                [v for v in self.val_losses if v is not None],
                label="Validation Loss",
                color="r",
            )

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
