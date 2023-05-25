import matplotlib.pyplot as plt
import torch


def linear_model(x: torch.Tensor, weights: float, bias: float) -> torch.Tensor:
    """
    Scales x with weights then adds bias

    Args:
        x (torch.Tensor): Input data to pass through the linear model
        weights (float): Weights of predicting model
        bias (float): Offset of the predicting model

    Returns:
        torch.tensor: Resultant output of the model
    """
    return (weights * x) + bias


def loss_fn(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    """
    Computes the error between two tensors using mean squared error formula

    Args:
        predicted (torch.Tensor): A tensor usually containing a prediction from
            a model
        actual (torch.Tensor): Data to compare predicted with

    Returns:
        float: Mean squared error between predicted and actual
    """
    squared_diff = (predicted - actual) ** 2
    return squared_diff.mean()


def model_forward(
    train: torch.Tensor,
    test: torch.Tensor,
    params: torch.Tensor,
    is_training: bool,
) -> float:
    """
    Forward pass function for linear model

    Args:
        train (torch.Tensor): Training data to pass to linear model
        test (torch.Tensor): Testing data to compare with prediction from
            linear model
        params (torch.Tensor): Parameters of linear model, expected to be
            composed of weight and bias
        is_training (bool): Flag to enable gradient tracking, should be done
            with training data forward pass, not recommended with validation
            data as it will lead to overfitting

    Returns:
        float: Loss between model prediction data and testing data
    """
    with torch.set_grad_enabled(is_training):
        pred = linear_model(train, *params)
        loss = loss_fn(pred, test)
    return loss


def train(
    n_epochs: int,
    optimizer: torch.optim,
    params: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trains params as a linear model to fit the data in x_train, compares the
        models prediction with y_train to calculate training loss, params are
        then run with validation dataset to track overfitting, optimizer is
        used for backpropagation to adjust params. This is done n_epochs times

    Args:
        n_epochs (int): Number of iterations to train model
        optimizer (torch.optim): Optimizer used in backpropagation
        params (torch.Tensor): Parameters of model
        x_train (torch.Tensor): Training dataset for model to learn from
        y_train (torch.Tensor): Testing dataset to compare model performance
        x_val (torch.Tensor): Validation dataset for model to test on
        y_val (torch.Tensor): Dataset to compare validation output
    """
    train_loss, val_loss = torch.zeros(n_epochs), torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        t_loss = model_forward(x_train, y_train, params, True)
        train_loss[epoch] = t_loss
        val_loss[epoch] = model_forward(x_val, y_val, params, False)
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch: {epoch}, training loss: {t_loss:.3f}', end='')
            print(f', validation loss: {val_loss[epoch]:.3f}')

    return train_loss, val_loss


def main() -> None:
    temp_c = torch.tensor(
        [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
    )
    temp_unknown = torch.tensor(
        [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
    )
    n_samples = temp_unknown.shape[0]
    n_val = int(0.3 * n_samples)
    shuffled_idx = torch.randperm(n_samples)

    train_idx = shuffled_idx[:-n_val]
    val_idx = shuffled_idx[-n_val:]

    x_train = temp_unknown[train_idx]
    y_train = temp_c[train_idx]

    x_val = temp_unknown[val_idx]
    y_val = temp_c[val_idx]

    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    opt = torch.optim.Adam([params], lr=learning_rate)
    epochs = 2000
    train_loss, val_loss = train(
        epochs, opt, params, x_train, y_train, x_val, y_val
    )

    plt.title('Training loss vs. validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(
        [i for i in range(epochs)],
        train_loss.detach(),
        'b',
        label='Training loss',
    )
    plt.plot(
        [i for i in range(epochs)],
        val_loss.detach(),
        'r',
        label='Validation loss',
    )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
