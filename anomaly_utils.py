


def calculate_reconstruction_errors(dataloader, criterion):
    reconstruction_errors = []
    for i, data in enumerate(dataloader):
        inputs = data.as_in_context(ctx).reshape((-1, 1, 1))
        predicted_value = model(inputs)
        reconstruction_error = criterion(predicted_value, input).asnumpy().flatten()
        reconstruction_errors.append(reconstruction_error)

    return reconstruction_errors


def evaluate_accuracy(data_iterator, model, L):
    loss_avg = 0.
    for i, data in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 1, feature_count))

        output = model(data)
        loss = L(output, data)
        loss_avg = (loss_avg * i + nd.mean(loss).asscalar()) / (i + 1)
#         print(f"dd{nd.mean(loss).asscalar() / (i + 1)}")
    print(f"loss_avg: {loss_avg}")
    return loss_avg


def calculate_threshold(tr_reconstruction_errors):
    return np.mean(tr_reconstruction_errors) + 3 * np.std(tr_reconstruction_errors)