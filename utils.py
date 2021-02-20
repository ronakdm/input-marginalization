import numpy as np
import pickle
import datetime
import time
import torch
import random
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def load_dataset(dataset_name):
    # `dataset_name` can be "train", "valid", or "test".
    input_ids = pickle.load(
        open(
            "input-marginalization/preprocessed_data/SST-2/input_ids_%s" % dataset_name,
            "rb",
        )
    )
    attention_masks = pickle.load(
        open(
            "input-marginalization/preprocessed_data/SST-2/attention_masks_%s"
            % dataset_name,
            "rb",
        )
    )
    labels = pickle.load(
        open(
            "input-marginalization/preprocessed_data/SST-2/labels_%s" % dataset_name,
            "rb",
        )
    )

    return TensorDataset(input_ids, attention_masks, labels)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def generate_dataloaders(batch_size):
    train_dataset = load_dataset("train")
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    print("{:>5,} training samples.".format(len(train_dataset)))

    val_dataset = load_dataset("valid")
    validation_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )
    print("{:>5,} validation samples.".format(len(val_dataset)))

    test_dataset = load_dataset("test")
    test_dataloader = DataLoader(
        test_dataset, sampler=RandomSampler(test_dataset), batch_size=batch_size
    )
    print("{:>5,} test samples.".format(len(test_dataset)))

    return train_dataloader, validation_dataloader, test_dataloader


def train(
    model,
    epochs,
    train_dataloader,
    validation_dataloader,
    optimizer,
    scheduler,
    save_dir,
    save_filename,
    device,
    seed_val=42,
):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []
    total_t0 = time.time()
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")

        t0 = time.time()
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_dataloader), elapsed
                    )
                )

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
                return_dict=True,
            )

            loss = output.loss
            logits = output.logits

            total_train_loss += loss.item()

            loss.backward()

            # TODO: See if this is needed.
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                output = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = output.loss
                logits = output.logits

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")

    print(
        "Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0))
    )

    # Save the model.
    torch.save(model, f"{save_dir}/{save_filename}.pt")
    pickle.dump(
        training_stats, open(f"{save_dir}/training_stats_{save_filename}.p", "wb")
    )


def test(model, test_dataloader, device, save_dir, save_filename):
    # ========================================
    #               Testing
    # ========================================

    print("")
    print("Testing...")

    t0 = time.time()

    model.eval()

    total_test_accuracy = 0
    total_test_loss = 0

    for batch in test_dataloader:

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss = output.loss
            logits = output.logits

        total_test_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        total_test_accuracy += flat_accuracy(logits, label_ids)

    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_test_accuracy))
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_time = format_time(time.time() - t0)
    print("  Test Loss: {0:.2f}".format(avg_test_loss))
    print("  Test took: {:}".format(test_time))

    test_stats = {
        "Test Loss": avg_test_loss,
        "Test Accur.": avg_test_accuracy,
        "Test Time": test_time,
    }
    pickle.dump(test_stats, open(f"{save_dir}/test_stats_{save_filename}.p", "wb"))
