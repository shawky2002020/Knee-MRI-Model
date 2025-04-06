import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tabulate import tabulate  # Import tabulate for better table formatting
from modelBackbone import modelBackbone  # Import the model class you defined                     
from dataloaderPngT import train_dataloader, valid_dataloader,test_dataloader,device, class_names
from typing import List, Dict
from timeit import default_timer as timer
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(modelBackbone.parameters(), lr=0.0001, weight_decay=0.00001)

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               threshold):
    model.train()
    
 #   output_shape = get_output_shape(model)  # Dynamically get the output shape
    total_train_loss, total_train_acc = 0, 0
    
    for X, y in dataloader:
        X, y = X.to(device), y.to(device).float().unsqueeze(1)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = (torch.sigmoid(y_pred) > threshold).float()
        
        total_train_acc += (y_pred_class == y).sum().item() / y.size(0)

    return  total_train_loss / len(dataloader), total_train_acc / len(dataloader)

def valid_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              threshold):
    model.eval() 
    
    total_valid_loss, total_valid_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            total_valid_loss += loss.item()

            valid_pred_labels = (torch.sigmoid(y_pred) > threshold).float()       

            total_valid_acc += (valid_pred_labels == y).sum().item() / y.size(0)

    return  total_valid_loss / len(dataloader), total_valid_acc / len(dataloader)

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                threshold):
    model.eval()
    total_test_loss, total_test_acc = 0, 0
    all_labels = []  # To collect true labels for AUC calculation
    all_preds = []   # To collect predicted probabilities for AUC calculation

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            total_test_loss += loss.item()

            y_pred_probs = torch.sigmoid(y_pred)
            y_pred_class = (y_pred_probs > threshold).float()

            total_test_acc += (y_pred_class == y).sum().item() / y.size(0)

            all_labels.append(y.cpu().numpy())
            all_preds.append(y_pred_probs.cpu().numpy())

    # Calculate AUC for each class after gathering all labels and predictions
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, (all_preds > threshold).astype(int))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                 xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    return {
        "test_loss": total_test_loss / len(dataloader),
        "test_acc": total_test_acc / len(dataloader),
        "auc": roc_auc,
    }

def train(model, train_dataloader, valid_dataloader, optimizer, loss_fn, epochs, threshold):
    results = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer,
            threshold
        )
        valid_loss, valid_acc = valid_step(
            model, valid_dataloader, loss_fn,threshold
        )

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

    return results

def evaluate_on_test_set(model, test_dataloader, loss_fn,threshold):
    print("Evaluating on test set...")
    results = test_step(model, test_dataloader, loss_fn,threshold)
    headers = ["Metric", "Value"]
    table = [["Test Loss", f"{results['test_loss']:.4f}"],
             ["Test Accuracy", f"{results['test_acc']:.4f}"],
             ["AUC Score", f"{results['auc']:.4f}"]]
    
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    return results
    
# Plotting function for loss curves
def plot_curves(results: Dict[str, List[float]]):
    epochs = range(len(results['train_loss']))

    # Plot loss
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, results['valid_loss'], label='Valid Loss', color='red')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(epochs, results['valid_acc'], label='Valid Accuracy', color='red')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.legend()
    plt.show()

# Set number of epochs
NUM_EPOCHS = 5

if __name__ == '__main__':
    # Start the training timer
    start_time = timer()

    # Train the model
    model_results = train(model=modelBackbone, 
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn, 
                      epochs=NUM_EPOCHS,
                      threshold=0.5)

    # End timer and print total training time
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # Plot the results
    plot_curves(model_results)

    # Evaluate the model on the test set
    metrics = evaluate_on_test_set(modelBackbone, test_dataloader, loss_fn,threshold=0.5)

    # Access individual metrics
    test_loss = metrics['test_loss']
    test_acc = metrics['test_acc']
    auc = metrics['auc']  
    print(f"Test Accuracy: {test_acc:.2f}, AUC: {auc:.2f}")

    # Save the trained model
    MODEL_SAVE_PATH = "ACL_sagittal.pth"
    torch.save(modelBackbone.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")
