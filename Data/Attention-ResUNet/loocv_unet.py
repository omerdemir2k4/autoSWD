device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

        if self.data.ndim == 2:
            self.data = self.data.unsqueeze(1)  # [N, 1, T]
        elif self.data.shape[-1] == 1:  # [N, T, 1] → [N, 1, T]
            self.data = self.data.permute(0, 2, 1)

        if self.label.ndim == 2:
            self.label = self.label.unsqueeze(1)
        elif self.label.shape[-1] == 1:
            self.label = self.label.permute(0, 2, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    spec = confusion_matrix(y_true, y_pred)[0,0] / np.sum(y_true==0)
    bacc = 0.5 * (rec + spec)
    roc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    swd = np.sum(y_pred)
    background = len(y_pred)
    swd_percent = 100 * swd / background
    return acc, f1, prec, rec, spec, bacc, roc, ap, swd, background, swd_percent

def train_one_fold(train_loader, val_loader, epochs=100):
    """
    Trains and evaluates the model for one fold of a cross-validation loop.
    Includes Early Stopping and a Learning Rate Scheduler.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        epochs (int): Maximum number of training epochs.
    
    Returns:
        Tuple: Performance metrics on the validation set.
    """
    model = AttentionResUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    patience = 10
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_state = None

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for x, y in loop:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())
        
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                out_val = model(x_val)
                probs = out_val.cpu().numpy().flatten()
                preds = (out_val.cpu().numpy() > 0.5).astype(int).flatten()
                labels = y_val.cpu().numpy().flatten()
                y_true.extend(labels)
                y_pred.extend(preds)
                y_prob.extend(probs)

        val_f1 = f1_score(y_true, y_pred)
        print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f} | Val F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict() 
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation F1 score.")
                break

        scheduler.step(val_f1)
        # Manually print the current learning rate to replace the verbose output
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")
        
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    final_y_true, final_y_pred, final_y_prob = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            probs = out.cpu().numpy().flatten()
            preds = (out.cpu().numpy() > 0.5).astype(int).flatten()
            labels = y.cpu().numpy().flatten()
            final_y_true.extend(labels)
            final_y_prob.extend(probs)

            # --- MORPHOLOGICAL OPERATIONS ADDED HERE ---
            kernel_size = 100 # You can tune this parameter
            kernel = np.ones(kernel_size)
            
            # Apply morphological closing
            preds = scipy.ndimage.binary_closing(preds, structure=kernel).astype(preds.dtype)
            # Apply morphological opening
            preds = scipy.ndimage.binary_opening(preds, structure=kernel).astype(preds.dtype)
            
            final_y_pred.extend(preds)

    return compute_metrics(np.array(final_y_true), np.array(final_y_pred), np.array(final_y_prob))


def segment_signal(x, y, window_size=1024, stride=1024):
    X_segments, Y_segments = [], []
    total_len = x.shape[0]
    for start in range(0, total_len - window_size + 1, stride):
        end = start + window_size
        X_segments.append(x[start:end].T)  # (1, 1024)
        Y_segments.append(y[start:end].T)  # (1, 1024)
    return np.array(X_segments), np.array(Y_segments)

def run_loocv(data_dir):
    x_paths = sorted(glob.glob(os.path.join(data_dir, "X_cleaned_*.npy")))
    y_paths = sorted(glob.glob(os.path.join(data_dir, "Y_cleaned_*.npy")))

    assert len(x_paths) == len(y_paths), "Mismatch in X/Y files"
    results = []

    for i in range(len(x_paths)):
        print(f"\n=== Fold {i+1}/{len(x_paths)} | Left out: {os.path.basename(x_paths[i])} ===")

        x_val_raw = np.load(x_paths[i])  # (T, 1)
        y_val_raw = np.load(y_paths[i])  # (T, 1)
        x_val, y_val = segment_signal(x_val_raw, y_val_raw)
        val_loader = DataLoader(EEGDataset(x_val, y_val), batch_size=256, shuffle=False)

        x_train_list, y_train_list = [], []
        for j in range(len(x_paths)):
            if j == i: continue
            x_raw = np.load(x_paths[j])
            y_raw = np.load(y_paths[j])
            x_seg, y_seg = segment_signal(x_raw, y_raw)
            x_train_list.append(x_seg)
            y_train_list.append(y_seg)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        train_loader = DataLoader(EEGDataset(x_train, y_train), batch_size=256, shuffle=True)

        fold_metrics = train_one_fold(train_loader, val_loader)
        results.append(fold_metrics)

    # Metrics Table
    columns = ["Accuracy", "F1 Score", "Precision", "Recall", "Specificity", "Balanced Acc",
               "ROC AUC", "Avg Precision", "SWD", "Background", "SWD %"]
    df = pd.DataFrame(results, columns=columns)
    df.index = [f"Animal {i+1}" for i in range(len(results))]
    df.loc["Mean ± SD"] = [f"{df[c].mean():.2f} ± {df[c].std():.2f}" if df[c].dtype != object else "" for c in df.columns]
    df.to_csv("loocv_results.csv", index=True)
    print("\n=== Final Table ===")
    print(df)


if __name__ == "__main__":
    run_loocv("data_dir_with_xy")
