"""
Training and Evaluation Pipeline for Deep Learning Option Pricing Models.
Compares:
1. Classical Black-Scholes (Baseline)
2. Simple ANN Direct (X -> C_market)
3. Simple ANN Residual (X -> E, C_pred = C_bs + E_hat)
4. Option LSTM (Sequential dynamics + static contract features)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import generate_market_options_data, generate_time_series_option_sequences
from model import SimpleOptionANN, ResidualOptionANN, OptionLSTM


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_by_moneyness(df_test, y_pred_col, pred_name="Model"):
    results = []
    for cat in ['OTM', 'ATM', 'ITM']:
        sub = df_test[df_test['moneyness_cat'] == cat]
        if len(sub) == 0:
            continue
        rmse = np.sqrt(mean_squared_error(sub['C_market'], sub[y_pred_col]))
        mae = mean_absolute_error(sub['C_market'], sub[y_pred_col])
        results.append({
            'Model': pred_name,
            'Moneyness': cat,
            'Count': len(sub),
            'RMSE': rmse,
            'MAE': mae
        })
    return results


def train_ann_direct(X_train, y_train, X_val, y_val, epochs=60, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 1] Training Simple ANN Direct: X -> C_market")
    print("="*60)
    
    input_dim = X_train.shape[1]
    model = SimpleOptionANN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(bx)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_ann_residual(X_train, y_err_train, X_val, y_err_val, epochs=60, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 2] Training Research Model (Residual ANN): X -> E (E = C_market - C_bs)")
    print("="*60)
    
    input_dim = X_train.shape[1]
    model = ResidualOptionANN(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_err_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_err_val, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(bx)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Error Loss: {train_loss:.6f} | Val Error Loss: {val_loss:.6f}")

    return model, train_losses, val_losses


def train_lstm(X_seq_tr, X_st_tr, y_tr, X_seq_val, X_st_val, y_val, epochs=50, batch_size=64, lr=0.001, device='cpu'):
    print("\n" + "="*60)
    print(" [Step 3] Training Option LSTM: (X_seq, X_static) -> C_market")
    print("="*60)
    
    seq_input_dim = X_seq_tr.shape[2]
    static_input_dim = X_st_tr.shape[1]
    
    model = OptionLSTM(
        seq_input_dim=seq_input_dim,
        static_input_dim=static_input_dim,
        lstm_hidden=64,
        dense_hidden=32,
        dropout_prob=0.1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_seq_tr, dtype=torch.float32),
        torch.tensor(X_st_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_seq_val, dtype=torch.float32),
        torch.tensor(X_st_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for b_seq, b_st, by in train_loader:
            b_seq, b_st, by = b_seq.to(device), b_st.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(b_seq, b_st)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(b_seq)
        
        train_loss = running_loss / len(train_ds)

        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for b_seq, b_st, by in val_loader:
                b_seq, b_st, by = b_seq.to(device), b_st.to(device), by.to(device)
                pred = model(b_seq, b_st)
                vloss = criterion(pred, by)
                running_vloss += vloss.item() * len(b_seq)
        val_loss = running_vloss / len(val_ds)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

    return model, train_losses, val_losses


def plot_results(df_test, train_losses_dict, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training Loss Curves
    plt.figure(figsize=(10, 5))
    for name, (tr_loss, val_loss) in train_losses_dict.items():
        plt.plot(val_loss, label=f'{name} (Val Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Training & Validation Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300)
    plt.close()

    # 2. Predicted vs. Actual Scatter (Sample of 300 points)
    sample_df = df_test.sample(min(300, len(df_test)), random_state=42).sort_values('C_market')
    plt.figure(figsize=(12, 6))
    plt.plot(sample_df['C_market'].values, sample_df['C_market'].values, 'k--', label='Ideal 1:1 Parity', alpha=0.7)
    plt.scatter(sample_df['C_market'].values, sample_df['pred_bs'].values, color='gray', alpha=0.5, label='Black-Scholes (Constant Vol)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_ann_direct'].values, color='blue', alpha=0.6, label='ANN Direct (X -> C)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_ann_resid'].values, color='green', alpha=0.6, label='ANN Residual (X -> E)')
    plt.scatter(sample_df['C_market'].values, sample_df['pred_lstm'].values, color='purple', alpha=0.6, label='Option LSTM')
    plt.xlabel('Actual Market Price ($)')
    plt.ylabel('Predicted Option Price ($)')
    plt.title('Option Pricing Accuracy: Predicted vs. Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # 3. Pricing Error vs Moneyness (The Volatility Smile Effect)
    plt.figure(figsize=(12, 6))
    sort_idx = df_test.sort_values('moneyness')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_bs']), color='red', alpha=0.3, label='Black-Scholes Error |C_market - C_bs|')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_ann_direct']), color='blue', alpha=0.3, label='ANN Direct Absolute Error')
    plt.scatter(sort_idx['moneyness'], np.abs(sort_idx['C_market'] - sort_idx['pred_ann_resid']), color='green', alpha=0.3, label='ANN Residual Absolute Error')
    plt.axvline(0.97, color='black', linestyle=':', label='OTM/ATM Boundary')
    plt.axvline(1.03, color='black', linestyle='--', label='ATM/ITM Boundary')
    plt.xlabel('Moneyness (S / K)')
    plt.ylabel('Absolute Pricing Error ($)')
    plt.title('Pricing Discrepancy Across Moneyness (Smile / Skew Impact)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_across_moneyness.png'), dpi=300)
    plt.close()

    print(f"\n[Artifacts] Visual plots successfully saved to '{output_dir}/'")


def run_pipeline(n_samples=10000, epochs=50, batch_size=64, lr=0.001):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Compute Device: {device}")
    
    # 1. Generate realistic market dataset
    print(f"Generating synthetic options dataset with {n_samples} contracts...")
    df = generate_market_options_data(n_samples=n_samples, seed=42)
    
    # Train / Validation / Test split (70% train, 15% val, 15% test)
    n = len(df)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train+n_val].copy()
    df_test = df.iloc[n_train+n_val:].copy()

    # Prepare features for ANN Direct: [S, K, T, r, sigma_hist] or normalized [S/K, T, r, sigma_hist, C_bs/K]
    ann_features = ['moneyness', 'T', 'r', 'sigma_hist']
    
    scaler_x = StandardScaler()
    X_train_ann = scaler_x.fit_transform(df_train[ann_features].values)
    X_val_ann = scaler_x.transform(df_val[ann_features].values)
    X_test_ann = scaler_x.transform(df_test[ann_features].values)

    scaler_y_direct = StandardScaler()
    y_train_direct = scaler_y_direct.fit_transform(df_train[['C_market']].values)
    y_val_direct = scaler_y_direct.transform(df_val[['C_market']].values)

    # Prepare targets for ANN Residual: E = C_market - C_bs
    scaler_y_resid = StandardScaler()
    y_train_resid = scaler_y_resid.fit_transform(df_train[['C_error']].values)
    y_val_resid = scaler_y_resid.transform(df_val[['C_error']].values)

    # 2. Step 1: Train ANN Direct
    ann_direct, tr_loss_dir, val_loss_dir = train_ann_direct(
        X_train_ann, y_train_direct, X_val_ann, y_val_direct,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 3. Step 2: Train ANN Residual
    ann_resid, tr_loss_res, val_loss_res = train_ann_residual(
        X_train_ann, y_train_resid, X_val_ann, y_val_resid,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 4. Step 3: Train Option LSTM
    print("\nPreparing temporal sequences for LSTM...")
    X_seq, X_static, _, _ = generate_time_series_option_sequences(df, seq_len=15, seed=42)
    
    X_seq_tr = X_seq[:n_train]
    X_st_tr = X_static[:n_train]
    
    X_seq_val = X_seq[n_train:n_train+n_val]
    X_st_val = X_static[n_train:n_train+n_val]
    
    X_seq_te = X_seq[n_train+n_val:]
    X_st_te = X_static[n_train+n_val:]

    lstm_model, tr_loss_lstm, val_loss_lstm = train_lstm(
        X_seq_tr, X_st_tr, y_train_direct,
        X_seq_val, X_st_val, y_val_direct,
        epochs=epochs, batch_size=batch_size, lr=lr, device=device
    )

    # 5. Evaluate All Models on Unseen Test Set
    print("\n" + "="*60)
    print(" Evaluating Models on Out-of-Sample Test Set")
    print("="*60)

    # Baseline BS
    pred_bs = df_test['C_bs'].values

    # ANN Direct
    ann_direct.eval()
    with torch.no_grad():
        t_x_test = torch.tensor(X_test_ann, dtype=torch.float32).to(device)
        pred_scaled_dir = ann_direct(t_x_test).cpu().numpy()
        pred_ann_direct = scaler_y_direct.inverse_transform(pred_scaled_dir).flatten()
        pred_ann_direct = np.maximum(0.0, pred_ann_direct)

    # ANN Residual
    ann_resid.eval()
    with torch.no_grad():
        pred_scaled_res = ann_resid(t_x_test).cpu().numpy()
        pred_error = scaler_y_resid.inverse_transform(pred_scaled_res).flatten()
        # Final price = C_bs + E_hat
        pred_ann_resid = np.maximum(0.0, df_test['C_bs'].values + pred_error)

    # LSTM
    lstm_model.eval()
    with torch.no_grad():
        t_seq = torch.tensor(X_seq_te, dtype=torch.float32).to(device)
        t_st = torch.tensor(X_st_te, dtype=torch.float32).to(device)
        pred_scaled_lstm = lstm_model(t_seq, t_st).cpu().numpy()
        pred_lstm = scaler_y_direct.inverse_transform(pred_scaled_lstm).flatten()
        pred_lstm = np.maximum(0.0, pred_lstm)

    df_test['pred_bs'] = pred_bs
    df_test['pred_ann_direct'] = pred_ann_direct
    df_test['pred_ann_resid'] = pred_ann_resid
    df_test['pred_lstm'] = pred_lstm

    y_test_true = df_test['C_market'].values

    # Overall Metrics Table
    m_bs = evaluate_predictions(y_test_true, pred_bs, "Classical Black-Scholes")
    m_dir = evaluate_predictions(y_test_true, pred_ann_direct, "ANN Direct (X -> C)")
    m_res = evaluate_predictions(y_test_true, pred_ann_resid, "ANN Residual (X -> E)")
    m_lstm = evaluate_predictions(y_test_true, pred_lstm, "Option LSTM")

    overall_df = pd.DataFrame([m_bs, m_dir, m_res, m_lstm])
    print("\n--- OVERALL TEST PERFORMANCE ---")
    print(overall_df.to_string(index=False))

    # Moneyness Stratified Metrics Table (Matching the Research Paper)
    b_bs = evaluate_by_moneyness(df_test, 'pred_bs', "Classical Black-Scholes")
    b_dir = evaluate_by_moneyness(df_test, 'pred_ann_direct', "ANN Direct (X -> C)")
    b_res = evaluate_by_moneyness(df_test, 'pred_ann_resid', "ANN Residual (X -> E)")
    b_lstm = evaluate_by_moneyness(df_test, 'pred_lstm', "Option LSTM")

    money_df = pd.DataFrame(b_bs + b_dir + b_res + b_lstm)
    print("\n--- PERFORMANCE STRATIFIED BY MONEYNESS (OTM / ATM / ITM) ---")
    print(money_df.to_string(index=False))

    # Save metrics
    os.makedirs("outputs", exist_ok=True)
    overall_df.to_csv("outputs/overall_metrics.csv", index=False)
    money_df.to_csv("outputs/moneyness_metrics.csv", index=False)

    # Save models
    torch.save(ann_direct.state_dict(), "outputs/ann_direct.pth")
    torch.save(ann_resid.state_dict(), "outputs/ann_resid.pth")
    torch.save(lstm_model.state_dict(), "outputs/lstm_model.pth")

    # Plot graphs
    loss_dict = {
        'ANN Direct': (tr_loss_dir, val_loss_dir),
        'ANN Residual': (tr_loss_res, val_loss_res),
        'Option LSTM': (tr_loss_lstm, val_loss_lstm)
    }
    plot_results(df_test, loss_dict, output_dir="outputs")

    return overall_df, money_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Option Pricing Models (ANN vs. LSTM & Residual)")
    parser.add_argument('--samples', type=int, default=8000, help="Number of option contracts to generate")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    run_pipeline(n_samples=args.samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
