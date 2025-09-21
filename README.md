# Air Quality Forecasting with LSTM  

## Overview  
This project builds an **LSTM-based deep learning model** to forecast **PM2.5 air pollutant concentration** using time-series data.  
The workflow includes **data preprocessing, feature engineering, sequence creation, model design, hyperparameter experiments, and final model training with callbacks**.  

## Dataset  
- **Train set (`train.csv`)** - contains features and target (`pm2.5`).  
- **Test set (`test.csv`)** - contains only features (no target).  
- Features include meteorological variables and station readings.  
- Target: **PM2.5 concentration**.  

## Preprocessing  
1. **Datetime conversion** - `datetime` column converted to proper datetime format.  
2. **Set index** - `datetime` used as index for time-series handling.  
3. **Missing values** - handled with mean imputation and time interpolation.  
4. **Feature/target split** - `X_train`, `y_train` from training set; `X_test` from test set.  
5. **Sequence creation** - converted into sequences of 24 time steps for LSTM input.  

## Model Architectures  
We experimented with multiple LSTM designs:  
1. **Baseline LSTM** (single LSTM + Dense).  
2. **Stacked LSTM** (two LSTMs + dropout).  
3. **Shallow/Deep variants** with different units, dropout, and optimizers.  
4. **Final refined model**:  
   - `LSTM(32, return_sequences=True)` - `BatchNorm` - `Dropout(0.03)`  
   - `LSTM(16)` - `BatchNorm` - `Dropout(0.03)`  
   - `Dense(1)`  

## Experiments  
- Conducted **15 experiments** varying:  
  - Units per layer  
  - Number of LSTM layers  
  - Dropout rates  
  - Learning rates  
  - Optimizers (`adam`, `sgd`, `rmsprop`)  
  - Batch sizes (16, 32, 64)  

**Results Summary (Validation RMSE):**

| Experiment              | Units1 | Units2 | Dropout | LR     | Optimizer | Batch | Epochs | Val_RMSE |
|--------------------------|--------|--------|---------|--------|-----------|-------|--------|----------|
| Exp1: baseline          | 64     | 32     | 0.2     | 0.001  | adam      | 32    | 50     | 75.67    |
| Exp2: more units        | 128    | 64     | 0.2     | 0.001  | adam      | 32    | 50     | 75.02    |
| Exp3: less units        | 32     | 16     | 0.2     | 0.001  | adam      | 32    | 50     | 75.41    |
| Exp4: higher dropout    | 64     | 32     | 0.4     | 0.001  | adam      | 32    | 50     | 77.74    |
| Exp5: lower dropout     | 64     | 32     | 0.1     | 0.001  | adam      | 32    | 50     | 75.71    |
| Exp6: lr=0.01           | 64     | 32     | 0.2     | 0.010  | adam      | 32    | 50     | 88.02    |
| Exp7: lr=0.0001         | 64     | 32     | 0.2     | 0.0001 | adam      | 32    | 50     | 77.66    |
| Exp8: SGD optimizer     | 64     | 32     | 0.2     | 0.010  | sgd       | 32    | 50     | NaN      |
| Exp9: RMSprop optimizer | 64     | 32     | 0.2     | 0.001  | rmsprop   | 32    | 50     | 78.50    |
| Exp10: batch size 64    | 64     | 32     | 0.2     | 0.001  | adam      | 64    | 50     | 79.83    |
| Exp11: batch size 16    | 64     | 32     | 0.2     | 0.001  | adam      | 16    | 50     | **74.85** |
| Exp12: 3 LSTM layers    | 128    | 64     | 0.3     | 0.001  | adam      | 32    | 50     | 78.54    |
| Exp13: shallow LSTM     | 32     | 0      | 0.2     | 0.001  | adam      | 32    | 50     | 107.30   |
| Exp14: big model        | 256    | 128    | 0.3     | 0.001  | adam      | 32    | 50     | 78.51    |
| Exp15: small fast model | 16     | 8      | 0.2     | 0.001  | adam      | 64    | 50     | 90.06    |

 **Best model**: Exp11 (Batch size 16, Adam optimizer, 2 LSTMs).  

## Training Strategy  
- **Optimizer**: Adam (lr=0.001, gradient clipping).  
- **Loss**: MSE.  
- **Metric**: RMSE.  
- **Callbacks**:  
  - EarlyStopping (patience=8)  
  - ModelCheckpoint (save best model)  
  - ReduceLROnPlateau (dynamic LR scheduling)  
  - TensorBoard (training monitoring)  

## Evaluation  
- Training loss curve plotted.  
- Best model selected by lowest validation RMSE.  
- Predictions generated on test set (no ground truth available).  

##  Next Steps  
- Train for longer (50–100 epochs) with early stopping.  
- Use cross-validation with rolling windows.  
- Deploy with FastAPI or Streamlit for real-time forecasting.  
