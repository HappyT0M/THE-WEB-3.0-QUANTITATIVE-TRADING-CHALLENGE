import datetime
import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import talib    # 用于计算技术指标
import torch

# ======================
# 路径与常量定义
# ======================
# Kaggle 数据路径（请根据你的实际目录调整，如果在本地请修改为本地路径）
DATA_PATH = "/kaggle/input/avenir-hku-web/kline_data/train_data"  # ⚠️ 本地请改成你的 parquet 文件所在目录
SUBMISSION_ID_PATH = "/kaggle/input/avenir-hku-web/submission_id.csv"  # ⚠️ 本地请放置 submission_id.csv
OUTPUT_SUBMIT_PATH = "/kaggle/working/submit.csv"  # Kaggle 上的输出路径，本地可改成 ./submit.csv

# 模型训练使用的起始时间
START_DATETIME = datetime.datetime(2021, 3, 1, 0, 0, 0)

# 目标：预测未来多少个15分钟后的收益（如 96 = 1天 * 96个15分钟）
TARGET_HORIZON = 96

# 设备配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用计算设备: {DEVICE}")


class OptimizedModel:
    def __init__(self):
        self.data_path = DATA_PATH
        self.submission_id_path = SUBMISSION_ID_PATH
        self.start_datetime = START_DATETIME
        self.target_horizon = TARGET_HORIZON
        self.use_gpu = USE_GPU
        print(f"✅ 初始化模型，数据路径: {self.data_path}")
        print(f"🔍 是否启用 GPU 训速加速: {'是' if self.use_gpu else '否'}")

    def get_all_symbols(self):
        files = glob.glob(os.path.join(self.data_path, "*.parquet"))
        symbols = [os.path.basename(f).split(".")[0] for f in files]
        print(f"🔍 发现 {len(symbols)} 个交易对（symbol）")
        return symbols

    def load_all_data(self, symbols):
        dfs = []
        valid_symbols = []
        failed_symbols = []
        for s in symbols:
            file_path = os.path.join(self.data_path, f"{s}.parquet")
            if not os.path.exists(file_path):
                print(f"[!] 文件不存在: {file_path}")
                failed_symbols.append(s)
                continue

            try:
                df = pd.read_parquet(file_path)
                if 'timestamp' not in df.columns:
                    print(f"[!] {s} 缺少 timestamp 列")
                    failed_symbols.append(s)
                    continue
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp').sort_index()
                df = df.astype(np.float64)

                required_cols = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'amount', 'buy_volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"[!] {s} 缺少必要列: {missing_cols}")
                    failed_symbols.append(s)
                    continue

                df = df[df.index >= self.start_datetime]

                future_return = df['close_price'].shift(-self.target_horizon) / df['close_price'] - 1
                df['target'] = future_return
                df['target'] = df['target'].replace([np.inf, -np.inf], np.nan)

                df['rsi'] = talib.momentum.RSIIndicator(df['close_price'], window=14).rsi()
                macd = talib.trend.MACD(df['close_price'])
                df['macd'] = macd.macd_diff()
                df['atr'] = talib.volatility.AverageTrueRange(df['high_price'], df['low_price'], df['close_price'], window=14).average_true_range()
                bb = talib.volatility.BollingerBands(df['close_price'], window=20, window_dev=2)
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = df['bb_upper'] - df['bb_lower']
                df['1h_momentum'] = df['close_price'].pct_change(periods=4)
                df['vol_momentum'] = df['volume'].pct_change(periods=96)
                df['amount_sum'] = df['amount'].rolling(window=7 * 96).sum()
                df['4h_momentum'] = df['close_price'].pct_change(periods=16)
                df['7d_momentum'] = df['close_price'].pct_change(periods=7 * 96)

                for col in ['rsi', 'macd', 'atr', 'bb_width', '1h_momentum', 'vol_momentum', 'amount_sum', '4h_momentum', '7d_momentum']:
                    if col in df.columns:
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

                df = df.dropna(subset=['target'])
                df['symbol'] = s

                dfs.append(df)
                valid_symbols.append(s)
                print(f"✅ 加载成功: {s}, shape={df.shape}")
            except Exception as e:
                print(f"[!] 加载 {s} 出错: {e}")
                failed_symbols.append(s)

        print(f"🔍 总计：成功加载 {len(valid_symbols)} 个 symbol，失败 {len(failed_symbols)} 个")
        return valid_symbols, dfs

    def train(self, dfs, valid_symbols):
        full_df = pd.concat(dfs, axis=0).sort_index()
        if full_df.empty:
            print("[!] 错误：没有有效数据用于训练！")
            return None

        if 'symbol' not in full_df.columns:
            raise ValueError("full_df 中缺少 'symbol' 列，请检查数据加载部分是否保留了 symbol 列")

        feature_cols = [
            '4h_momentum', '7d_momentum', 'amount_sum', 'vol_momentum',
            'atr', 'macd', 'bb_width', '1h_momentum'
        ]
        available_features = [f for f in feature_cols if f in full_df.columns]
        X = full_df[available_features]
        y = full_df['target']

        combined_xy = pd.concat([X, y], axis=1)
        combined_xy = combined_xy.replace([np.inf, -np.inf], np.nan).dropna()

        X = combined_xy[available_features]
        y = combined_xy['target']

        print(f"✅ 清理后训练数据: X shape={X.shape}, y shape={y.shape}")

        # 标准化
        X_scaled = StandardScaler().fit_transform(X)

        # 设置 tree_method: GPU 加速 or CPU
        tree_method = 'gpu_hist' if self.use_gpu else 'hist'

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            tree_method=tree_method,  # ✅ GPU 加速：'gpu_hist'，否则 'hist'
            random_state=42
        )
        model.fit(X_scaled, y)

        y_pred = model.predict(X_scaled)

        result_df = combined_xy.copy()
        result_df['y_pred'] = y_pred

        if 'symbol' not in result_df.columns:
            raise ValueError("result_df 中缺少 'symbol' 列，无法构造 id")

        result_df['id'] = result_df.index.strftime("%Y%m%d%H%M%S") + "_" + result_df['symbol']
        result_df = result_df[['id', 'y_pred']]
        result_df.columns = ['id', 'predict_return']
        result_df['predict_return'] = result_df['predict_return'].clip(-1, 1)

        print(f"✅ 构造完成：result_df 包含 {len(result_df)} 条记录")
        return result_df

    def generate_submission(self, predictions_df):
        submission_ids = pd.read_csv(self.submission_id_path)
        if 'id' not in submission_ids.columns:
            raise ValueError("submission_id.csv 必须包含 'id' 列")

        if 'id' not in predictions_df.columns or 'predict_return' not in predictions_df.columns:
            raise ValueError("预测结果 DataFrame 必须包含 'id' 和 'predict_return' 列")

        final_submission = submission_ids.merge(
            predictions_df[['id', 'predict_return']],
            on='id',
            how='left'
        )
        final_submission['predict_return'] = final_submission['predict_return'].fillna(0.0)

        final_submission = final_submission[['id', 'predict_return']]
        final_submission.to_csv(OUTPUT_SUBMIT_PATH, index=False)
        print(f"✅ 提交文件已保存至: {OUTPUT_SUBMIT_PATH}")
# ======================
# 4. 主程序入口
# ======================
if __name__ == "__main__":
    model = OptimizedModel()
    symbols = model.get_all_symbols()
    valid_symbols, dfs = model.load_all_data(symbols)

    if not valid_symbols:
        print("❌ 没有有效数据，退出")
    else:
        print(f"🔍 将使用 {len(valid_symbols)} 个有效 symbol 进行训练")
        result_df = model.train(dfs, valid_symbols)
        if result_df is not None:
            model.generate_submission(result_df)