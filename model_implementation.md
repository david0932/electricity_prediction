# 電力數據預測模型實作

本文檔提供了辦公室電錶數據（每5秒收集一次）的預測模型實作方案，包括數據預處理流程和多種預測模型的實作代碼。

## 1. 數據預處理流程實作

以下是完整的數據預處理流程實作，包括數據清洗、特徵工程、降採樣和標準化等步驟。

```python
# electricity_preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class ElectricityPreprocessor:
    """
    電力數據預處理類，用於處理每5秒收集一次的電錶數據
    """
    
    def __init__(self, config=None):
        """
        初始化預處理器
        
        Parameters:
        -----------
        config : dict, optional
            預處理配置參數
        """
        self.config = config or {
            'missing_threshold': 60,  # 缺失值處理閾值（秒）
            'anomaly_std_threshold': 3.0,  # 異常值標準差閾值
            'smoothing_window': 12,  # 平滑窗口大小（60秒）
            'resampling': {
                'short_term': '1min',  # 短期預測降採樣
                'medium_term': '15min',  # 中期預測降採樣
                'long_term': '1H'  # 長期預測降採樣
            }
        }
        self.scalers = {}
    
    def load_data(self, file_path):
        """
        載入原始電錶數據
        
        Parameters:
        -----------
        file_path : str
            數據文件路徑
            
        Returns:
        --------
        pd.DataFrame
            載入的數據框
        """
        try:
            # 假設數據格式為CSV，包含時間戳和電力讀數
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            df.set_index('timestamp', inplace=True)
            print(f"成功載入數據，共 {len(df)} 筆記錄")
            return df
        except Exception as e:
            print(f"載入數據時出錯: {e}")
            # 如果沒有實際數據，創建模擬數據用於演示
            return self._create_sample_data()
    
    def _create_sample_data(self, days=7):
        """
        創建模擬電錶數據用於演示
        
        Parameters:
        -----------
        days : int
            模擬數據的天數
            
        Returns:
        --------
        pd.DataFrame
            模擬的電錶數據
        """
        # 創建時間索引（每5秒一個點）
        end_time = datetime.now().replace(microsecond=0, second=0, minute=0)
        start_time = end_time - timedelta(days=days)
        time_index = pd.date_range(start=start_time, end=end_time, freq='5S')
        
        # 創建基本負載模式
        hourly_pattern = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.5 + 0.5  # 日內模式
        weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6])  # 週內模式
        
        # 生成用電量數據
        power_values = []
        for t in time_index:
            hour_val = hourly_pattern[t.hour]
            day_val = weekly_pattern[t.weekday()]
            
            # 基本負載
            base_load = 50 + 30 * hour_val * day_val
            
            # 隨機波動
            noise = np.random.normal(0, 5)
            
            # 總負載
            total_load = base_load + noise
            
            power_values.append(max(0, total_load))
        
        # 創建數據框
        df = pd.DataFrame({
            'power': power_values,
            'voltage': np.random.normal(220, 5, len(time_index)),
            'current': np.array(power_values) / 220 * (1 + np.random.normal(0, 0.05, len(time_index)))
        }, index=time_index)
        
        # 添加累計用電量
        df['cumulative_power'] = df['power'].cumsum() * 5 / 3600  # kWh
        
        # 隨機添加一些缺失值和異常值
        mask_missing = np.random.random(len(df)) < 0.01
        df.loc[mask_missing, 'power'] = np.nan
        
        mask_anomaly = np.random.random(len(df)) < 0.005
        df.loc[mask_anomaly, 'power'] = df.loc[mask_anomaly, 'power'] * np.random.uniform(2, 5)
        
        print(f"已創建模擬數據，共 {len(df)} 筆記錄，時間範圍: {df.index.min()} 到 {df.index.max()}")
        return df
    
    def clean_data(self, df):
        """
        數據清洗：處理缺失值、異常值和噪聲
        
        Parameters:
        -----------
        df : pd.DataFrame
            原始數據框
            
        Returns:
        --------
        pd.DataFrame
            清洗後的數據框
        """
        print("開始數據清洗...")
        df_cleaned = df.copy()
        
        # 1. 確保時間索引連續
        full_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='5S'
        )
        df_cleaned = df_cleaned.reindex(full_index)
        
        # 2. 處理缺失值
        missing_count = df_cleaned['power'].isna().sum()
        print(f"發現 {missing_count} 個缺失值 ({missing_count/len(df_cleaned)*100:.2f}%)")
        
        # 對於短時間缺失使用插值
        df_cleaned['power'] = self._fill_missing_values(df_cleaned['power'])
        
        # 3. 處理異常值
        df_cleaned['power'] = self._handle_anomalies(df_cleaned['power'])
        
        # 4. 噪聲過濾
        df_cleaned['power_smooth'] = self._smooth_data(df_cleaned['power'])
        
        # 5. 數據一致性檢查（如果有電壓和電流）
        if 'voltage' in df_cleaned.columns and 'current' in df_cleaned.columns:
            df_cleaned = self._check_consistency(df_cleaned)
        
        print("數據清洗完成")
        return df_cleaned
    
    def _fill_missing_values(self, series):
        """
        填充缺失值
        
        Parameters:
        -----------
        series : pd.Series
            包含缺失值的數據序列
            
        Returns:
        --------
        pd.Series
            填充後的數據序列
        """
        # 計算每個缺失值的持續時間
        missing_mask = series.isna()
        
        if not missing_mask.any():
            return series
        
        # 對於短時間缺失使用插值
        filled_series = series.interpolate(method='linear')
        
        # 對於長時間缺失（超過閾值）使用前一天同時段數據
        long_gaps = self._find_long_gaps(series, self.config['missing_threshold'])
        
        if long_gaps:
            print(f"發現 {len(long_gaps)} 個長時間缺失區間")
            for start, end in long_gaps:
                # 使用前一天同時段數據填充
                day_offset = pd.Timedelta(days=1)
                for idx in series.loc[start:end].index:
                    if idx - day_offset in series.index and not pd.isna(series[idx - day_offset]):
                        filled_series[idx] = series[idx - day_offset]
        
        return filled_series
    
    def _find_long_gaps(self, series, threshold_seconds):
        """
        找出長時間缺失區間
        
        Parameters:
        -----------
        series : pd.Series
            數據序列
        threshold_seconds : int
            缺失時間閾值（秒）
            
        Returns:
        --------
        list
            長時間缺失區間的起止索引列表
        """
        missing_mask = series.isna()
        if not missing_mask.any():
            return []
        
        # 找出缺失值的起止點
        missing_indices = np.where(missing_mask)[0]
        gaps = []
        gap_start = missing_indices[0]
        
        for i in range(1, len(missing_indices)):
            if missing_indices[i] > missing_indices[i-1] + 1:
                gap_end = missing_indices[i-1]
                # 計算缺失持續時間
                gap_duration = (series.index[gap_end] - series.index[gap_start]).total_seconds()
                if gap_duration >= threshold_seconds:
                    gaps.append((series.index[gap_start], series.index[gap_end]))
                gap_start = missing_indices[i]
        
        # 處理最後一個缺失區間
        gap_end = missing_indices[-1]
        gap_duration = (series.index[gap_end] - series.index[gap_start]).total_seconds()
        if gap_duration >= threshold_seconds:
            gaps.append((series.index[gap_start], series.index[gap_end]))
        
        return gaps
    
    def _handle_anomalies(self, series):
        """
        處理異常值
        
        Parameters:
        -----------
        series : pd.Series
            數據序列
            
        Returns:
        --------
        pd.Series
            處理後的數據序列
        """
        # 使用移動窗口Z-score檢測異常
        window_size = 60*12  # 1小時窗口
        
        # 計算移動平均和標準差
        rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
        
        # 計算Z-score
        z_scores = (series - rolling_mean) / rolling_std
        
        # 識別異常值
        threshold = self.config['anomaly_std_threshold']
        anomalies = (z_scores.abs() > threshold)
        anomaly_count = anomalies.sum()
        
        print(f"發現 {anomaly_count} 個異常值 ({anomaly_count/len(series)*100:.2f}%)")
        
        # 替換異常值
        if anomaly_count > 0:
            # 使用移動中位數替換異常值
            median_window = 5  # 25秒窗口
            rolling_median = series.rolling(window=median_window, center=True, min_periods=1).median()
            series_cleaned = series.copy()
            series_cleaned[anomalies] = rolling_median[anomalies]
            return series_cleaned
        
        return series
    
    def _smooth_data(self, series):
        """
        平滑數據以減少噪聲
        
        Parameters:
        -----------
        series : pd.Series
            數據序列
            
        Returns:
        --------
        pd.Series
            平滑後的數據序列
        """
        # 使用移動平均進行平滑
        window_size = self.config['smoothing_window']
        return series.rolling(window=window_size, center=True, min_periods=1).mean()
    
    def _check_consistency(self, df):
        """
        檢查數據一致性（如功率=電壓×電流）
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
            
        Returns:
        --------
        pd.DataFrame
            檢查後的數據框
        """
        # 計算理論功率
        df['power_calculated'] = df['voltage'] * df['current']
        
        # 檢查實際功率與理論功率的差異
        power_diff = (df['power'] - df['power_calculated']).abs() / df['power_calculated']
        inconsistent = power_diff > 0.2  # 允許20%的誤差
        
        inconsistent_count = inconsistent.sum()
        print(f"發現 {inconsistent_count} 個不一致數據點 ({inconsistent_count/len(df)*100:.2f}%)")
        
        # 修正不一致數據
        if inconsistent_count > 0:
            df.loc[inconsistent, 'power'] = df.loc[inconsistent, 'power_calculated']
        
        return df
    
    def engineer_features(self, df):
        """
        特徵工程：創建時間特徵、滯後特徵和統計特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            清洗後的數據框
            
        Returns:
        --------
        pd.DataFrame
            添加特徵後的數據框
        """
        print("開始特徵工程...")
        df_featured = df.copy()
        
        # 1. 時間特徵
        df_featured = self._create_time_features(df_featured)
        
        # 2. 滯後特徵
        df_featured = self._create_lag_features(df_featured)
        
        # 3. 統計特徵
        df_featured = self._create_statistical_features(df_featured)
        
        # 4. 外部特徵（如果有）
        # 這裡可以添加外部數據，如天氣數據
        
        print("特徵工程完成，創建了 {} 個特徵".format(len(df_featured.columns) - len(df.columns)))
        return df_featured
    
    def _create_time_features(self, df):
        """
        創建時間特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
            
        Returns:
        --------
        pd.DataFrame
            添加時間特徵後的數據框
        """
        # 提取基本時間特徵
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        # 工作日標記
        df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
        
        # 工作時間標記
        df['is_work_hour'] = ((df['hour'] >= 9) & (df['hour'] < 18) & (df['is_weekday'] == 1)).astype(int)
        
        # 循環時間特徵
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_lag_features(self, df, target_col='power'):
        """
        創建滯後特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
        target_col : str
            目標列名
            
        Returns:
        --------
        pd.DataFrame
            添加滯後特徵後的數據框
        """
        # 短期滯後特徵
        short_lags = [12, 60, 180, 360]  # 1分鐘, 5分鐘, 15分鐘, 30分鐘
        
        # 長期滯後特徵
        day_lag = 60*60*24 // 5  # 一天的點數
        week_lag = day_lag * 7  # 一週的點數
        
        # 創建滯後特徵
        for lag in short_lags:
            df[f'{target_col}_lag_{lag*5}s'] = df[target_col].shift(lag)
        
        # 添加日滯後和週滯後
        if len(df) > day_lag:
            df[f'{target_col}_lag_1d'] = df[target_col].shift(day_lag)
        
        if len(df) > week_lag:
            df[f'{target_col}_lag_1w'] = df[target_col].shift(week_lag)
        
        return df
    
    def _create_statistical_features(self, df, target_col='power'):
        """
        創建統計特徵
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
        target_col : str
            目標列名
            
        Returns:
        --------
        pd.DataFrame
            添加統計特徵後的數據框
        """
        # 定義窗口大小（以點數計）
        windows = {
            '5min': 60,      # 5分鐘 = 60點
            '15min': 180,    # 15分鐘 = 180點
            '1hour': 720,    # 1小時 = 720點
        }
        
        # 為每個窗口創建統計特徵
        for window_name, window_size in windows.items():
            # 移動平均
            df[f'{target_col}_mean_{window_name}'] = df[target_col].rolling(
                window=window_size, min_periods=1).mean()
            
            # 移動標準差
            df[f'{target_col}_std_{window_name}'] = df[target_col].rolling(
                window=window_size, min_periods=1).std()
            
            # 移動最大值和最小值
            df[f'{target_col}_max_{window_name}'] = df[target_col].rolling(
                window=window_size, min_periods=1).max()
            df[f'{target_col}_min_{window_name}'] = df[target_col].rolling(
                window=window_size, min_periods=1).min()
        
        # 趨勢特徵：簡單移動平均差異
        df[f'{target_col}_trend_5min'] = df[f'{target_col}_mean_5min'] - df[f'{target_col}_mean_5min'].shift(60)
        df[f'{target_col}_trend_1hour'] = df[f'{target_col}_mean_1hour'] - df[f'{target_col}_mean_1hour'].shift(720)
        
        return df
    
    def resample_data(self, df, freq=None):
        """
        數據降採樣
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
        freq : str, optional
            降採樣頻率，如'1min', '15min', '1H'
            
        Returns:
        --------
        pd.DataFrame
            降採樣後的數據框
        """
        if freq is None:
            return df
        
        print(f"將數據從5秒降採樣至{freq}...")
        
        # 選擇數值列進行聚合
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 定義聚合方法
        agg_dict = {}
        for col in numeric_cols:
            if col.startswith('is_') or 'sin' in col or 'cos' in col:
                # 對於二元特徵和循環特徵，使用眾數
                agg_dict[col] = lambda x: x.mode()[0] if not x.empty else np.nan
            elif 'lag' in col or 'trend' in col:
                # 對於滯後特徵和趨勢特徵，使用最後一個值
                agg_dict[col] = 'last'
            else:
                # 對於其他數值特徵，使用平均值
                agg_dict[col] = 'mean'
        
        # 執行降採樣
        df_resampled = df.resample(freq).agg(agg_dict)
        
        print(f"降採樣完成，從 {len(df)} 筆減少到 {len(df_resampled)} 筆")
        return df_resampled
    
    def normalize_data(self, df, method='standard', target_col='power'):
        """
        數據標準化
        
        Parameters:
        -----------
        df : pd.DataFrame
            數據框
        method : str
            標準化方法，'standard', 'minmax', 或 'robust'
        target_col : str
            目標列名
            
        Returns:
        --------
        pd.DataFrame
            標準化後的數據框
        """
        print(f"使用 {method} 方法標準化數據...")
        df_norm = df.copy()
        
        # 選擇要標準化的列（數值型特徵）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 排除不需要標準化的列
        exclude_cols = ['hour', 'day_of_week', 'month', 'is_weekday', 'is_work_hour']
        norm_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 選擇標準化方法
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的標準化方法: {method}")
        
        # 對每列分別進行標準化
        for col in norm_cols:
            # 保存目標列的原始值（用於還原預測結果）
            if col == target_col:
                self.scalers[col] = scaler.fit(df[[col]])
            
            # 標準化
            df_norm[col] = scaler.fit_transform(df[[col]])
        
        print("數據標準化完成")
        return df_norm
    
    def prepare_train_test_data(self, df, target_col='power', test_size=0.2):
        """
        準備訓練和測試數據
        
        Parameters:
        -----------
        df : pd.DataFrame
            處理後的數據框
        target_col : str
            目標列名
        test_size : float
            測試集比例
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("準備訓練和測試數據...")
        
        # 移除包含NaN的行
        df_clean = df.dropna()
        
        # 確定分割點
        split_idx = int(len(df_clean) * (1 - test_size))
        
        # 分割數據
        train_df = df_clean.iloc[:split_idx]
        test_df = df_clean.iloc[split_idx:]
        
        # 選擇特徵和目標
        feature_cols = [col for col in df_clean.columns if col != target_col]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        print(f"訓練集: {len(X_train)} 筆, 測試集: {len(X_test)} 筆")
        return X_train, X_test, y_train, y_test
    
    def process_pipeline(self, file_path=None, target_col='power', freq=None, test_size=0.2):
        """
        完整的數據預處理流程
        
        Parameters:
        -----------
        file_path : str, optional
            數據文件路徑
        target_col : str
            目標列名
        freq : str, optional
            降採樣頻率
        test_size : float
            測試集比例
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, preprocessor)
        """
        # 1. 載入數據
        df = self.load_data(file_path)
        
        # 2. 數據清洗
        df_cleaned = self.clean_data(df)
        
        # 3. 特徵工程
        df_featured = self.engineer_features(df_cleaned)
        
        # 4. 數據降採樣（如果需要）
        if freq:
            df_resampled = self.resample_data(df_featured, freq)
        else:
            df_resampled = df_featured
        
        # 5. 數據標準化
        df_normalized = self.normalize_data(df_resampled, method='standard', target_col=target_col)
        
        # 6. 準備訓練和測試數據
        X_train, X_test, y_train, y_test = self.prepare_train_test_data(
            df_normalized, target_col=target_col, test_size=test_size
        )
        
        return X_train, X_test, y_train, y_test, self
    
    def inverse_transform(self, predictions, col='power'):
        """
        將標準化的預測結果轉換回原始尺度
        
        Parameters:
        -----------
        predictions : array-like
            標準化的預測結果
        col : str
            目標列名
            
        Returns:
        --------
        array-like
            原始尺度的預測結果
        """
        if col in self.scalers:
            return self.scalers[col].inverse_transform(predictions.reshape(-1, 1)).flatten()
        return predictions
```

## 2. 基準模型實作

以下是基準模型的實作，包括ARIMA、指數平滑和簡單機器學習模型。

```python
# electricity_baseline_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BaselineModels:
    """
    電力數據預測的基準模型
    """
    
    def __init__(self):
        """
        初始化基準模型類
        """
        self.models = {}
        self.results = {}
    
    def fit_arima(self, train_data, order=(1, 1, 1)):
        """
        擬合ARIMA模型
        
        Parameters:
        -----------
        train_data : pd.Series
            訓練數據，時間序列
        order : tuple
            ARIMA模型的階數 (p, d, q)
            
        Returns:
        --------
        self
        """
        print(f"擬合ARIMA模型，階數={order}...")
        
        # 確保數據是時間序列
        if not isinstance(train_data, pd.Series):
            train_data = pd.Series(train_data)
        
        # 擬合模型
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        
        # 保存模型
        self.models['arima'] = fitted_model
        print("ARIMA模型擬合完成")
        
        return self
    
    def predict_arima(self, steps=24):
        """
        使用ARIMA模型進行預測
        
        Parameters:
        -----------
        steps : int
            預測步數
            
        Returns:
        --------
        pd.Series
            預測結果
        """
        if 'arima' not in self.models:
            raise ValueError("ARIMA模型尚未擬合")
        
        # 進行預測
        forecast = self.models['arima'].forecast(steps=steps)
        
        # 保存結果
        self.results['arima'] = forecast
        
        return forecast
    
    def fit_exponential_smoothing(self, train_data, seasonal_periods=24, trend=None, seasonal=None):
        """
        擬合指數平滑模型
        
        Parameters:
        -----------
        train_data : pd.Series
            訓練數據，時間序列
        seasonal_periods : int
            季節性週期長度
        trend : str, optional
            趨勢類型，'add'或'mul'
        seasonal : str, optional
            季節性類型，'add'或'mul'
            
        Returns:
        --------
        self
        """
        print(f"擬合指數平滑模型，季節性週期={seasonal_periods}...")
        
        # 確保數據是時間序列
        if not isinstance(train_data, pd.Series):
            train_data = pd.Series(train_data)
        
        # 擬合模型
        model = ExponentialSmoothing(
            train_data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        fitted_model = model.fit()
        
        # 保存模型
        self.models['exp_smoothing'] = fitted_model
        print("指數平滑模型擬合完成")
        
        return self
    
    def predict_exponential_smoothing(self, steps=24):
        """
        使用指數平滑模型進行預測
        
        Parameters:
        -----------
        steps : int
            預測步數
            
        Returns:
        --------
        pd.Series
            預測結果
        """
        if 'exp_smoothing' not in self.models:
            raise ValueError("指數平滑模型尚未擬合")
        
        # 進行預測
        forecast = self.models['exp_smoothing'].forecast(steps=steps)
        
        # 保存結果
        self.results['exp_smoothing'] = forecast
        
        return forecast
    
    def fit_linear_regression(self, X_train, y_train):
        """
        擬合線性迴歸模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        y_train : pd.Series
            訓練目標
            
        Returns:
        --------
        self
        """
        print("擬合線性迴歸模型...")
        
        # 擬合模型
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 保存模型
        self.models['linear_regression'] = model
        print("線性迴歸模型擬合完成")
        
        return self
    
    def predict_linear_regression(self, X_test):
        """
        使用線性迴歸模型進行預測
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            測試特徵
            
        Returns:
        --------
        np.array
            預測結果
        """
        if 'linear_regression' not in self.models:
            raise ValueError("線性迴歸模型尚未擬合")
        
        # 進行預測
        predictions = self.models['linear_regression'].predict(X_test)
        
        # 保存結果
        self.results['linear_regression'] = predictions
        
        return predictions
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        評估模型性能
        
        Parameters:
        -----------
        y_true : array-like
            真實值
        y_pred : array-like
            預測值
        model_name : str
            模型名稱
            
        Returns:
        --------
        dict
            性能指標
        """
        # 計算性能指標
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 計算MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 打印結果
        print(f"{model_name} 模型評估結果:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 返回指標字典
        metrics = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, model_name, figsize=(12, 6)):
        """
        繪製預測結果
        
        Parameters:
        -----------
        y_true : array-like
            真實值
        y_pred : array-like
            預測值
        model_name : str
            模型名稱
        figsize : tuple
            圖形大小
            
        Returns:
        --------
        matplotlib.figure.Figure
            圖形對象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 繪製真實值和預測值
        if isinstance(y_true, pd.Series):
            ax.plot(y_true.index, y_true.values, label='實際值', color='blue')
            if isinstance(y_pred, pd.Series):
                ax.plot(y_pred.index, y_pred.values, label='預測值', color='red')
            else:
                ax.plot(y_true.index, y_pred, label='預測值', color='red')
        else:
            ax.plot(y_true, label='實際值', color='blue')
            ax.plot(y_pred, label='預測值', color='red')
        
        # 添加標題和標籤
        ax.set_title(f'{model_name} 預測結果')
        ax.set_xlabel('時間')
        ax.set_ylabel('用電量')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
```

## 3. 進階模型實作

以下是進階模型的實作，包括XGBoost、LSTM和Prophet模型。

```python
# electricity_advanced_models.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入Prophet（可能需要安裝）
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    print("Prophet 未安裝，將無法使用 Prophet 模型")

class AdvancedModels:
    """
    電力數據預測的進階模型
    """
    
    def __init__(self):
        """
        初始化進階模型類
        """
        self.models = {}
        self.results = {}
        self.history = {}
    
    def fit_xgboost(self, X_train, y_train, params=None):
        """
        擬合XGBoost模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        y_train : pd.Series
            訓練目標
        params : dict, optional
            XGBoost參數
            
        Returns:
        --------
        self
        """
        print("擬合XGBoost模型...")
        
        # 默認參數
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # 使用提供的參數或默認參數
        model_params = params or default_params
        
        # 擬合模型
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        
        # 保存模型
        self.models['xgboost'] = model
        print("XGBoost模型擬合完成")
        
        return self
    
    def predict_xgboost(self, X_test):
        """
        使用XGBoost模型進行預測
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            測試特徵
            
        Returns:
        --------
        np.array
            預測結果
        """
        if 'xgboost' not in self.models:
            raise ValueError("XGBoost模型尚未擬合")
        
        # 進行預測
        predictions = self.models['xgboost'].predict(X_test)
        
        # 保存結果
        self.results['xgboost'] = predictions
        
        return predictions
    
    def fit_lstm(self, X_train, y_train, lookback=24, epochs=50, batch_size=32, verbose=1):
        """
        擬合LSTM模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        y_train : pd.Series
            訓練目標
        lookback : int
            回看窗口大小
        epochs : int
            訓練輪數
        batch_size : int
            批次大小
        verbose : int
            詳細程度
            
        Returns:
        --------
        self
        """
        print("擬合LSTM模型...")
        
        # 準備LSTM輸入數據
        X_lstm, y_lstm = self._prepare_lstm_data(X_train, y_train, lookback)
        
        # 定義模型
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        # 編譯模型
        model.compile(optimizer='adam', loss='mse')
        
        # 早停
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 擬合模型
        history = model.fit(
            X_lstm, y_lstm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        # 保存模型和訓練歷史
        self.models['lstm'] = model
        self.history['lstm'] = history.history
        
        print("LSTM模型擬合完成")
        return self
    
    def _prepare_lstm_data(self, X, y, lookback):
        """
        準備LSTM輸入數據
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徵
        y : pd.Series
            目標
        lookback : int
            回看窗口大小
            
        Returns:
        --------
        tuple
            (X_lstm, y_lstm)
        """
        # 轉換為numpy數組
        X_values = X.values
        y_values = y.values
        
        # 創建序列
        X_lstm = []
        y_lstm = []
        
        for i in range(len(X_values) - lookback):
            X_lstm.append(X_values[i:i+lookback])
            y_lstm.append(y_values[i+lookback])
        
        return np.array(X_lstm), np.array(y_lstm)
    
    def predict_lstm(self, X_test, lookback=24):
        """
        使用LSTM模型進行預測
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            測試特徵
        lookback : int
            回看窗口大小
            
        Returns:
        --------
        np.array
            預測結果
        """
        if 'lstm' not in self.models:
            raise ValueError("LSTM模型尚未擬合")
        
        # 準備測試數據
        X_test_values = X_test.values
        
        # 創建預測序列
        X_test_lstm = []
        for i in range(len(X_test_values) - lookback):
            X_test_lstm.append(X_test_values[i:i+lookback])
        
        X_test_lstm = np.array(X_test_lstm)
        
        # 進行預測
        predictions = self.models['lstm'].predict(X_test_lstm)
        
        # 保存結果
        self.results['lstm'] = predictions.flatten()
        
        return predictions.flatten()
    
    def fit_prophet(self, train_data, date_col='ds', target_col='y', seasonality_mode='additive'):
        """
        擬合Prophet模型
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            訓練數據
        date_col : str
            日期列名
        target_col : str
            目標列名
        seasonality_mode : str
            季節性模式，'additive'或'multiplicative'
            
        Returns:
        --------
        self
        """
        if not prophet_available:
            raise ImportError("Prophet 未安裝，無法使用此功能")
        
        print("擬合Prophet模型...")
        
        # 準備Prophet輸入數據
        prophet_data = pd.DataFrame({
            'ds': train_data.index if date_col == 'ds' else train_data[date_col],
            'y': train_data[target_col]
        })
        
        # 創建模型
        model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # 擬合模型
        model.fit(prophet_data)
        
        # 保存模型
        self.models['prophet'] = model
        print("Prophet模型擬合完成")
        
        return self
    
    def predict_prophet(self, periods=24, freq='H'):
        """
        使用Prophet模型進行預測
        
        Parameters:
        -----------
        periods : int
            預測期數
        freq : str
            頻率，如'H'（小時）、'D'（天）
            
        Returns:
        --------
        pd.DataFrame
            預測結果
        """
        if not prophet_available:
            raise ImportError("Prophet 未安裝，無法使用此功能")
        
        if 'prophet' not in self.models:
            raise ValueError("Prophet模型尚未擬合")
        
        # 創建未來數據框
        future = self.models['prophet'].make_future_dataframe(
            periods=periods,
            freq=freq
        )
        
        # 進行預測
        forecast = self.models['prophet'].predict(future)
        
        # 保存結果
        self.results['prophet'] = forecast
        
        return forecast
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        評估模型性能
        
        Parameters:
        -----------
        y_true : array-like
            真實值
        y_pred : array-like
            預測值
        model_name : str
            模型名稱
            
        Returns:
        --------
        dict
            性能指標
        """
        # 計算性能指標
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # 計算MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # 打印結果
        print(f"{model_name} 模型評估結果:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 返回指標字典
        metrics = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, model_name, figsize=(12, 6)):
        """
        繪製預測結果
        
        Parameters:
        -----------
        y_true : array-like
            真實值
        y_pred : array-like
            預測值
        model_name : str
            模型名稱
        figsize : tuple
            圖形大小
            
        Returns:
        --------
        matplotlib.figure.Figure
            圖形對象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 繪製真實值和預測值
        if isinstance(y_true, pd.Series):
            ax.plot(y_true.index, y_true.values, label='實際值', color='blue')
            if isinstance(y_pred, pd.Series):
                ax.plot(y_pred.index, y_pred.values, label='預測值', color='red')
            else:
                ax.plot(y_true.index[-len(y_pred):], y_pred, label='預測值', color='red')
        else:
            ax.plot(y_true, label='實際值', color='blue')
            ax.plot(y_pred, label='預測值', color='red')
        
        # 添加標題和標籤
        ax.set_title(f'{model_name} 預測結果')
        ax.set_xlabel('時間')
        ax.set_ylabel('用電量')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model_name='xgboost', figsize=(12, 8)):
        """
        繪製特徵重要性
        
        Parameters:
        -----------
        model_name : str
            模型名稱
        figsize : tuple
            圖形大小
            
        Returns:
        --------
        matplotlib.figure.Figure
            圖形對象
        """
        if model_name not in self.models:
            raise ValueError(f"{model_name} 模型尚未擬合")
        
        if model_name != 'xgboost':
            raise ValueError("目前只支持XGBoost模型的特徵重要性繪製")
        
        # 獲取特徵重要性
        importance = self.models[model_name].feature_importances_
        feature_names = self.models[model_name].feature_names_in_
        
        # 創建特徵重要性數據框
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # 繪製
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance_df['feature'][:20], importance_df['importance'][:20])
        ax.set_title('XGBoost 特徵重要性')
        ax.set_xlabel('重要性')
        
        plt.tight_layout()
        return fig
```

## 4. 模型訓練和預測流程

以下是完整的模型訓練和預測流程實作。

```python
# electricity_prediction_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pickle
from electricity_preprocessing import ElectricityPreprocessor
from electricity_baseline_models import BaselineModels
from electricity_advanced_models import AdvancedModels

class ElectricityPredictionPipeline:
    """
    電力數據預測流程
    """
    
    def __init__(self, config=None):
        """
        初始化預測流程
        
        Parameters:
        -----------
        config : dict, optional
            配置參數
        """
        self.config = config or {
            'data_path': None,
            'target_col': 'power',
            'test_size': 0.2,
            'models': {
                'baseline': ['arima', 'exp_smoothing', 'linear_regression'],
                'advanced': ['xgboost', 'lstm', 'prophet']
            },
            'prediction_horizon': {
                'short_term': '1min',
                'medium_term': '15min',
                'long_term': '1H'
            },
            'output_dir': 'output'
        }
        
        # 創建輸出目錄
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 初始化預處理器和模型
        self.preprocessor = ElectricityPreprocessor()
        self.baseline_models = BaselineModels()
        self.advanced_models = AdvancedModels()
        
        # 存儲結果
        self.results = {}
        self.metrics = {}
    
    def run_pipeline(self, prediction_type='short_term'):
        """
        運行完整預測流程
        
        Parameters:
        -----------
        prediction_type : str
            預測類型，'short_term', 'medium_term', 或 'long_term'
            
        Returns:
        --------
        dict
            預測結果和評估指標
        """
        print(f"開始 {prediction_type} 預測流程...")
        
        # 獲取對應的降採樣頻率
        freq = self.config['prediction_horizon'][prediction_type]
        
        # 1. 數據預處理
        X_train, X_test, y_train, y_test, preprocessor = self.preprocessor.process_pipeline(
            file_path=self.config['data_path'],
            target_col=self.config['target_col'],
            freq=freq,
            test_size=self.config['test_size']
        )
        
        # 保存預處理器
        self.preprocessor = preprocessor
        
        # 2. 訓練和評估基準模型
        self._train_evaluate_baseline_models(X_train, X_test, y_train, y_test, prediction_type)
        
        # 3. 訓練和評估進階模型
        self._train_evaluate_advanced_models(X_train, X_test, y_train, y_test, prediction_type)
        
        # 4. 比較模型性能
        best_model = self._compare_models(prediction_type)
        
        # 5. 保存結果
        self._save_results(prediction_type)
        
        print(f"{prediction_type} 預測流程完成，最佳模型: {best_model}")
        
        return {
            'results': self.results,
            'metrics': self.metrics,
            'best_model': best_model
        }
    
    def _train_evaluate_baseline_models(self, X_train, X_test, y_train, y_test, prediction_type):
        """
        訓練和評估基準模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        X_test : pd.DataFrame
            測試特徵
        y_train : pd.Series
            訓練目標
        y_test : pd.Series
            測試目標
        prediction_type : str
            預測類型
        """
        print("訓練和評估基準模型...")
        
        # 獲取要訓練的基準模型
        baseline_models = self.config['models']['baseline']
        
        # 訓練和評估每個模型
        for model_name in baseline_models:
            if model_name == 'arima':
                # ARIMA模型
                order = (1, 1, 1)  # 可以根據需要調整
                self.baseline_models.fit_arima(y_train, order=order)
                
                # 預測
                forecast_steps = len(y_test)
                predictions = self.baseline_models.predict_arima(steps=forecast_steps)
                
                # 評估
                metrics = self.baseline_models.evaluate_model(
                    y_test.values, predictions.values, 'ARIMA'
                )
                
                # 保存結果
                self.results[f'arima_{prediction_type}'] = predictions
                self.metrics[f'arima_{prediction_type}'] = metrics
                
                # 繪製結果
                fig = self.baseline_models.plot_predictions(
                    y_test, predictions, 'ARIMA'
                )
                fig.savefig(os.path.join(self.config['output_dir'], f'arima_{prediction_type}_predictions.png'))
                plt.close(fig)
            
            elif model_name == 'exp_smoothing':
                # 指數平滑模型
                seasonal_periods = 24  # 可以根據需要調整
                self.baseline_models.fit_exponential_smoothing(
                    y_train, seasonal_periods=seasonal_periods, trend='add', seasonal='add'
                )
                
                # 預測
                forecast_steps = len(y_test)
                predictions = self.baseline_models.predict_exponential_smoothing(steps=forecast_steps)
                
                # 評估
                metrics = self.baseline_models.evaluate_model(
                    y_test.values, predictions.values, '指數平滑'
                )
                
                # 保存結果
                self.results[f'exp_smoothing_{prediction_type}'] = predictions
                self.metrics[f'exp_smoothing_{prediction_type}'] = metrics
                
                # 繪製結果
                fig = self.baseline_models.plot_predictions(
                    y_test, predictions, '指數平滑'
                )
                fig.savefig(os.path.join(self.config['output_dir'], f'exp_smoothing_{prediction_type}_predictions.png'))
                plt.close(fig)
            
            elif model_name == 'linear_regression':
                # 線性迴歸模型
                self.baseline_models.fit_linear_regression(X_train, y_train)
                
                # 預測
                predictions = self.baseline_models.predict_linear_regression(X_test)
                
                # 評估
                metrics = self.baseline_models.evaluate_model(
                    y_test.values, predictions, '線性迴歸'
                )
                
                # 保存結果
                self.results[f'linear_regression_{prediction_type}'] = predictions
                self.metrics[f'linear_regression_{prediction_type}'] = metrics
                
                # 繪製結果
                fig = self.baseline_models.plot_predictions(
                    y_test, predictions, '線性迴歸'
                )
                fig.savefig(os.path.join(self.config['output_dir'], f'linear_regression_{prediction_type}_predictions.png'))
                plt.close(fig)
    
    def _train_evaluate_advanced_models(self, X_train, X_test, y_train, y_test, prediction_type):
        """
        訓練和評估進階模型
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            訓練特徵
        X_test : pd.DataFrame
            測試特徵
        y_train : pd.Series
            訓練目標
        y_test : pd.Series
            測試目標
        prediction_type : str
            預測類型
        """
        print("訓練和評估進階模型...")
        
        # 獲取要訓練的進階模型
        advanced_models = self.config['models']['advanced']
        
        # 訓練和評估每個模型
        for model_name in advanced_models:
            if model_name == 'xgboost':
                # XGBoost模型
                self.advanced_models.fit_xgboost(X_train, y_train)
                
                # 預測
                predictions = self.advanced_models.predict_xgboost(X_test)
                
                # 評估
                metrics = self.advanced_models.evaluate_model(
                    y_test.values, predictions, 'XGBoost'
                )
                
                # 保存結果
                self.results[f'xgboost_{prediction_type}'] = predictions
                self.metrics[f'xgboost_{prediction_type}'] = metrics
                
                # 繪製結果
                fig = self.advanced_models.plot_predictions(
                    y_test, predictions, 'XGBoost'
                )
                fig.savefig(os.path.join(self.config['output_dir'], f'xgboost_{prediction_type}_predictions.png'))
                plt.close(fig)
                
                # 繪製特徵重要性
                fig = self.advanced_models.plot_feature_importance('xgboost')
                fig.savefig(os.path.join(self.config['output_dir'], f'xgboost_{prediction_type}_feature_importance.png'))
                plt.close(fig)
            
            elif model_name == 'lstm':
                # LSTM模型
                lookback = 24  # 可以根據需要調整
                self.advanced_models.fit_lstm(
                    X_train, y_train, lookback=lookback, epochs=50, batch_size=32, verbose=1
                )
                
                # 預測
                predictions = self.advanced_models.predict_lstm(X_test, lookback=lookback)
                
                # 由於LSTM預測結果長度可能與y_test不同，需要調整
                min_len = min(len(y_test) - lookback, len(predictions))
                y_test_lstm = y_test.values[lookback:lookback + min_len]
                predictions_lstm = predictions[:min_len]
                
                # 評估
                metrics = self.advanced_models.evaluate_model(
                    y_test_lstm, predictions_lstm, 'LSTM'
                )
                
                # 保存結果
                self.results[f'lstm_{prediction_type}'] = predictions_lstm
                self.metrics[f'lstm_{prediction_type}'] = metrics
                
                # 繪製結果
                fig = self.advanced_models.plot_predictions(
                    y_test_lstm, predictions_lstm, 'LSTM'
                )
                fig.savefig(os.path.join(self.config['output_dir'], f'lstm_{prediction_type}_predictions.png'))
                plt.close(fig)
            
            elif model_name == 'prophet':
                try:
                    # Prophet模型
                    # 準備Prophet數據
                    train_data = pd.DataFrame({
                        'ds': y_train.index,
                        'y': y_train.values
                    })
                    
                    self.advanced_models.fit_prophet(
                        train_data, date_col='ds', target_col='y', seasonality_mode='additive'
                    )
                    
                    # 預測
                    forecast = self.advanced_models.predict_prophet(
                        periods=len(y_test),
                        freq=self.config['prediction_horizon'][prediction_type]
                    )
                    
                    # 提取預測結果
                    predictions = forecast['yhat'].values[-len(y_test):]
                    
                    # 評估
                    metrics = self.advanced_models.evaluate_model(
                        y_test.values, predictions, 'Prophet'
                    )
                    
                    # 保存結果
                    self.results[f'prophet_{prediction_type}'] = predictions
                    self.metrics[f'prophet_{prediction_type}'] = metrics
                    
                    # 繪製結果
                    fig = self.advanced_models.plot_predictions(
                        y_test, predictions, 'Prophet'
                    )
                    fig.savefig(os.path.join(self.config['output_dir'], f'prophet_{prediction_type}_predictions.png'))
                    plt.close(fig)
                except ImportError:
                    print("Prophet 未安裝，跳過 Prophet 模型")
    
    def _compare_models(self, prediction_type):
        """
        比較模型性能
        
        Parameters:
        -----------
        prediction_type : str
            預測類型
            
        Returns:
        --------
        str
            最佳模型名稱
        """
        print(f"比較 {prediction_type} 預測的模型性能...")
        
        # 收集所有模型的指標
        model_metrics = []
        for model_key, metrics in self.metrics.items():
            if prediction_type in model_key:
                model_metrics.append(metrics)
        
        # 創建比較表
        comparison_df = pd.DataFrame(model_metrics)
        
        # 按RMSE排序
        comparison_df = comparison_df.sort_values('rmse')
        
        # 保存比較表
        comparison_df.to_csv(os.path.join(self.config['output_dir'], f'{prediction_type}_model_comparison.csv'), index=False)
        
        # 繪製比較圖
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 繪製RMSE比較
        ax.bar(comparison_df['model'], comparison_df['rmse'])
        ax.set_title(f'{prediction_type} 預測模型RMSE比較')
        ax.set_xlabel('模型')
        ax.set_ylabel('RMSE')
        ax.grid(axis='y')
        
        # 旋轉x軸標籤
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.config['output_dir'], f'{prediction_type}_model_comparison.png'))
        plt.close(fig)
        
        # 返回最佳模型
        best_model = comparison_df.iloc[0]['model']
        print(f"最佳模型: {best_model}, RMSE: {comparison_df.iloc[0]['rmse']:.4f}")
        
        return best_model
    
    def _save_results(self, prediction_type):
        """
        保存預測結果和模型
        
        Parameters:
        -----------
        prediction_type : str
            預測類型
        """
        print(f"保存 {prediction_type} 預測結果和模型...")
        
        # 保存預測結果
        results_df = pd.DataFrame()
        for model_key, predictions in self.results.items():
            if prediction_type in model_key:
                model_name = model_key.split('_')[0]
                results_df[model_name] = predictions
        
        # 保存結果
        results_df.to_csv(os.path.join(self.config['output_dir'], f'{prediction_type}_predictions.csv'))
        
        # 保存模型
        model_path = os.path.join(self.config['output_dir'], f'{prediction_type}_models.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'baseline_models': self.baseline_models.models,
                'advanced_models': self.advanced_models.models,
                'preprocessor': self.preprocessor
            }, f)
        
        print(f"結果和模型已保存至 {self.config['output_dir']} 目錄")
    
    def make_future_predictions(self, prediction_type='short_term', model_name=None, periods=24):
        """
        進行未來預測
        
        Parameters:
        -----------
        prediction_type : str
            預測類型，'short_term', 'medium_term', 或 'long_term'
        model_name : str, optional
            模型名稱，如果為None則使用最佳模型
        periods : int
            預測期數
            
        Returns:
        --------
        pd.DataFrame
            預測結果
        """
        print(f"使用 {model_name or '最佳模型'} 進行 {prediction_type} 未來預測...")
        
        # 如果未指定模型，使用最佳模型
        if model_name is None:
            # 找出最佳模型
            best_metrics = float('inf')
            for model_key, metrics in self.metrics.items():
                if prediction_type in model_key and metrics['rmse'] < best_metrics:
                    best_metrics = metrics['rmse']
                    model_name = model_key.split('_')[0]
        
        # 獲取對應的頻率
        freq = self.config['prediction_horizon'][prediction_type]
        
        # 創建未來時間索引
        last_date = datetime.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq=freq)
        
        # 根據模型類型進行預測
        if model_name in ['arima', 'exp_smoothing']:
            # 統計模型
            if model_name == 'arima':
                predictions = self.baseline_models.predict_arima(steps=periods)
            else:
                predictions = self.baseline_models.predict_exponential_smoothing(steps=periods)
            
            # 創建預測數據框
            future_df = pd.DataFrame({
                'timestamp': future_dates,
                'prediction': predictions
            })
        
        elif model_name in ['linear_regression', 'xgboost']:
            # 需要特徵的模型
            # 這裡需要為未來時間點創建特徵
            # 簡化處理：使用最後一個時間點的特徵
            if model_name == 'linear_regression':
                # 使用最後一個時間點的特徵進行預測
                last_features = X_test.iloc[-1:].copy()
                future_predictions = []
                
                for i in range(periods):
                    # 預測下一個時間點
                    next_pred = self.baseline_models.predict_linear_regression(last_features)[0]
                    future_predictions.append(next_pred)
                    
                    # 更新特徵（這裡是簡化處理）
                    # 實際應用中需要更新時間特徵和滯後特徵
                    last_features[f'{self.config["target_col"]}_lag_5s'] = next_pred
            
            elif model_name == 'xgboost':
                # 使用最後一個時間點的特徵進行預測
                last_features = X_test.iloc[-1:].copy()
                future_predictions = []
                
                for i in range(periods):
                    # 預測下一個時間點
                    next_pred = self.advanced_models.predict_xgboost(last_features)[0]
                    future_predictions.append(next_pred)
                    
                    # 更新特徵（這裡是簡化處理）
                    # 實際應用中需要更新時間特徵和滯後特徵
                    last_features[f'{self.config["target_col"]}_lag_5s'] = next_pred
            
            # 創建預測數據框
            future_df = pd.DataFrame({
                'timestamp': future_dates,
                'prediction': future_predictions
            })
        
        elif model_name == 'prophet':
            # Prophet模型
            forecast = self.advanced_models.predict_prophet(periods=periods, freq=freq)
            
            # 創建預測數據框
            future_df = pd.DataFrame({
                'timestamp': forecast['ds'].values[-periods:],
                'prediction': forecast['yhat'].values[-periods:]
            })
        
        elif model_name == 'lstm':
            # LSTM模型
            # 使用最後lookback個時間點的特徵進行預測
            lookback = 24  # 與訓練時相同
            last_features = X_test.iloc[-lookback:].copy()
            future_predictions = []
            
            for i in range(periods):
                # 預測下一個時間點
                next_pred = self.advanced_models.predict_lstm(
                    pd.concat([last_features.iloc[1:], pd.DataFrame([last_features.iloc[-1]])]),
                    lookback=lookback
                )[0]
                future_predictions.append(next_pred)
                
                # 更新特徵（這裡是簡化處理）
                # 實際應用中需要更新時間特徵和滯後特徵
                last_features = pd.concat([last_features.iloc[1:], pd.DataFrame([last_features.iloc[-1]])])
                last_features.iloc[-1, last_features.columns.get_loc(f'{self.config["target_col"]}_lag_5s')] = next_pred
            
            # 創建預測數據框
            future_df = pd.DataFrame({
                'timestamp': future_dates,
                'prediction': future_predictions
            })
        
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 將標準化的預測值轉換回原始尺度
        if hasattr(self.preprocessor, 'inverse_transform'):
            future_df['prediction'] = self.preprocessor.inverse_transform(
                future_df['prediction'].values, col=self.config['target_col']
            )
        
        # 保存未來預測結果
        future_df.to_csv(os.path.join(
            self.config['output_dir'],
            f'{prediction_type}_{model_name}_future_predictions.csv'
        ), index=False)
        
        # 繪製未來預測圖
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_df['timestamp'], future_df['prediction'], marker='o', linestyle='-')
        ax.set_title(f'{model_name} {prediction_type} 未來預測')
        ax.set_xlabel('時間')
        ax.set_ylabel('預測用電量')
        ax.grid(True)
        
        # 格式化x軸日期
        fig.autofmt_xdate()
        
        plt.tight_layout()
        fig.savefig(os.path.join(
            self.config['output_dir'],
            f'{prediction_type}_{model_name}_future_predictions.png'
        ))
        plt.close(fig)
        
        print(f"未來預測完成，結果已保存")
        return future_df
```

## 5. 使用範例

以下是如何使用上述實作的範例。

```python
# electricity_prediction_example.py

from electricity_prediction_pipeline import ElectricityPredictionPipeline

def main():
    """
    電力預測範例主函數
    """
    # 配置參數
    config = {
        'data_path': None,  # 使用模擬數據
        'target_col': 'power',
        'test_size': 0.2,
        'models': {
            'baseline': ['arima', 'exp_smoothing', 'linear_regression'],
            'advanced': ['xgboost']  # 可以添加'lstm', 'prophet'
        },
        'prediction_horizon': {
            'short_term': '1min',
            'medium_term': '15min',
            'long_term': '1H'
        },
        'output_dir': 'electricity_prediction/output'
    }
    
    # 創建預測流程
    pipeline = ElectricityPredictionPipeline(config)
    
    # 運行短期預測
    short_term_results = pipeline.run_pipeline(prediction_type='short_term')
    
    # 運行中期預測
    medium_term_results = pipeline.run_pipeline(prediction_type='medium_term')
    
    # 運行長期預測
    long_term_results = pipeline.run_pipeline(prediction_type='long_term')
    
    # 進行未來預測
    short_term_future = pipeline.make_future_predictions(
        prediction_type='short_term',
        model_name=short_term_results['best_model'],
        periods=24
    )
    
    medium_term_future = pipeline.make_future_predictions(
        prediction_type='medium_term',
        model_name=medium_term_results['best_model'],
        periods=24
    )
    
    long_term_future = pipeline.make_future_predictions(
        prediction_type='long_term',
        model_name=long_term_results['best_model'],
        periods=24
    )
    
    print("電力預測範例完成")

if __name__ == "__main__":
    main()
```

這個完整的實作包括了數據預處理、基準模型、進階模型和預測流程，可以處理每5秒收集一次的高頻電力數據，並進行短期、中期和長期的用電預測。
