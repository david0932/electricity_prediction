# 電力數據預測可視化儀表板設計

本文檔詳細說明了辦公室電錶數據（每5秒收集一次）的可視化儀表板設計，包括用電數據可視化、預測結果可視化、性能指標可視化和互動式儀表板實作。

## 1. 用電數據可視化設計

### 1.1 即時用電監控儀表板

#### 1.1.1 即時用電量儀表盤

```python
import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_gauge_chart(current_value, min_value, max_value, threshold_value, title):
    """
    創建即時用電量儀表盤
    
    Parameters:
    -----------
    current_value : float
        當前用電量
    min_value : float
        最小用電量
    max_value : float
        最大用電量
    threshold_value : float
        閾值（契約容量）
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        儀表盤圖形
    """
    # 計算用電量百分比
    percentage = (current_value - min_value) / (max_value - min_value) * 100
    
    # 設定顏色區間
    if current_value < threshold_value * 0.8:
        color = "green"
    elif current_value < threshold_value:
        color = "yellow"
    else:
        color = "red"
    
    # 創建儀表盤
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_value, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_value, threshold_value * 0.8], 'color': "lightgreen"},
                {'range': [threshold_value * 0.8, threshold_value], 'color': "lightyellow"},
                {'range': [threshold_value, max_value], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_value
            }
        }
    ))
    
    # 設定佈局
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
```

#### 1.1.2 即時用電趨勢圖

```python
def create_realtime_trend_chart(data, window_hours=1, title="即時用電趨勢"):
    """
    創建即時用電趨勢圖
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含時間戳和用電量的數據框
    window_hours : int
        顯示窗口大小（小時）
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        趨勢圖
    """
    # 過濾最近數據
    now = datetime.now()
    start_time = now - timedelta(hours=window_hours)
    recent_data = data[data['timestamp'] >= start_time]
    
    # 創建趨勢圖
    fig = go.Figure()
    
    # 添加用電量曲線
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['power'],
        mode='lines',
        name='實際用電量',
        line=dict(color='blue', width=2)
    ))
    
    # 添加移動平均線
    window_size = int(len(recent_data) / 20) if len(recent_data) > 20 else 1
    recent_data['power_ma'] = recent_data['power'].rolling(window=window_size).mean()
    fig.add_trace(go.Scatter(
        x=recent_data['timestamp'],
        y=recent_data['power_ma'],
        mode='lines',
        name='移動平均',
        line=dict(color='red', width=1, dash='dot')
    ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

#### 1.1.3 用電分佈熱圖

```python
def create_heatmap(data, title="每日用電分佈熱圖"):
    """
    創建用電分佈熱圖
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含時間戳和用電量的數據框
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        熱圖
    """
    # 提取小時和星期幾
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    # 計算每小時每天的平均用電量
    heatmap_data = data.groupby(['day_of_week', 'hour'])['power'].mean().reset_index()
    
    # 創建樞紐表
    pivot_table = heatmap_data.pivot(index='day_of_week', columns='hour', values='power')
    
    # 創建熱圖
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=['週一', '週二', '週三', '週四', '週五', '週六', '週日'],
        colorscale='Viridis',
        colorbar=dict(title="用電量 (kW)")
    ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="小時",
        yaxis_title="星期",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
```

### 1.2 歷史用電分析儀表板

#### 1.2.1 日/週/月用電量比較圖

```python
def create_comparison_chart(data, comparison_type='daily', title=None):
    """
    創建日/週/月用電量比較圖
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含時間戳和用電量的數據框
    comparison_type : str
        比較類型，'daily', 'weekly', 或 'monthly'
    title : str, optional
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        比較圖
    """
    # 根據比較類型設定重採樣頻率和標題
    if comparison_type == 'daily':
        resample_freq = 'H'
        x_title = "小時"
        if title is None:
            title = "日用電量比較"
        # 獲取最近幾天的數據
        days = [
            (datetime.now() - timedelta(days=i)).date()
            for i in range(3)
        ]
        
        # 創建每天的數據框
        day_dfs = []
        for day in days:
            day_data = data[data['timestamp'].dt.date == day].copy()
            day_data['hour'] = day_data['timestamp'].dt.hour
            day_df = day_data.groupby('hour')['power'].mean().reset_index()
            day_df['day'] = day.strftime('%Y-%m-%d')
            day_dfs.append(day_df)
        
        # 合併數據框
        comparison_df = pd.concat(day_dfs)
        
        # 創建比較圖
        fig = go.Figure()
        
        for day in comparison_df['day'].unique():
            day_df = comparison_df[comparison_df['day'] == day]
            fig.add_trace(go.Scatter(
                x=day_df['hour'],
                y=day_df['power'],
                mode='lines+markers',
                name=day
            ))
        
    elif comparison_type == 'weekly':
        resample_freq = 'D'
        x_title = "星期"
        if title is None:
            title = "週用電量比較"
        # 獲取最近幾週的數據
        weeks = [
            (datetime.now() - timedelta(weeks=i))
            for i in range(3)
        ]
        
        # 創建每週的數據框
        week_dfs = []
        for week_start in weeks:
            week_end = week_start + timedelta(days=6)
            week_data = data[(data['timestamp'] >= week_start) & 
                             (data['timestamp'] <= week_end)].copy()
            week_data['day_of_week'] = week_data['timestamp'].dt.dayofweek
            week_df = week_data.groupby('day_of_week')['power'].mean().reset_index()
            week_df['week'] = f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
            week_dfs.append(week_df)
        
        # 合併數據框
        comparison_df = pd.concat(week_dfs)
        
        # 創建比較圖
        fig = go.Figure()
        
        for week in comparison_df['week'].unique():
            week_df = comparison_df[comparison_df['week'] == week]
            fig.add_trace(go.Scatter(
                x=['週一', '週二', '週三', '週四', '週五', '週六', '週日'],
                y=week_df['power'],
                mode='lines+markers',
                name=week
            ))
        
    elif comparison_type == 'monthly':
        resample_freq = 'W'
        x_title = "週"
        if title is None:
            title = "月用電量比較"
        # 獲取最近幾個月的數據
        months = [
            (datetime.now() - timedelta(days=30*i))
            for i in range(3)
        ]
        
        # 創建每月的數據框
        month_dfs = []
        for month_start in months:
            month_end = month_start + timedelta(days=29)
            month_data = data[(data['timestamp'] >= month_start) & 
                              (data['timestamp'] <= month_end)].copy()
            month_data['week'] = month_data['timestamp'].dt.isocalendar().week
            month_df = month_data.groupby('week')['power'].mean().reset_index()
            month_df['month'] = month_start.strftime('%Y-%m')
            month_dfs.append(month_df)
        
        # 合併數據框
        comparison_df = pd.concat(month_dfs)
        
        # 創建比較圖
        fig = go.Figure()
        
        for month in comparison_df['month'].unique():
            month_df = comparison_df[comparison_df['month'] == month]
            fig.add_trace(go.Scatter(
                x=month_df['week'],
                y=month_df['power'],
                mode='lines+markers',
                name=month
            ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="平均用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

#### 1.2.2 用電量統計圖

```python
def create_statistics_chart(data, period='daily', title=None):
    """
    創建用電量統計圖
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含時間戳和用電量的數據框
    period : str
        統計週期，'daily', 'weekly', 或 'monthly'
    title : str, optional
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        統計圖
    """
    # 根據統計週期設定重採樣頻率和標題
    if period == 'daily':
        resample_freq = 'D'
        if title is None:
            title = "日用電量統計"
    elif period == 'weekly':
        resample_freq = 'W'
        if title is None:
            title = "週用電量統計"
    elif period == 'monthly':
        resample_freq = 'M'
        if title is None:
            title = "月用電量統計"
    
    # 重採樣計算統計量
    resampled = data.set_index('timestamp').resample(resample_freq)
    stats_df = pd.DataFrame({
        'mean': resampled['power'].mean(),
        'max': resampled['power'].max(),
        'min': resampled['power'].min(),
        'std': resampled['power'].std()
    }).reset_index()
    
    # 創建統計圖
    fig = go.Figure()
    
    # 添加平均值曲線
    fig.add_trace(go.Scatter(
        x=stats_df['timestamp'],
        y=stats_df['mean'],
        mode='lines+markers',
        name='平均值',
        line=dict(color='blue', width=2)
    ))
    
    # 添加最大值和最小值範圍
    fig.add_trace(go.Scatter(
        x=stats_df['timestamp'],
        y=stats_df['max'],
        mode='lines',
        name='最大值',
        line=dict(color='red', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=stats_df['timestamp'],
        y=stats_df['min'],
        mode='lines',
        name='最小值',
        line=dict(color='green', width=1),
        fill='tonexty'
    ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

#### 1.2.3 用電量分解圖

```python
def create_decomposition_chart(data, title="用電量時間序列分解"):
    """
    創建用電量分解圖
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含時間戳和用電量的數據框
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        分解圖
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # 重採樣到小時級別
    hourly_data = data.set_index('timestamp').resample('H')['power'].mean()
    
    # 時間序列分解
    decomposition = seasonal_decompose(hourly_data, model='additive', period=24)
    
    # 創建分解圖
    fig = go.Figure()
    
    # 創建子圖
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=("原始數據", "趨勢", "季節性", "殘差"),
                        shared_xaxes=True,
                        vertical_spacing=0.05)
    
    # 添加原始數據
    fig.add_trace(
        go.Scatter(x=hourly_data.index, y=hourly_data.values, mode='lines', name='原始數據'),
        row=1, col=1
    )
    
    # 添加趨勢
    fig.add_trace(
        go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, mode='lines', name='趨勢'),
        row=2, col=1
    )
    
    # 添加季節性
    fig.add_trace(
        go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, mode='lines', name='季節性'),
        row=3, col=1
    )
    
    # 添加殘差
    fig.add_trace(
        go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, mode='lines', name='殘差'),
        row=4, col=1
    )
    
    # 設定佈局
    fig.update_layout(
        title=title,
        height=800,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    
    # 更新y軸標題
    fig.update_yaxes(title_text="用電量 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="趨勢", row=2, col=1)
    fig.update_yaxes(title_text="季節性", row=3, col=1)
    fig.update_yaxes(title_text="殘差", row=4, col=1)
    
    return fig
```

## 2. 預測結果可視化設計

### 2.1 預測vs實際值比較圖

```python
def create_prediction_comparison_chart(actual, predicted, prediction_interval=None, 
                                      title="預測vs實際用電量"):
    """
    創建預測vs實際值比較圖
    
    Parameters:
    -----------
    actual : pd.Series
        實際用電量，帶時間索引
    predicted : pd.Series
        預測用電量，帶時間索引
    prediction_interval : tuple, optional
        預測區間 (lower, upper)
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        比較圖
    """
    # 創建比較圖
    fig = go.Figure()
    
    # 添加實際值
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='實際值',
        line=dict(color='blue', width=2)
    ))
    
    # 添加預測值
    fig.add_trace(go.Scatter(
        x=predicted.index,
        y=predicted.values,
        mode='lines',
        name='預測值',
        line=dict(color='red', width=2)
    ))
    
    # 添加預測區間（如果有）
    if prediction_interval is not None:
        lower, upper = prediction_interval
        
        # 添加上界
        fig.add_trace(go.Scatter(
            x=upper.index,
            y=upper.values,
            mode='lines',
            name='上界 (95%)',
            line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
            showlegend=False
        ))
        
        # 添加下界
        fig.add_trace(go.Scatter(
            x=lower.index,
            y=lower.values,
            mode='lines',
            name='下界 (95%)',
            line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            showlegend=True
        ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

### 2.2 預測誤差分析圖

```python
def create_error_analysis_chart(actual, predicted, title="預測誤差分析"):
    """
    創建預測誤差分析圖
    
    Parameters:
    -----------
    actual : pd.Series
        實際用電量，帶時間索引
    predicted : pd.Series
        預測用電量，帶時間索引
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        誤差分析圖
    """
    # 計算誤差
    error = actual - predicted
    percentage_error = (error / actual) * 100
    
    # 創建子圖
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("絕對誤差", "百分比誤差"),
                        shared_xaxes=True,
                        vertical_spacing=0.1)
    
    # 添加絕對誤差
    fig.add_trace(
        go.Scatter(x=error.index, y=error.values, mode='lines', name='絕對誤差'),
        row=1, col=1
    )
    
    # 添加零線
    fig.add_trace(
        go.Scatter(x=[error.index.min(), error.index.max()], y=[0, 0], 
                  mode='lines', name='零線', line=dict(color='black', dash='dash')),
        row=1, col=1
    )
    
    # 添加百分比誤差
    fig.add_trace(
        go.Scatter(x=percentage_error.index, y=percentage_error.values, mode='lines', name='百分比誤差'),
        row=2, col=1
    )
    
    # 添加零線
    fig.add_trace(
        go.Scatter(x=[percentage_error.index.min(), percentage_error.index.max()], y=[0, 0], 
                  mode='lines', name='零線', line=dict(color='black', dash='dash')),
        row=2, col=1
    )
    
    # 設定佈局
    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        hovermode="x unified"
    )
    
    # 更新y軸標題
    fig.update_yaxes(title_text="絕對誤差 (kW)", row=1, col=1)
    fig.update_yaxes(title_text="百分比誤差 (%)", row=2, col=1)
    
    return fig
```

### 2.3 多模型預測比較圖

```python
def create_multi_model_comparison_chart(actual, predictions_dict, title="多模型預測比較"):
    """
    創建多模型預測比較圖
    
    Parameters:
    -----------
    actual : pd.Series
        實際用電量，帶時間索引
    predictions_dict : dict
        模型名稱到預測值的字典
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        多模型比較圖
    """
    # 創建比較圖
    fig = go.Figure()
    
    # 添加實際值
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='實際值',
        line=dict(color='blue', width=2)
    ))
    
    # 添加各模型預測值
    colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink']
    for i, (model_name, predicted) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode='lines',
            name=f'{model_name}預測',
            line=dict(color=color, width=1.5)
        ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

### 2.4 預測視野比較圖

```python
def create_horizon_comparison_chart(actual, horizon_predictions, title="不同預測視野比較"):
    """
    創建預測視野比較圖
    
    Parameters:
    -----------
    actual : pd.Series
        實際用電量，帶時間索引
    horizon_predictions : dict
        預測視野到預測值的字典
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        預測視野比較圖
    """
    # 創建比較圖
    fig = go.Figure()
    
    # 添加實際值
    fig.add_trace(go.Scatter(
        x=actual.index,
        y=actual.values,
        mode='lines',
        name='實際值',
        line=dict(color='blue', width=2)
    ))
    
    # 添加各預測視野的預測值
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, (horizon, predicted) in enumerate(horizon_predictions.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode='lines',
            name=f'{horizon}預測',
            line=dict(color=color, width=1.5)
        ))
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="時間",
        yaxis_title="用電量 (kW)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig
```

## 3. 性能指標可視化設計

### 3.1 模型性能比較圖

```python
def create_model_performance_chart(metrics_dict, metric_name='rmse', title=None):
    """
    創建模型性能比較圖
    
    Parameters:
    -----------
    metrics_dict : dict
        模型名稱到性能指標字典的字典
    metric_name : str
        要比較的指標名稱
    title : str, optional
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        性能比較圖
    """
    # 設定標題
    if title is None:
        metric_labels = {
            'rmse': 'RMSE (均方根誤差)',
            'mae': 'MAE (平均絕對誤差)',
            'mape': 'MAPE (平均絕對百分比誤差)',
            'smape': 'SMAPE (對稱平均絕對百分比誤差)',
            'r2': 'R² (決定係數)',
            'direction_accuracy': '方向準確率'
        }
        title = f"模型{metric_labels.get(metric_name, metric_name)}比較"
    
    # 提取模型名稱和指標值
    models = list(metrics_dict.keys())
    values = [metrics[metric_name] for metrics in metrics_dict.values()]
    
    # 創建條形圖
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            text=values,
            textposition='auto',
            marker_color='skyblue'
        )
    ])
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="模型",
        yaxis_title=metric_labels.get(metric_name, metric_name),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
```

### 3.2 預測視野性能圖

```python
def create_horizon_performance_chart(horizon_metrics, metric_name='rmse', model_name=None, title=None):
    """
    創建預測視野性能圖
    
    Parameters:
    -----------
    horizon_metrics : dict
        預測視野到性能指標字典的字典
    metric_name : str
        要比較的指標名稱
    model_name : str, optional
        模型名稱
    title : str, optional
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        預測視野性能圖
    """
    # 設定標題
    if title is None:
        metric_labels = {
            'rmse': 'RMSE (均方根誤差)',
            'mae': 'MAE (平均絕對誤差)',
            'mape': 'MAPE (平均絕對百分比誤差)',
            'smape': 'SMAPE (對稱平均絕對百分比誤差)',
            'r2': 'R² (決定係數)',
            'direction_accuracy': '方向準確率'
        }
        model_str = f"{model_name} " if model_name else ""
        title = f"{model_str}不同預測視野的{metric_labels.get(metric_name, metric_name)}"
    
    # 提取預測視野和指標值
    horizons = list(horizon_metrics.keys())
    values = [metrics[metric_name] for metrics in horizon_metrics.values()]
    
    # 創建線圖
    fig = go.Figure(data=[
        go.Scatter(
            x=horizons,
            y=values,
            mode='lines+markers',
            marker=dict(size=10),
            line=dict(width=2),
            text=values,
            textposition="top center"
        )
    ])
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="預測視野",
        yaxis_title=metric_labels.get(metric_name, metric_name),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
```

### 3.3 特徵重要性圖

```python
def create_feature_importance_chart(feature_importance, title="特徵重要性"):
    """
    創建特徵重要性圖
    
    Parameters:
    -----------
    feature_importance : dict or pd.Series
        特徵名稱到重要性值的字典或Series
    title : str
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        特徵重要性圖
    """
    # 轉換為Series（如果是字典）
    if isinstance(feature_importance, dict):
        feature_importance = pd.Series(feature_importance)
    
    # 排序並選取前15個特徵
    top_features = feature_importance.sort_values(ascending=False).head(15)
    
    # 創建水平條形圖
    fig = go.Figure(data=[
        go.Bar(
            y=top_features.index,
            x=top_features.values,
            orientation='h',
            marker_color='lightgreen'
        )
    ])
    
    # 設定佈局
    fig.update_layout(
        title=title,
        xaxis_title="重要性",
        yaxis_title="特徵",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig
```

### 3.4 交叉驗證性能圖

```python
def create_cv_performance_chart(cv_results, metric_name='rmse', title=None):
    """
    創建交叉驗證性能圖
    
    Parameters:
    -----------
    cv_results : dict
        交叉驗證結果字典
    metric_name : str
        要比較的指標名稱
    title : str, optional
        標題
        
    Returns:
    --------
    plotly.graph_objects.Figure
        交叉驗證性能圖
    """
    # 設定標題
    if title is None:
        metric_labels = {
            'rmse': 'RMSE (均方根誤差)',
            'mae': 'MAE (平均絕對誤差)',
            'mape': 'MAPE (平均絕對百分比誤差)',
            'smape': 'SMAPE (對稱平均絕對百分比誤差)',
            'r2': 'R² (決定係數)',
            'direction_accuracy': '方向準確率'
        }
        title = f"交叉驗證{metric_labels.get(metric_name, metric_name)}"
    
    # 提取每次交叉驗證的指標值
    cv_metrics = [metrics[metric_name] for metrics in cv_results['metrics']]
    
    # 計算平均值和標準差
    mean_metric = np.mean(cv_metrics)
    std_metric = np.std(cv_metrics)
    
    # 創建箱形圖
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=cv_metrics,
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        name=metric_name
    ))
    
    # 添加平均值線
    fig.add_shape(
        type='line',
        x0=-0.5,
        x1=0.5,
        y0=mean_metric,
        y1=mean_metric,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        )
    )
    
    # 添加標註
    fig.add_annotation(
        x=0.5,
        y=mean_metric,
        text=f"平均值: {mean_metric:.4f}<br>標準差: {std_metric:.4f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30
    )
    
    # 設定佈局
    fig.update_layout(
        title=title,
        yaxis_title=metric_labels.get(metric_name, metric_name),
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig
```

## 4. 互動式儀表板實作

### 4.1 Dash應用架構

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 初始化Dash應用
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# 定義應用佈局
app.layout = html.Div([
    # 頁面標題
    html.H1("辦公室用電預測儀表板", style={'textAlign': 'center'}),
    
    # 頁面選項卡
    dcc.Tabs(id='tabs', value='tab-realtime', children=[
        dcc.Tab(label='即時監控', value='tab-realtime'),
        dcc.Tab(label='歷史分析', value='tab-historical'),
        dcc.Tab(label='預測結果', value='tab-prediction'),
        dcc.Tab(label='模型性能', value='tab-performance'),
    ]),
    
    # 頁面內容
    html.Div(id='tabs-content')
])

# 定義頁面回調
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-realtime':
        return render_realtime_tab()
    elif tab == 'tab-historical':
        return render_historical_tab()
    elif tab == 'tab-prediction':
        return render_prediction_tab()
    elif tab == 'tab-performance':
        return render_performance_tab()

# 定義即時監控頁面
def render_realtime_tab():
    return html.Div([
        html.H2("即時用電監控"),
        
        # 控制面板
        html.Div([
            html.Label("更新頻率:"),
            dcc.Dropdown(
                id='update-interval-dropdown',
                options=[
                    {'label': '5秒', 'value': 5000},
                    {'label': '10秒', 'value': 10000},
                    {'label': '30秒', 'value': 30000},
                    {'label': '1分鐘', 'value': 60000},
                ],
                value=10000,
                style={'width': '200px'}
            ),
            
            html.Label("顯示時間範圍:"),
            dcc.Dropdown(
                id='time-window-dropdown',
                options=[
                    {'label': '最近30分鐘', 'value': 0.5},
                    {'label': '最近1小時', 'value': 1},
                    {'label': '最近3小時', 'value': 3},
                    {'label': '最近6小時', 'value': 6},
                ],
                value=1,
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 儀表盤區域
        html.Div([
            html.Div([
                dcc.Graph(id='gauge-current-power')
            ], style={'width': '33%'}),
            
            html.Div([
                dcc.Graph(id='gauge-daily-energy')
            ], style={'width': '33%'}),
            
            html.Div([
                dcc.Graph(id='gauge-power-factor')
            ], style={'width': '33%'}),
        ], style={'display': 'flex'}),
        
        # 趨勢圖區域
        html.Div([
            dcc.Graph(id='realtime-trend-chart')
        ]),
        
        # 熱圖區域
        html.Div([
            dcc.Graph(id='power-heatmap')
        ]),
        
        # 異常檢測區域
        html.Div([
            html.H3("異常檢測", style={'textAlign': 'center'}),
            html.Div(id='anomaly-alerts', style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
        ]),
        
        # 定時更新組件
        dcc.Interval(
            id='interval-component',
            interval=10*1000,  # 10秒
            n_intervals=0
        )
    ])

# 定義歷史分析頁面
def render_historical_tab():
    return html.Div([
        html.H2("歷史用電分析"),
        
        # 控制面板
        html.Div([
            html.Label("分析類型:"),
            dcc.Dropdown(
                id='analysis-type-dropdown',
                options=[
                    {'label': '日用電量比較', 'value': 'daily'},
                    {'label': '週用電量比較', 'value': 'weekly'},
                    {'label': '月用電量比較', 'value': 'monthly'},
                ],
                value='daily',
                style={'width': '200px'}
            ),
            
            html.Label("日期範圍:"),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                style={'width': '400px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 比較圖區域
        html.Div([
            dcc.Graph(id='comparison-chart')
        ]),
        
        # 統計圖區域
        html.Div([
            dcc.Graph(id='statistics-chart')
        ]),
        
        # 分解圖區域
        html.Div([
            dcc.Graph(id='decomposition-chart')
        ]),
    ])

# 定義預測結果頁面
def render_prediction_tab():
    return html.Div([
        html.H2("用電預測結果"),
        
        # 控制面板
        html.Div([
            html.Label("預測模型:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'ARIMA', 'value': 'arima'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'LSTM', 'value': 'lstm'},
                    {'label': '所有模型', 'value': 'all'},
                ],
                value='xgboost',
                style={'width': '200px'}
            ),
            
            html.Label("預測視野:"),
            dcc.Dropdown(
                id='horizon-dropdown',
                options=[
                    {'label': '短期 (1小時)', 'value': 'short'},
                    {'label': '中期 (1天)', 'value': 'medium'},
                    {'label': '長期 (1週)', 'value': 'long'},
                ],
                value='short',
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 預測比較圖區域
        html.Div([
            dcc.Graph(id='prediction-comparison-chart')
        ]),
        
        # 誤差分析圖區域
        html.Div([
            dcc.Graph(id='error-analysis-chart')
        ]),
        
        # 多模型比較圖區域
        html.Div([
            dcc.Graph(id='multi-model-comparison-chart')
        ]),
        
        # 預測視野比較圖區域
        html.Div([
            dcc.Graph(id='horizon-comparison-chart')
        ]),
    ])

# 定義模型性能頁面
def render_performance_tab():
    return html.Div([
        html.H2("模型性能評估"),
        
        # 控制面板
        html.Div([
            html.Label("評估指標:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'RMSE (均方根誤差)', 'value': 'rmse'},
                    {'label': 'MAE (平均絕對誤差)', 'value': 'mae'},
                    {'label': 'MAPE (平均絕對百分比誤差)', 'value': 'mape'},
                    {'label': 'R² (決定係數)', 'value': 'r2'},
                    {'label': '方向準確率', 'value': 'direction_accuracy'},
                ],
                value='rmse',
                style={'width': '300px'}
            ),
            
            html.Label("預測視野:"),
            dcc.Dropdown(
                id='performance-horizon-dropdown',
                options=[
                    {'label': '短期 (1小時)', 'value': 'short'},
                    {'label': '中期 (1天)', 'value': 'medium'},
                    {'label': '長期 (1週)', 'value': 'long'},
                    {'label': '所有視野', 'value': 'all'},
                ],
                value='all',
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 模型性能比較圖區域
        html.Div([
            dcc.Graph(id='model-performance-chart')
        ]),
        
        # 預測視野性能圖區域
        html.Div([
            dcc.Graph(id='horizon-performance-chart')
        ]),
        
        # 特徵重要性圖區域
        html.Div([
            dcc.Graph(id='feature-importance-chart')
        ]),
        
        # 交叉驗證性能圖區域
        html.Div([
            dcc.Graph(id='cv-performance-chart')
        ]),
    ])
```

### 4.2 即時監控回調

```python
# 更新儀表盤
@app.callback(
    [Output('gauge-current-power', 'figure'),
     Output('gauge-daily-energy', 'figure'),
     Output('gauge-power-factor', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_gauges(n):
    # 獲取最新數據
    latest_data = get_latest_data()
    
    # 創建當前功率儀表盤
    current_power = latest_data['power'].iloc[-1]
    current_power_fig = create_gauge_chart(
        current_value=current_power,
        min_value=0,
        max_value=100,
        threshold_value=80,
        title="當前功率 (kW)"
    )
    
    # 創建日累計用電量儀表盤
    today = datetime.now().date()
    today_data = latest_data[latest_data.index.date == today]
    daily_energy = today_data['power'].mean() * len(today_data) * 5 / 3600  # kWh
    daily_energy_fig = create_gauge_chart(
        current_value=daily_energy,
        min_value=0,
        max_value=1000,
        threshold_value=800,
        title="今日用電量 (kWh)"
    )
    
    # 創建功率因數儀表盤
    if 'power_factor' in latest_data.columns:
        power_factor = latest_data['power_factor'].iloc[-1]
    else:
        power_factor = 0.95  # 假設值
    power_factor_fig = create_gauge_chart(
        current_value=power_factor,
        min_value=0.7,
        max_value=1.0,
        threshold_value=0.9,
        title="功率因數"
    )
    
    return current_power_fig, daily_energy_fig, power_factor_fig

# 更新即時趨勢圖
@app.callback(
    Output('realtime-trend-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('time-window-dropdown', 'value')]
)
def update_realtime_trend(n, window_hours):
    # 獲取數據
    data = get_data()
    
    # 創建即時趨勢圖
    fig = create_realtime_trend_chart(data, window_hours=window_hours)
    
    return fig

# 更新熱圖
@app.callback(
    Output('power-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_heatmap(n):
    # 獲取數據
    data = get_data()
    
    # 創建熱圖
    fig = create_heatmap(data)
    
    return fig

# 更新異常檢測
@app.callback(
    Output('anomaly-alerts', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_anomaly_detection(n):
    # 獲取最新數據
    latest_data = get_latest_data()
    
    # 檢測異常
    anomalies = detect_anomalies(latest_data)
    
    if not anomalies:
        return html.P("目前沒有檢測到異常", style={'color': 'green'})
    
    # 創建異常警報列表
    alerts = []
    for anomaly in anomalies:
        alerts.append(html.Div([
            html.H4(f"異常類型: {anomaly['type']}", style={'color': 'red'}),
            html.P(f"時間: {anomaly['time']}"),
            html.P(f"詳情: {anomaly['details']}"),
            html.Hr()
        ]))
    
    return alerts

# 異常檢測函數
def detect_anomalies(data, window=60, threshold=3):
    """
    檢測用電異常
    
    Parameters:
    -----------
    data : pd.DataFrame
        電力數據
    window : int
        移動窗口大小
    threshold : float
        異常閾值（標準差倍數）
        
    Returns:
    --------
    list
        異常列表
    """
    anomalies = []
    
    # 計算移動平均和標準差
    rolling_mean = data['power'].rolling(window=window, min_periods=1).mean()
    rolling_std = data['power'].rolling(window=window, min_periods=1).std()
    
    # 計算Z分數
    z_scores = (data['power'] - rolling_mean) / rolling_std
    
    # 檢測異常
    anomaly_mask = abs(z_scores) > threshold
    
    if anomaly_mask.any():
        anomaly_times = data.index[anomaly_mask]
        anomaly_values = data.loc[anomaly_mask, 'power']
        
        for time, value in zip(anomaly_times, anomaly_values):
            z_score = z_scores.loc[time]
            direction = "高於" if z_score > 0 else "低於"
            anomalies.append({
                'type': '用電量異常',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': f"用電量 {value:.2f} kW，{direction}預期值 {abs(z_score):.2f} 個標準差"
            })
    
    return anomalies
```

### 4.3 歷史分析回調

```python
# 更新比較圖
@app.callback(
    Output('comparison-chart', 'figure'),
    [Input('analysis-type-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_comparison_chart(analysis_type, start_date, end_date):
    # 獲取數據
    data = get_data(start_date, end_date)
    
    # 創建比較圖
    fig = create_comparison_chart(data, comparison_type=analysis_type)
    
    return fig

# 更新統計圖
@app.callback(
    Output('statistics-chart', 'figure'),
    [Input('analysis-type-dropdown', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_statistics_chart(analysis_type, start_date, end_date):
    # 獲取數據
    data = get_data(start_date, end_date)
    
    # 創建統計圖
    fig = create_statistics_chart(data, period=analysis_type)
    
    return fig

# 更新分解圖
@app.callback(
    Output('decomposition-chart', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_decomposition_chart(start_date, end_date):
    # 獲取數據
    data = get_data(start_date, end_date)
    
    # 創建分解圖
    fig = create_decomposition_chart(data)
    
    return fig
```

### 4.4 預測結果回調

```python
# 更新預測比較圖
@app.callback(
    Output('prediction-comparison-chart', 'figure'),
    [Input('model-dropdown', 'value'),
     Input('horizon-dropdown', 'value')]
)
def update_prediction_comparison(model, horizon):
    # 獲取預測結果
    actual, predicted, prediction_interval = get_prediction_results(model, horizon)
    
    # 創建預測比較圖
    fig = create_prediction_comparison_chart(
        actual, predicted, prediction_interval,
        title=f"{model.upper()} {horizon}期預測vs實際用電量"
    )
    
    return fig

# 更新誤差分析圖
@app.callback(
    Output('error-analysis-chart', 'figure'),
    [Input('model-dropdown', 'value'),
     Input('horizon-dropdown', 'value')]
)
def update_error_analysis(model, horizon):
    # 獲取預測結果
    actual, predicted, _ = get_prediction_results(model, horizon)
    
    # 創建誤差分析圖
    fig = create_error_analysis_chart(
        actual, predicted,
        title=f"{model.upper()} {horizon}期預測誤差分析"
    )
    
    return fig

# 更新多模型比較圖
@app.callback(
    Output('multi-model-comparison-chart', 'figure'),
    [Input('horizon-dropdown', 'value')]
)
def update_multi_model_comparison(horizon):
    # 獲取所有模型的預測結果
    models = ['arima', 'xgboost', 'lstm']
    predictions_dict = {}
    
    actual = None
    for model in models:
        actual, predicted, _ = get_prediction_results(model, horizon)
        predictions_dict[model] = predicted
    
    # 創建多模型比較圖
    fig = create_multi_model_comparison_chart(
        actual, predictions_dict,
        title=f"多模型{horizon}期預測比較"
    )
    
    return fig

# 更新預測視野比較圖
@app.callback(
    Output('horizon-comparison-chart', 'figure'),
    [Input('model-dropdown', 'value')]
)
def update_horizon_comparison(model):
    # 獲取不同預測視野的結果
    horizons = ['short', 'medium', 'long']
    horizon_predictions = {}
    
    actual = None
    for horizon in horizons:
        actual, predicted, _ = get_prediction_results(model, horizon)
        horizon_predictions[horizon] = predicted
    
    # 創建預測視野比較圖
    fig = create_horizon_comparison_chart(
        actual, horizon_predictions,
        title=f"{model.upper()} 不同預測視野比較"
    )
    
    return fig
```

### 4.5 模型性能回調

```python
# 更新模型性能比較圖
@app.callback(
    Output('model-performance-chart', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('performance-horizon-dropdown', 'value')]
)
def update_model_performance(metric, horizon):
    # 獲取模型性能指標
    metrics_dict = get_model_metrics(horizon)
    
    # 創建模型性能比較圖
    fig = create_model_performance_chart(
        metrics_dict, metric_name=metric,
        title=f"{horizon}期預測模型{metric.upper()}比較"
    )
    
    return fig

# 更新預測視野性能圖
@app.callback(
    Output('horizon-performance-chart', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_horizon_performance(metric, model):
    # 獲取不同預測視野的性能指標
    horizon_metrics = get_horizon_metrics(model)
    
    # 創建預測視野性能圖
    fig = create_horizon_performance_chart(
        horizon_metrics, metric_name=metric, model_name=model
    )
    
    return fig

# 更新特徵重要性圖
@app.callback(
    Output('feature-importance-chart', 'figure'),
    [Input('model-dropdown', 'value')]
)
def update_feature_importance(model):
    # 獲取特徵重要性
    feature_importance = get_feature_importance(model)
    
    # 創建特徵重要性圖
    fig = create_feature_importance_chart(
        feature_importance,
        title=f"{model.upper()} 特徵重要性"
    )
    
    return fig

# 更新交叉驗證性能圖
@app.callback(
    Output('cv-performance-chart', 'figure'),
    [Input('metric-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_cv_performance(metric, model):
    # 獲取交叉驗證結果
    cv_results = get_cv_results(model)
    
    # 創建交叉驗證性能圖
    fig = create_cv_performance_chart(
        cv_results, metric_name=metric,
        title=f"{model.upper()} 交叉驗證{metric.upper()}"
    )
    
    return fig
```

## 5. 異常檢測警報設計

### 5.1 異常檢測算法

```python
def detect_anomalies_advanced(data, config=None):
    """
    高級異常檢測
    
    Parameters:
    -----------
    data : pd.DataFrame
        電力數據
    config : dict, optional
        配置參數
        
    Returns:
    --------
    list
        異常列表
    """
    # 默認配置
    default_config = {
        'z_score_threshold': 3.0,
        'window_size': 60,
        'sudden_change_threshold': 0.3,
        'flatline_window': 12,
        'flatline_std_threshold': 0.01,
        'peak_threshold': 0.8,
        'contract_capacity': 100  # kW
    }
    
    # 使用提供的配置或默認配置
    config = config or default_config
    
    anomalies = []
    
    # 1. Z分數異常檢測
    z_score_anomalies = detect_z_score_anomalies(
        data, 
        window=config['window_size'], 
        threshold=config['z_score_threshold']
    )
    anomalies.extend(z_score_anomalies)
    
    # 2. 突變檢測
    sudden_change_anomalies = detect_sudden_changes(
        data, 
        threshold=config['sudden_change_threshold']
    )
    anomalies.extend(sudden_change_anomalies)
    
    # 3. 平線檢測
    flatline_anomalies = detect_flatlines(
        data, 
        window=config['flatline_window'], 
        std_threshold=config['flatline_std_threshold']
    )
    anomalies.extend(flatline_anomalies)
    
    # 4. 峰值檢測
    peak_anomalies = detect_peaks(
        data, 
        threshold=config['peak_threshold'] * config['contract_capacity']
    )
    anomalies.extend(peak_anomalies)
    
    # 5. 週期性異常檢測
    periodic_anomalies = detect_periodic_anomalies(data)
    anomalies.extend(periodic_anomalies)
    
    return anomalies

def detect_z_score_anomalies(data, window=60, threshold=3.0):
    """
    基於Z分數的異常檢測
    """
    anomalies = []
    
    # 計算移動平均和標準差
    rolling_mean = data['power'].rolling(window=window, min_periods=1).mean()
    rolling_std = data['power'].rolling(window=window, min_periods=1).std()
    
    # 計算Z分數
    z_scores = (data['power'] - rolling_mean) / rolling_std
    
    # 檢測異常
    anomaly_mask = abs(z_scores) > threshold
    
    if anomaly_mask.any():
        anomaly_times = data.index[anomaly_mask]
        anomaly_values = data.loc[anomaly_mask, 'power']
        
        for time, value in zip(anomaly_times, anomaly_values):
            z_score = z_scores.loc[time]
            direction = "高於" if z_score > 0 else "低於"
            anomalies.append({
                'type': 'Z分數異常',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': f"用電量 {value:.2f} kW，{direction}預期值 {abs(z_score):.2f} 個標準差",
                'severity': 'high' if abs(z_score) > 5 else 'medium'
            })
    
    return anomalies

def detect_sudden_changes(data, threshold=0.3):
    """
    突變檢測
    """
    anomalies = []
    
    # 計算差分
    power_diff = data['power'].diff().abs()
    power_pct_diff = power_diff / data['power'].shift(1)
    
    # 檢測突變
    anomaly_mask = power_pct_diff > threshold
    
    if anomaly_mask.any():
        anomaly_times = data.index[anomaly_mask]
        anomaly_values = data.loc[anomaly_mask, 'power']
        anomaly_diffs = power_pct_diff.loc[anomaly_mask]
        
        for time, value, diff in zip(anomaly_times, anomaly_values, anomaly_diffs):
            direction = "增加" if data['power'].loc[time] > data['power'].shift(1).loc[time] else "減少"
            anomalies.append({
                'type': '突變異常',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': f"用電量突然{direction} {diff*100:.1f}%，從 {data['power'].shift(1).loc[time]:.2f} kW 到 {value:.2f} kW",
                'severity': 'high' if diff > 0.5 else 'medium'
            })
    
    return anomalies

def detect_flatlines(data, window=12, std_threshold=0.01):
    """
    平線檢測
    """
    anomalies = []
    
    # 計算移動標準差
    rolling_std = data['power'].rolling(window=window).std()
    rolling_mean = data['power'].rolling(window=window).mean()
    
    # 檢測平線
    anomaly_mask = (rolling_std / rolling_mean) < std_threshold
    
    if anomaly_mask.any():
        # 合併連續的平線區間
        anomaly_starts = []
        anomaly_ends = []
        
        in_anomaly = False
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_anomaly:
                in_anomaly = True
                anomaly_starts.append(i)
            elif not is_anomaly and in_anomaly:
                in_anomaly = False
                anomaly_ends.append(i - 1)
        
        if in_anomaly:
            anomaly_ends.append(len(anomaly_mask) - 1)
        
        # 創建異常記錄
        for start, end in zip(anomaly_starts, anomaly_ends):
            if end - start + 1 >= window:  # 只報告足夠長的平線
                start_time = data.index[start]
                end_time = data.index[end]
                duration_minutes = (end_time - start_time).total_seconds() / 60
                
                anomalies.append({
                    'type': '平線異常',
                    'time': f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    'details': f"用電量在 {duration_minutes:.1f} 分鐘內幾乎不變，保持在 {data['power'].iloc[start:end+1].mean():.2f} kW",
                    'severity': 'medium' if duration_minutes > 30 else 'low'
                })
    
    return anomalies

def detect_peaks(data, threshold=80):
    """
    峰值檢測
    """
    anomalies = []
    
    # 檢測峰值
    anomaly_mask = data['power'] > threshold
    
    if anomaly_mask.any():
        # 合併連續的峰值區間
        anomaly_starts = []
        anomaly_ends = []
        
        in_anomaly = False
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_anomaly:
                in_anomaly = True
                anomaly_starts.append(i)
            elif not is_anomaly and in_anomaly:
                in_anomaly = False
                anomaly_ends.append(i - 1)
        
        if in_anomaly:
            anomaly_ends.append(len(anomaly_mask) - 1)
        
        # 創建異常記錄
        for start, end in zip(anomaly_starts, anomaly_ends):
            start_time = data.index[start]
            end_time = data.index[end]
            duration_minutes = (end_time - start_time).total_seconds() / 60
            max_value = data['power'].iloc[start:end+1].max()
            
            anomalies.append({
                'type': '峰值異常',
                'time': f"{start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                'details': f"用電量在 {duration_minutes:.1f} 分鐘內超過閾值，最高達到 {max_value:.2f} kW",
                'severity': 'high' if max_value > threshold * 1.2 else 'medium'
            })
    
    return anomalies

def detect_periodic_anomalies(data):
    """
    週期性異常檢測
    """
    anomalies = []
    
    # 按小時和星期幾分組
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    
    # 計算每小時每天的平均用電量和標準差
    hourly_stats = data.groupby(['day_of_week', 'hour'])['power'].agg(['mean', 'std']).reset_index()
    
    # 合併回原始數據
    merged = pd.merge(
        data, hourly_stats, 
        on=['day_of_week', 'hour'], 
        how='left'
    )
    
    # 計算Z分數
    merged['z_score'] = (merged['power'] - merged['mean']) / merged['std']
    
    # 檢測異常
    anomaly_mask = abs(merged['z_score']) > 3
    
    if anomaly_mask.any():
        anomaly_times = merged.index[anomaly_mask]
        anomaly_values = merged.loc[anomaly_mask, 'power']
        anomaly_means = merged.loc[anomaly_mask, 'mean']
        anomaly_z_scores = merged.loc[anomaly_mask, 'z_score']
        
        for time, value, mean, z_score in zip(anomaly_times, anomaly_values, anomaly_means, anomaly_z_scores):
            direction = "高於" if z_score > 0 else "低於"
            day_name = ['週一', '週二', '週三', '週四', '週五', '週六', '週日'][time.dayofweek]
            anomalies.append({
                'type': '週期性異常',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': f"用電量 {value:.2f} kW，{direction}{day_name} {time.hour}時的正常水平 {mean:.2f} kW，差異 {abs(z_score):.2f} 個標準差",
                'severity': 'medium' if abs(z_score) > 5 else 'low'
            })
    
    return anomalies
```

### 5.2 警報通知系統

```python
def create_alert_notification(anomalies):
    """
    創建警報通知
    
    Parameters:
    -----------
    anomalies : list
        異常列表
        
    Returns:
    --------
    html.Div
        警報通知組件
    """
    if not anomalies:
        return html.Div([
            html.H4("系統狀態", style={'textAlign': 'center'}),
            html.P("正常運行中", style={'color': 'green', 'textAlign': 'center'}),
            html.Div([
                html.I(className="fas fa-check-circle", style={'fontSize': '48px', 'color': 'green'}),
            ], style={'textAlign': 'center', 'margin': '20px'})
        ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'margin': '10px'})
    
    # 按嚴重性排序
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_anomalies = sorted(anomalies, key=lambda x: severity_order.get(x.get('severity', 'low'), 3))
    
    # 創建警報列表
    alerts = []
    for anomaly in sorted_anomalies:
        severity = anomaly.get('severity', 'low')
        
        if severity == 'high':
            color = 'red'
            icon = 'fas fa-exclamation-triangle'
        elif severity == 'medium':
            color = 'orange'
            icon = 'fas fa-exclamation-circle'
        else:
            color = 'blue'
            icon = 'fas fa-info-circle'
        
        alerts.append(html.Div([
            html.Div([
                html.I(className=icon, style={'fontSize': '24px', 'marginRight': '10px'}),
                html.H4(f"{anomaly['type']}", style={'margin': '0', 'display': 'inline'})
            ], style={'display': 'flex', 'alignItems': 'center', 'color': color}),
            html.P(f"時間: {anomaly['time']}"),
            html.P(f"詳情: {anomaly['details']}"),
            html.Hr()
        ], style={'margin': '10px 0'}))
    
    # 創建警報面板
    return html.Div([
        html.H4("異常警報", style={'textAlign': 'center', 'color': 'red'}),
        html.P(f"檢測到 {len(anomalies)} 個異常", style={'textAlign': 'center'}),
        html.Div(alerts)
    ], style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '15px', 'margin': '10px'})
```

### 5.3 預測偏差警報

```python
def detect_prediction_deviations(actual, predicted, threshold=0.2):
    """
    檢測預測偏差
    
    Parameters:
    -----------
    actual : pd.Series
        實際用電量，帶時間索引
    predicted : pd.Series
        預測用電量，帶時間索引
    threshold : float
        偏差閾值（百分比）
        
    Returns:
    --------
    list
        偏差警報列表
    """
    deviations = []
    
    # 計算偏差
    error = actual - predicted
    percentage_error = (error / actual).abs()
    
    # 檢測偏差
    deviation_mask = percentage_error > threshold
    
    if deviation_mask.any():
        deviation_times = percentage_error.index[deviation_mask]
        deviation_values = percentage_error.loc[deviation_mask]
        actual_values = actual.loc[deviation_mask]
        predicted_values = predicted.loc[deviation_mask]
        
        for time, value, actual_val, predicted_val in zip(deviation_times, deviation_values, actual_values, predicted_values):
            direction = "高於" if actual_val > predicted_val else "低於"
            deviations.append({
                'type': '預測偏差',
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'details': f"實際用電量 {actual_val:.2f} kW，{direction}預測值 {predicted_val:.2f} kW，偏差 {value*100:.1f}%",
                'severity': 'high' if value > 0.5 else 'medium' if value > 0.3 else 'low'
            })
    
    return deviations
```

## 6. 儀表板部署

### 6.1 儀表板啟動腳本

```python
# app.py

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# 導入可視化函數
from visualization_functions import (
    create_gauge_chart, create_realtime_trend_chart, create_heatmap,
    create_comparison_chart, create_statistics_chart, create_decomposition_chart,
    create_prediction_comparison_chart, create_error_analysis_chart,
    create_multi_model_comparison_chart, create_horizon_comparison_chart,
    create_model_performance_chart, create_horizon_performance_chart,
    create_feature_importance_chart, create_cv_performance_chart
)

# 導入異常檢測函數
from anomaly_detection import (
    detect_anomalies_advanced, detect_prediction_deviations,
    create_alert_notification
)

# 導入數據處理函數
from data_functions import (
    get_data, get_latest_data, get_prediction_results,
    get_model_metrics, get_horizon_metrics,
    get_feature_importance, get_cv_results
)

# 初始化Dash應用
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# 設定應用標題
app.title = "辦公室用電預測儀表板"

# 定義應用佈局
app.layout = html.Div([
    # 頁面標題
    html.Div([
        html.H1("辦公室用電預測儀表板", style={'textAlign': 'center'}),
        html.P("每5秒收集一次的電錶數據分析與預測", style={'textAlign': 'center'})
    ], style={'margin': '20px 0'}),
    
    # 頁面選項卡
    dcc.Tabs(id='tabs', value='tab-realtime', children=[
        dcc.Tab(label='即時監控', value='tab-realtime'),
        dcc.Tab(label='歷史分析', value='tab-historical'),
        dcc.Tab(label='預測結果', value='tab-prediction'),
        dcc.Tab(label='模型性能', value='tab-performance'),
    ]),
    
    # 頁面內容
    html.Div(id='tabs-content')
])

# 定義頁面回調
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'tab-realtime':
        return render_realtime_tab()
    elif tab == 'tab-historical':
        return render_historical_tab()
    elif tab == 'tab-prediction':
        return render_prediction_tab()
    elif tab == 'tab-performance':
        return render_performance_tab()

# 定義即時監控頁面
def render_realtime_tab():
    return html.Div([
        html.H2("即時用電監控"),
        
        # 控制面板
        html.Div([
            html.Label("更新頻率:"),
            dcc.Dropdown(
                id='update-interval-dropdown',
                options=[
                    {'label': '5秒', 'value': 5000},
                    {'label': '10秒', 'value': 10000},
                    {'label': '30秒', 'value': 30000},
                    {'label': '1分鐘', 'value': 60000},
                ],
                value=10000,
                style={'width': '200px'}
            ),
            
            html.Label("顯示時間範圍:"),
            dcc.Dropdown(
                id='time-window-dropdown',
                options=[
                    {'label': '最近30分鐘', 'value': 0.5},
                    {'label': '最近1小時', 'value': 1},
                    {'label': '最近3小時', 'value': 3},
                    {'label': '最近6小時', 'value': 6},
                ],
                value=1,
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 儀表盤區域
        html.Div([
            html.Div([
                dcc.Graph(id='gauge-current-power')
            ], style={'width': '33%'}),
            
            html.Div([
                dcc.Graph(id='gauge-daily-energy')
            ], style={'width': '33%'}),
            
            html.Div([
                dcc.Graph(id='gauge-power-factor')
            ], style={'width': '33%'}),
        ], style={'display': 'flex'}),
        
        # 趨勢圖區域
        html.Div([
            dcc.Graph(id='realtime-trend-chart')
        ]),
        
        # 熱圖區域
        html.Div([
            dcc.Graph(id='power-heatmap')
        ]),
        
        # 異常檢測區域
        html.Div([
            html.H3("異常檢測", style={'textAlign': 'center'}),
            html.Div(id='anomaly-alerts', style={'margin': '20px', 'padding': '10px'})
        ]),
        
        # 定時更新組件
        dcc.Interval(
            id='interval-component',
            interval=10*1000,  # 10秒
            n_intervals=0
        )
    ])

# 定義歷史分析頁面
def render_historical_tab():
    return html.Div([
        html.H2("歷史用電分析"),
        
        # 控制面板
        html.Div([
            html.Label("分析類型:"),
            dcc.Dropdown(
                id='analysis-type-dropdown',
                options=[
                    {'label': '日用電量比較', 'value': 'daily'},
                    {'label': '週用電量比較', 'value': 'weekly'},
                    {'label': '月用電量比較', 'value': 'monthly'},
                ],
                value='daily',
                style={'width': '200px'}
            ),
            
            html.Label("日期範圍:"),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                style={'width': '400px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 比較圖區域
        html.Div([
            dcc.Graph(id='comparison-chart')
        ]),
        
        # 統計圖區域
        html.Div([
            dcc.Graph(id='statistics-chart')
        ]),
        
        # 分解圖區域
        html.Div([
            dcc.Graph(id='decomposition-chart')
        ]),
    ])

# 定義預測結果頁面
def render_prediction_tab():
    return html.Div([
        html.H2("用電預測結果"),
        
        # 控制面板
        html.Div([
            html.Label("預測模型:"),
            dcc.Dropdown(
                id='model-dropdown',
                options=[
                    {'label': 'ARIMA', 'value': 'arima'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'LSTM', 'value': 'lstm'},
                    {'label': '所有模型', 'value': 'all'},
                ],
                value='xgboost',
                style={'width': '200px'}
            ),
            
            html.Label("預測視野:"),
            dcc.Dropdown(
                id='horizon-dropdown',
                options=[
                    {'label': '短期 (1小時)', 'value': 'short'},
                    {'label': '中期 (1天)', 'value': 'medium'},
                    {'label': '長期 (1週)', 'value': 'long'},
                ],
                value='short',
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 預測比較圖區域
        html.Div([
            dcc.Graph(id='prediction-comparison-chart')
        ]),
        
        # 誤差分析圖區域
        html.Div([
            dcc.Graph(id='error-analysis-chart')
        ]),
        
        # 多模型比較圖區域
        html.Div([
            dcc.Graph(id='multi-model-comparison-chart')
        ]),
        
        # 預測視野比較圖區域
        html.Div([
            dcc.Graph(id='horizon-comparison-chart')
        ]),
        
        # 預測偏差警報區域
        html.Div([
            html.H3("預測偏差警報", style={'textAlign': 'center'}),
            html.Div(id='prediction-deviation-alerts', style={'margin': '20px', 'padding': '10px'})
        ]),
    ])

# 定義模型性能頁面
def render_performance_tab():
    return html.Div([
        html.H2("模型性能評估"),
        
        # 控制面板
        html.Div([
            html.Label("評估指標:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'RMSE (均方根誤差)', 'value': 'rmse'},
                    {'label': 'MAE (平均絕對誤差)', 'value': 'mae'},
                    {'label': 'MAPE (平均絕對百分比誤差)', 'value': 'mape'},
                    {'label': 'R² (決定係數)', 'value': 'r2'},
                    {'label': '方向準確率', 'value': 'direction_accuracy'},
                ],
                value='rmse',
                style={'width': '300px'}
            ),
            
            html.Label("預測視野:"),
            dcc.Dropdown(
                id='performance-horizon-dropdown',
                options=[
                    {'label': '短期 (1小時)', 'value': 'short'},
                    {'label': '中期 (1天)', 'value': 'medium'},
                    {'label': '長期 (1週)', 'value': 'long'},
                    {'label': '所有視野', 'value': 'all'},
                ],
                value='all',
                style={'width': '200px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'margin': '20px'}),
        
        # 模型性能比較圖區域
        html.Div([
            dcc.Graph(id='model-performance-chart')
        ]),
        
        # 預測視野性能圖區域
        html.Div([
            dcc.Graph(id='horizon-performance-chart')
        ]),
        
        # 特徵重要性圖區域
        html.Div([
            dcc.Graph(id='feature-importance-chart')
        ]),
        
        # 交叉驗證性能圖區域
        html.Div([
            dcc.Graph(id='cv-performance-chart')
        ]),
    ])

# 啟動應用
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
```

### 6.2 Docker部署配置

```dockerfile
# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用代碼
COPY . .

# 暴露端口
EXPOSE 8050

# 啟動應用
CMD ["python", "app.py"]
```

```
# requirements.txt

dash==2.9.3
dash-bootstrap-components==1.4.1
pandas==1.5.3
numpy==1.24.3
plotly==5.14.1
statsmodels==0.14.0
scikit-learn==1.2.2
xgboost==1.7.5
tensorflow==2.12.0
prophet==1.1.4
gunicorn==20.1.0
```

```yaml
# docker-compose.yml

version: '3'

services:
  dashboard:
    build: .
    ports:
      - "8050:8050"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### 6.3 部署指令

```bash
# 構建並啟動儀表板
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止儀表板
docker-compose down
```

## 7. 儀表板使用指南

### 7.1 即時監控頁面

- **功能**：顯示當前用電情況、即時趨勢和異常檢測
- **使用方法**：
  - 調整更新頻率和時間窗口
  - 查看儀表盤了解當前用電狀態
  - 監控趨勢圖了解近期用電變化
  - 查看熱圖了解用電模式
  - 關注異常警報區域的提示

### 7.2 歷史分析頁面

- **功能**：分析歷史用電數據、比較不同時期用電量
- **使用方法**：
  - 選擇分析類型（日、週、月）
  - 設定日期範圍
  - 查看比較圖了解不同時期用電差異
  - 查看統計圖了解用電統計特性
  - 查看分解圖了解用電的趨勢和季節性

### 7.3 預測結果頁面

- **功能**：顯示不同模型的預測結果和誤差分析
- **使用方法**：
  - 選擇預測模型和預測視野
  - 查看預測比較圖了解預測vs實際值
  - 查看誤差分析圖了解預測誤差
  - 比較不同模型的預測結果
  - 比較不同預測視野的結果
  - 關注預測偏差警報

### 7.4 模型性能頁面

- **功能**：評估和比較不同模型的性能
- **使用方法**：
  - 選擇評估指標和預測視野
  - 查看模型性能比較圖了解不同模型的表現
  - 查看預測視野性能圖了解模型在不同視野的表現
  - 查看特徵重要性圖了解影響預測的關鍵因素
  - 查看交叉驗證性能圖了解模型的穩定性
