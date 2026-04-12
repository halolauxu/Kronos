import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

from db import Database
from data_provider import DataProvider, generate_ashare_timestamps
from scanner import ScanTask, FREQ_DEFAULTS

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# A-share modules (lazy initialized)
database = None
data_provider = None
current_scan = None  # Active ScanTask instance


def get_db():
    """Lazy-init database and data provider."""
    global database, data_provider
    if database is None:
        database = Database()
        data_provider = DataProvider(database)
    return database, data_provider


# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor

    # Block model switching during scan
    if current_scan and current_scan.get_state()["status"] == "running":
        return jsonify({'error': 'Cannot switch model while a scan is running'}), 400

    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

# ==================== A-Share Scan API ====================

@app.route('/api/index-list')
def get_index_list():
    """Get available index list with constituent counts."""
    _, dp = get_db()
    indices = dp.get_index_list()
    return jsonify({'indices': indices})


@app.route('/api/scan', methods=['POST'])
def start_scan():
    """Start batch scan."""
    global current_scan
    if current_scan and current_scan.get_state()["status"] == "running":
        return jsonify({'error': 'A scan is already running'}), 400
    if predictor is None:
        return jsonify({'error': 'Model not loaded. Please load a model first.'}), 400

    data = request.get_json()
    index_code = data.get('index_code', '000300')
    freq = data.get('freq', 'daily')
    model_key = data.get('model_key', 'kronos-small')
    params_override = {}
    for key in ['T', 'top_p', 'sample_count', 'lookback', 'pred_len']:
        if key in data:
            params_override[key] = float(data[key]) if key in ('T', 'top_p') else int(data[key])

    db, dp = get_db()
    current_scan = ScanTask(db, dp, predictor, index_code, freq, model_key, params_override)
    current_scan.start()
    return jsonify({'success': True, 'scan_id': current_scan.scan_id, 'message': 'Scan started'})


@app.route('/api/scan/status')
def scan_status():
    """Get scan progress with lightweight results (no pred_data)."""
    if current_scan is None:
        return jsonify({'status': 'idle', 'message': 'No scan running'})
    state = current_scan.get_state()
    if state["completed"] > 0:
        db, _ = get_db()
        results = db.get_scan_results_summary(state["scan_id"], page=1, page_size=1000)
        state["results"] = results
    return jsonify(state)


@app.route('/api/scan/stop', methods=['POST'])
def stop_scan():
    """Stop running scan."""
    if current_scan is None or current_scan.get_state()["status"] != "running":
        return jsonify({'error': 'No scan running'}), 400
    current_scan.stop()
    return jsonify({'success': True, 'message': 'Scan stop requested'})


@app.route('/api/scan/history')
def scan_history():
    """List historical scans."""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))
    db, _ = get_db()
    history = db.get_scan_history(page, page_size)
    return jsonify({'history': history})


@app.route('/api/scan/<scan_id>/results')
def scan_results(scan_id):
    """Get results for a specific scan."""
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 50))
    db, _ = get_db()
    results = db.get_scan_results(scan_id, page, page_size)
    total = db.get_scan_results_count(scan_id)
    return jsonify({'results': results, 'total': total, 'page': page, 'page_size': page_size})


@app.route('/api/predict-symbol', methods=['POST'])
def predict_symbol():
    """Predict a single stock by code."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 400

    data = request.get_json()
    stock_code = data.get('stock_code', '').strip()
    freq = data.get('freq', 'daily')

    if not stock_code:
        return jsonify({'error': 'stock_code is required'}), 400

    params = FREQ_DEFAULTS.get(freq, FREQ_DEFAULTS["daily"]).copy()
    for key in ['T', 'top_p', 'sample_count']:
        if key in data:
            params[key] = float(data[key]) if key in ('T', 'top_p') else int(data[key])

    try:
        _, dp = get_db()
        kline_df = dp.fetch_kline(stock_code, freq)
        if kline_df is None or len(kline_df) < params["lookback"]:
            return jsonify({'error': f'Insufficient data for {stock_code}'}), 400

        lookback = params["lookback"]
        pred_len = params["pred_len"]
        x_df = kline_df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(kline_df.iloc[-lookback:]["dt"].values)

        y_ts = generate_ashare_timestamps(kline_df["dt"].iloc[-1], freq, pred_len)

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=params["T"], top_p=params["top_p"],
            sample_count=params["sample_count"],
        )

        last_close = float(kline_df.iloc[-1]["close"])
        pred_df = DataProvider.apply_price_limits(pred_df, last_close)

        pred_close = float(pred_df.iloc[-1]["close"])
        pred_change_pct = ((pred_close - last_close) / last_close) * 100

        hist_data = kline_df.iloc[-lookback:].to_dict(orient="records")
        for row in hist_data:
            row["dt"] = str(row["dt"])
        pred_data = pred_df.reset_index(drop=True).to_dict(orient="records")
        pred_timestamps = [str(t) for t in y_ts]

        db, _ = get_db()
        stock_name = db.get_stock_name(stock_code)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'freq': freq,
            'last_close': last_close,
            'pred_close': pred_close,
            'pred_change_pct': round(pred_change_pct, 4),
            'historical': hist_data,
            'predictions': pred_data,
            'pred_timestamps': pred_timestamps,
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/backtest-symbol', methods=['POST'])
def backtest_symbol():
    """Backtest: predict a past period and compare with actual data."""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 400

    data = request.get_json()
    stock_code = data.get('stock_code', '').strip()
    freq = data.get('freq', 'daily')

    if not stock_code:
        return jsonify({'error': 'stock_code is required'}), 400

    params = FREQ_DEFAULTS.get(freq, FREQ_DEFAULTS["daily"]).copy()
    for key in ['T', 'top_p', 'sample_count']:
        if key in data:
            params[key] = float(data[key]) if key in ('T', 'top_p') else int(data[key])

    try:
        _, dp = get_db()
        kline_df = dp.fetch_kline(stock_code, freq)
        lookback = params["lookback"]
        pred_len = params["pred_len"]

        if kline_df is None or len(kline_df) < lookback + pred_len:
            return jsonify({'error': f'Insufficient data for backtest (need {lookback + pred_len}, have {len(kline_df) if kline_df is not None else 0})'}), 400

        # Use data up to pred_len bars ago as input, predict pred_len bars, compare with actual
        split_idx = len(kline_df) - pred_len
        hist_df = kline_df.iloc[:split_idx]
        actual_df = kline_df.iloc[split_idx:]

        x_df = hist_df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = pd.Series(hist_df.iloc[-lookback:]["dt"].values)
        y_ts = pd.Series(actual_df["dt"].values)

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=params["T"], top_p=params["top_p"],
            sample_count=params.get("sample_count", 3),
        )

        last_close = float(hist_df.iloc[-1]["close"])
        pred_df = DataProvider.apply_price_limits(pred_df, last_close)

        # Calculate accuracy metrics
        import numpy as np
        actual_closes = actual_df["close"].values.astype(float)
        pred_closes = pred_df["close"].values.astype(float)
        min_len = min(len(actual_closes), len(pred_closes))
        actual_closes = actual_closes[:min_len]
        pred_closes = pred_closes[:min_len]

        mae = float(np.mean(np.abs(actual_closes - pred_closes)))
        mape = float(np.mean(np.abs((actual_closes - pred_closes) / (actual_closes + 1e-8))) * 100)

        # Direction accuracy: did the model predict the correct up/down direction?
        actual_dirs = np.diff(actual_closes) > 0
        pred_dirs = np.diff(pred_closes) > 0
        dir_min = min(len(actual_dirs), len(pred_dirs))
        direction_acc = float(np.mean(actual_dirs[:dir_min] == pred_dirs[:dir_min]) * 100) if dir_min > 0 else 0

        # Overall direction
        actual_change = (actual_closes[-1] - actual_closes[0]) / actual_closes[0] * 100
        pred_change = (pred_closes[-1] - last_close) / last_close * 100
        overall_dir_correct = bool((actual_change > 0) == (pred_change > 0))

        # Prepare response
        hist_recent = hist_df.iloc[-lookback:].to_dict(orient="records")
        for row in hist_recent:
            row["dt"] = str(row["dt"])

        actual_data = actual_df.to_dict(orient="records")
        for row in actual_data:
            row["dt"] = str(row["dt"])

        pred_data = pred_df.reset_index(drop=True).to_dict(orient="records")
        pred_timestamps = [str(t) for t in y_ts]

        db, _ = get_db()
        stock_name = db.get_stock_name(stock_code)

        return jsonify({
            'success': True,
            'stock_code': stock_code,
            'stock_name': stock_name,
            'freq': freq,
            'backtest_start': str(actual_df["dt"].iloc[0]),
            'backtest_end': str(actual_df["dt"].iloc[-1]),
            'historical': hist_recent,
            'predictions': pred_data,
            'pred_timestamps': pred_timestamps,
            'actual': actual_data,
            'metrics': {
                'mae': round(mae, 4),
                'mape': round(mape, 2),
                'direction_accuracy': round(direction_acc, 1),
                'overall_direction_correct': overall_dir_correct,
                'actual_change_pct': round(actual_change, 2),
                'pred_change_pct': round(pred_change, 2),
            },
        })

    except Exception as e:
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500


@app.route('/api/search-stock')
def search_stock():
    """Search stock by code or name in cached constituents."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'results': []})
    db, _ = get_db()
    results = db.search_stocks(query)
    return jsonify({'results': results})


# ==================== A-Share Page Routes ====================

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Stock detail page."""
    freq = request.args.get('freq', 'daily')
    db, _ = get_db()
    prediction = db.get_latest_prediction(symbol, freq)
    return render_template('stock_detail.html', symbol=symbol, freq=freq, prediction=prediction)


@app.route('/search')
def search_page():
    """Search prediction page."""
    return render_template('search.html')


if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=False, host='0.0.0.0', port=7070)
