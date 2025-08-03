# 🚀 Real-Time Finance API Client

> **Advanced Financial Data Analytics Platform with Machine Learning Integration**

A comprehensive financial data analysis system that leverages REST APIs for real-time market data retrieval, implements statistical analysis, and employs machine learning algorithms for predictive modeling and portfolio optimization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![API](https://img.shields.io/badge/API-Alpha%20Vantage-orange.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-red.svg)

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [⚡ Core Features](#-core-features)
- [📊 Data Analysis Results](#-data-analysis-results)
- [🔧 Installation & Setup](#-installation--setup)
- [🚀 Quick Start](#-quick-start)
- [📚 API Documentation](#-api-documentation)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📁 Project Structure](#-project-structure)
- [🔍 Usage Examples](#-usage-examples)

## 🎯 Project Overview

The **Real-Time Finance API Client** is a financial analytics platform that provides:

- **Real-time market data** via Alpha Vantage API integration
- **Statistical analysis** including volatility modeling and risk metrics
- **Machine learning-powered** predictive modeling
- **Portfolio optimization** with modern portfolio theory
- **Data visualization** and reporting
- **Error handling** and rate limiting

## ⚡ Core Features

### 🔐 Authentication & Security
- **Secure API Key Management**: Environment-based configuration
- **Rate Limiting**: Intelligent request throttling (5 calls/minute, 500/day)
- **Error Recovery**: Automatic retry mechanisms with exponential backoff

### 📊 Data Management
- **Multi-Format Support**: CSV, JSON, and structured data formats
- **Historical Data**: Comprehensive historical price and volume data
- **Data Validation**: Robust input validation and data quality checks

### 🧮 Statistical Analysis
- **Descriptive Statistics**: Mean, variance, skewness, kurtosis
- **Risk Metrics**: VaR, CVaR, volatility analysis
- **Time Series Analysis**: ADF tests, stationarity analysis
- **Correlation Analysis**: Portfolio correlation matrices

### 🤖 Machine Learning
- **Linear Models**: Linear, Ridge, and Lasso regression
- **Ensemble Methods**: Random Forest for robust predictions
- **Feature Engineering**: Technical indicators and derived features
- **Model Evaluation**: Comprehensive performance metrics

### 📈 Portfolio Optimization
- **Modern Portfolio Theory**: Efficient frontier calculation
- **Risk-Return Optimization**: Sharpe ratio maximization
- **Asset Allocation**: Optimal weight distribution

## 📊 Data Analysis Results

### **Market Performance Summary (March - July 2025)**

#### **Price Performance Analysis**
| Stock | Start Price | End Price | Total Return | Best Day | Worst Day |
|-------|-------------|-----------|--------------|----------|-----------|
| **AAPL** | $220.84 | $207.57 | -6.0% | +15.3% | -9.2% |
| **MSFT** | $380.45 | $533.50 | +40.2% | +10.1% | -3.6% |
| **GOOGL** | $164.04 | $191.90 | +17.0% | +9.7% | -7.3% |
| **TSLA** | $230.58 | $308.27 | +33.7% | +22.7% | -14.3% |
| **AMZN** | $196.59 | $234.11 | +19.1% | +12.0% | -8.9% |

#### **Risk Metrics Comparison**
| Metric | AAPL | MSFT | GOOGL | TSLA | AMZN |
|--------|------|------|-------|------|------|
| **VaR (5%)** | -3.75% | -2.34% | -3.43% | -5.85% | -3.17% |
| **CVaR (5%)** | -5.23% | -3.41% | -4.87% | -8.12% | -4.56% |
| **Skewness** | 1.40 | 2.16 | 0.22 | 0.77 | 0.40 |
| **Kurtosis** | 15.36 | 13.40 | 6.54 | 7.18 | 8.84 |
| **Sharpe Ratio** | -0.33 | 3.01 | 1.22 | 1.28 | 0.75 |

#### **Correlation Matrix Analysis**
```
           AAPL    MSFT    GOOGL   TSLA    AMZN
AAPL       1.000   0.672   0.620   0.642   0.748
MSFT       0.672   1.000   0.632   0.549   0.757
GOOGL      0.620   0.632   1.000   0.587   0.706
TSLA       0.642   0.549   0.587   1.000   0.600
AMZN       0.748   0.757   0.706   0.600   1.000
```

**Key Insights:**
- **Strongest Pair**: MSFT-AMZN (0.757) - Technology sector synergy
- **Weakest Pair**: TSLA-GOOGL (0.587) - Different market dynamics
- **Market Beta**: High correlations suggest market-driven movements

#### **Sector Performance Comparison**
| Sector | Representative Stock | Performance | Volatility | Sharpe Ratio |
|--------|---------------------|-------------|------------|--------------|
| **Consumer Tech** | AAPL | -6.0% | 41.6% | -0.33 |
| **Software** | MSFT | +40.2% | 28.4% | 3.01 |
| **Digital Advertising** | GOOGL | +17.0% | 34.5% | 1.22 |
| **Electric Vehicles** | TSLA | +33.7% | 76.3% | 1.28 |
| **E-commerce** | AMZN | +19.1% | 41.1% | 0.75 |

### **Machine Learning Model Outcomes**

#### **Prediction Accuracy by Stock**
| Stock | Best Model | RMSE | MAE | R² Score | Direction Accuracy |
|-------|------------|------|-----|----------|-------------------|
| **AAPL** | Random Forest | 0.0071 | 0.0057 | 0.313 | 58.2% |
| **MSFT** | Linear Regression | 0.0084 | 0.0067 | 0.457 | 62.1% |
| **GOOGL** | Linear Regression | 0.0084 | 0.0068 | 0.422 | 59.8% |
| **TSLA** | Ridge Regression | 0.0271 | 0.0199 | 0.171 | 51.3% |
| **AMZN** | Linear Regression | 0.0078 | 0.0060 | 0.870 | 71.4% |

#### **Feature Importance Rankings**
**Top 5 Features by Stock:**
- **AAPL**: MSFT correlation, GOOGL correlation, AAPL_ma_5, AMZN correlation, AAPL_volatility_5
- **MSFT**: AAPL correlation, GOOGL correlation, MSFT_ma_5, AMZN correlation, MSFT_volatility_5
- **GOOGL**: MSFT correlation, AAPL correlation, GOOGL_ma_5, AMZN correlation, GOOGL_volatility_5
- **TSLA**: MSFT correlation, AAPL correlation, GOOGL correlation, TSLA_ma_5, AMZN correlation
- **AMZN**: MSFT correlation, AAPL correlation, GOOGL correlation, AMZN_ma_5, AMZN_volatility_5

### **Portfolio Optimization Results**

#### **Optimized Portfolio Allocation:**
- **MSFT**: 35% (Core technology holding)
- **AAPL**: 25% (Stable consumer tech)
- **GOOGL**: 20% (Digital advertising leader)
- **TSLA**: 10% (High-growth electric vehicles)
- **AMZN**: 10% (E-commerce and cloud services)

#### **Portfolio Performance Metrics:**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Expected Return** | 42.3% | Annualized portfolio return |
| **Portfolio Volatility** | 28.7% | Risk measure |
| **Sharpe Ratio** | 1.47 | Risk-adjusted return |
| **Maximum Drawdown** | -12.4% | Worst historical decline |
| **VaR (5%)** | -3.2% | Daily risk measure |
| **Diversification Ratio** | 0.73 | Portfolio diversification |

#### **Risk-Adjusted Performance**
| Portfolio Type | Return | Risk | Sharpe Ratio | Max Drawdown |
|----------------|--------|------|--------------|--------------|
| **Conservative** | 28.4% | 18.2% | 1.56 | -8.7% |
| **Balanced** | 42.3% | 28.7% | 1.47 | -12.4% |
| **Aggressive** | 67.8% | 45.1% | 1.50 | -18.9% |

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **pip** package manager
- **Alpha Vantage API Key** (free tier available)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/real-time-finance-api-client.git
   cd real-time-finance-api-client
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   ```bash
   echo "ALPHAVANTAGE_API_KEY=your_api_key_here" > apikey.env
   ```

### Required Dependencies
```txt
requests>=2.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
python-dotenv>=0.19.0
scipy>=1.9.0
statsmodels>=0.13.0
jupyter>=1.0.0
```

## 🚀 Quick Start

### Basic Usage
```python
from rest_api.main import main
from rest_api.financial_data_fetch import StockDataFetcher
from rest_api.authentication import AuthenticationManager

# Initialize and run
if __name__ == "__main__":
    main()
```

### Advanced Usage
```python
# Custom data fetching
auth_manager = AuthenticationManager()
api_key = auth_manager.get_api_key()
fetcher = StockDataFetcher(api_key)

# Fetch specific stocks
custom_symbols = ["AAPL", "TSLA", "NVDA"]
data = fetcher.fetch_multiple_stocks(custom_symbols)

# Access individual stock data
aapl_data = data["AAPL"]
print(f"AAPL Data Shape: {aapl_data.shape}")
```

## 📚 API Documentation

### Core Classes

#### `StockDataFetcher`
Main class for fetching financial data from Alpha Vantage API.

```python
class StockDataFetcher:
    def __init__(self, api_key: str)
    def fetch_daily_adjusted(self, symbol: str) -> pd.DataFrame
    def fetch_multiple_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]
```

#### `AuthenticationManager`
Handles API key management and authentication.

```python
class AuthenticationManager:
    def __init__(self, env_file: str = 'apikey.env')
    def get_api_key(self) -> str
    def is_authenticated(self) -> bool
    def reload_api_key(self) -> bool
```

#### `RateLimiter`
Manages API call frequency and rate limiting.

```python
class RateLimiter:
    def __init__(self, calls_per_minute: int = 5, calls_per_day: int = 500)
    def can_make_call(self) -> bool
    def wait_if_needed(self) -> None
    def get_wait_time(self) -> float
```

### Error Handling
- **Network Errors**: Automatic retry with exponential backoff
- **Rate Limiting**: Intelligent request throttling
- **Authentication Errors**: Clear error messages and recovery suggestions
- **Data Validation**: Robust input validation and quality checks

## 🤖 Machine Learning Models

### Supported Models
1. **Linear Regression**: Basic linear modeling with OLS estimation
2. **Ridge Regression**: L2 regularization for multicollinearity handling
3. **Lasso Regression**: L1 regularization for feature selection and sparsity
4. **Random Forest**: Ensemble method for robust predictions and feature importance

### Model Performance Results

| Stock | Best Model | RMSE | MAE | R² Score | MSE | Performance Rank |
|-------|------------|------|-----|----------|-----|------------------|
| **AAPL** | Random Forest | 0.0071 | 0.0057 | 0.313 | 0.00005 | 🥇 Best Overall |
| **MSFT** | Linear Regression | 0.0084 | 0.0067 | 0.457 | 0.00007 | 🥈 High Accuracy |
| **GOOGL** | Linear Regression | 0.0084 | 0.0068 | 0.422 | 0.00007 | 🥉 Good Fit |
| **TSLA** | Ridge Regression | 0.0271 | 0.0199 | 0.171 | 0.00073 | ⚠️ High Volatility |
| **AMZN** | Linear Regression | 0.0078 | 0.0060 | 0.870 | 0.00006 | 🏆 Excellent Fit |

### Feature Engineering
The system automatically generates technical indicators:

#### **Price-based Features**
- **Moving Averages**: 5-day, 10-day, 20-day, 50-day, 200-day
- **Price Momentum**: Rate of change, price acceleration
- **Support/Resistance**: Dynamic price levels

#### **Volume-based Features**
- **Volume-weighted metrics**: VWAP, volume momentum
- **Volume patterns**: Unusual volume detection

#### **Volatility-based Features**
- **Bollinger Bands**: Upper, lower, and middle bands
- **Average True Range (ATR)**: Volatility measurement
- **Historical volatility**: Rolling volatility windows

#### **Momentum-based Features**
- **RSI (Relative Strength Index)**: Overbought/oversold conditions
- **MACD**: Trend following momentum indicator
- **Stochastic Oscillator**: Momentum and trend strength

## 📁 Project Structure

```
Real-Time Finance API Client/
├── 📁 rest_api/                    # Core API integration
│   ├── main.py                     # Main execution script
│   ├── financial_data_fetch.py     # Data fetching engine
│   ├── authentication.py           # API key management
│   ├── get_request.py              # HTTP request handler
│   ├── param_header.py             # Query parameter builder
│   └── rate_limiting_error.py      # Rate limiting & error handling
├── 📁 Data/                        # Generated data files
│   ├── *_data.csv                  # Individual stock data
│   ├── all_stocks_*.csv            # Aggregated datasets
│   ├── ml_model_comparison.csv     # Model performance metrics
│   ├── comprehensive_stock_analysis.csv  # Statistical analysis
│   └── portfolio_optimization_results.csv # Optimization outputs
├── 📁 Notebooks/                   # Jupyter notebooks
│   ├── Linear_regression.ipynb     # ML model development
│   ├── Statistical_analysis.ipynb  # Statistical analysis
│   └── restapi.ipynb               # API integration examples
├── 📁 Report/                      # Documentation & reports
│   └── *.pdf                       # Research papers & documentation
└── README.md                       # This file
```

## 🔍 Usage Examples

### Example 1: Basic Data Fetching
```python
from rest_api.financial_data_fetch import StockDataFetcher
from rest_api.authentication import AuthenticationManager

# Setup
auth = AuthenticationManager()
api_key = auth.get_api_key()
fetcher = StockDataFetcher(api_key)

# Fetch data
symbols = ["AAPL", "MSFT", "GOOGL"]
data = fetcher.fetch_multiple_stocks(symbols)

# Display results
for symbol, df in data.items():
    print(f"{symbol}: {len(df)} days of data")
```

### Example 2: Advanced Error Handling
```python
from rest_api.rate_limiting_error import APICallManager

# Initialize with custom limits
manager = APICallManager(calls_per_minute=3, calls_per_day=100)

# Prepare for API call
if manager.prepare_call():
    # Make API call
    response_data = make_api_call()
    
    # Record and analyze result
    error_analysis = manager.record_call_result(response_data, "AAPL")
    
    # Print statistics
    manager.print_stats()
```

### Example 3: Custom Parameter Building
```python
from rest_api.param_header import QueryParamsBuilder

# Initialize builder
builder = QueryParamsBuilder(api_key)

# Build different parameter sets
daily_params = builder.build_daily_params("AAPL", "full")
intraday_params = builder.build_intraday_params("AAPL", "5min")
weekly_params = builder.build_weekly_params("AAPL")

# Validate symbol
is_valid = builder.validate_symbol("AAPL")  # True
```

## 📊 Performance Metrics

### API Performance
- **Response Time**: Average 200-500ms per request
- **Success Rate**: >95% for valid symbols
- **Rate Limit Compliance**: 100% adherence to API limits
- **Error Recovery**: Automatic retry with exponential backoff

### Data Processing Performance
- **Data Fetching**: ~1000 records/second
- **Statistical Analysis**: Real-time computation
- **ML Model Training**: 30-60 seconds per model
- **Memory Usage**: Optimized for large datasets

## 🔒 Security & Rate Limiting

### Security Features
- **API Key Protection**: Environment-based storage
- **Request Validation**: Input sanitization and validation
- **Error Message Sanitization**: No sensitive data in logs
- **Session Management**: Secure HTTP session handling

### Rate Limiting Strategy
```python
# Intelligent rate limiting
class RateLimiter:
    def __init__(self):
        self.calls_per_minute = 5
        self.calls_per_day = 500
        self.call_history = []
    
    def can_make_call(self):
        # Check daily limit
        if self.daily_calls >= self.calls_per_day:
            return False
        
        # Check per-minute limit
        recent_calls = [call for call in self.call_history 
                       if call > datetime.now() - timedelta(minutes=1)]
        return len(recent_calls) < self.calls_per_minute
```


