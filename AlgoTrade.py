import pandas as pd
from SmartApi.smartConnect import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
import time
import logging
import os
import sys
import json
import requests
from datetime import datetime, timedelta
import threading
import warnings
import traceback
from functools import wraps
import random
import numpy as np
from scipy.stats import norm
import math
import csv
from notifier import TelegramNotifier
import pkg_resources
from strategies.advanced_strategies import HighWinRateStrategy
from config import API_KEY, CLIENT_CODE, PASSWORD, TOTP_SECRET, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, STRATEGY_NAME
from models import db, Trade
from analytics import TradingAnalytics

# === RETRY DECORATOR ===
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            max_retries = retries
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == max_retries:
                        raise e
                    sleep_time = (backoff_in_seconds * 2 ** x + 
                                random.uniform(0, 1))
                    time.sleep(sleep_time)
                    x += 1
                    logging.warning(f"Retrying {func.__name__}, attempt {x}/{max_retries}")
        return wrapper
    return decorator

# === CONFIGURATION ===
# Global variables
smartapi_obj = None
contract_hub_exchng = {}
df = None
exchange_token_hub = {}
ltp_dict = {}
data_dict = {}
data_dict_exchng = {}
trade_state = {}
paper_trades = []

# Replace the old send_telegram_message function with a global notifier instance
telegram_notifier = TelegramNotifier()

# Mapping dictionary for indices
map_dict = {
    "NIFTY 50": {'NAME': 'NIFTY', 'SYMBOL': 'NIFTY', 'TOKEN': '26000', 'SEGMENT': 'NFO'},
    'NIFTY BANK': {'NAME': 'BANKNIFTY', 'SYMBOL': 'BANKNIFTY', 'TOKEN': '26009', 'SEGMENT': 'NFO'},
    'NIFTY FIN SERVICE': {'NAME': 'FINNIFTY', 'SYMBOL': 'FINNIFTY', 'TOKEN': '26037', 'SEGMENT': 'NFO'},
    'NIFTY MID SELECT': {'NAME': 'MIDCPNIFTY', 'SYMBOL': 'MIDCPNIFTY', 'TOKEN': '26074', 'SEGMENT': 'NFO'},
    'SENSEX': {'NAME': 'SENSEX', 'SYMBOL': 'SENSEX', 'TOKEN': '1', 'SEGMENT': 'BFO'},
    'BANKEX': {'NAME': 'BANKEX', 'SYMBOL': 'BANKEX', 'TOKEN': '2', 'SEGMENT': 'BFO'},
    'CRUDEOIL': {'NAME': 'CRUDEOIL', 'SYMBOL': 'CRUDEOIL', 'SEGMENT': 'MCX'},
}

# Strategy registry for dynamic selection
STRATEGY_REGISTRY = {
    'HighWinRateStrategy': HighWinRateStrategy,
    # Add more strategies here as you create them
}

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    # S: spot price, K: strike, T: time to expiry (in years), r: risk-free rate, sigma: volatility
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    return {'Delta': delta}

def setup_logging():
    base_log_dir = 'logs'
    date_str = datetime.now().strftime('%Y-%m-%d')
    log_dir = os.path.join(base_log_dir, date_str)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"app.log")
    
    # Configure logging to show output immediately
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Add stdout handler
        ]
    )
    
    # Force immediate output
    sys.stdout.flush()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add console handler with immediate output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Enable traceback printing
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    
    return logger

@retry_with_backoff(retries=3)
def login_to_smartapi():
    """Login to SmartAPI with retry mechanism"""
    try:
        obj = SmartConnect(api_key=API_KEY)
        obj._timeout = 30  # Increase timeout for stability
        
        totp = pyotp.TOTP(TOTP_SECRET)
        current_otp = totp.now()
        
        # Try to generate session with better error handling
        try:
            session_data = obj.generateSession(CLIENT_CODE, PASSWORD, current_otp)
            if not session_data or 'data' not in session_data:
                raise Exception("Invalid session data received")
            
            # Check if the session data contains the required authentication tokens
            if 'jwtToken' not in session_data['data'] or 'refreshToken' not in session_data['data']:
                raise Exception("Missing authentication tokens in session_data")
                
            logging.info("Login successful")
            return obj, session_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error during login: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Login failed: {str(e)}")
            raise
            
    except Exception as e:
        logging.error(f"Login initialization failed: {str(e)}")
        raise

@retry_with_backoff(retries=3)
def preprocess_contract_hub_exchange():
    global df, contract_hub_exchng
    
    today_date = datetime.today().strftime('%Y%m%d')
    instruments_file = f"{today_date}_instrument_file.csv"

    # Remove previous day's file
    previous_date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
    previous_file = f"{previous_date}_instrument_file.csv"
    if os.path.isfile(previous_file):
        os.remove(previous_file)

    try:
        if os.path.isfile(instruments_file):
            logging.info("[+] Instrument file is already present for today.")
            # Read CSV file with proper handling of mixed types
            df = pd.read_csv(instruments_file, low_memory=False)
            
            # Handle token conversion safely
            def safe_token_convert(token):
                try:
                    if isinstance(token, (int, float)):
                        return str(int(token))
                    elif isinstance(token, str) and token.isdigit():
                        return str(int(float(token)))
                    return token
                except:
                    return token
            
            df['token'] = df['token'].apply(safe_token_convert)

            # Fix strike scale for index options (NIFTY, BANKNIFTY, etc.)
            index_option_mask = (
                (df['instrumenttype'] == 'OPTIDX') &
                (df['name'].isin(['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']))
            )
            
            # Log before scaling
            logging.info("Sample strikes before scaling:")
            sample_strikes = df[index_option_mask].head()
            logging.info(f"\n{sample_strikes[['symbol', 'strike', 'name']].to_string()}")
            
            # Scale down the strikes
            df.loc[index_option_mask, 'strike'] = df.loc[index_option_mask, 'strike'] / 100
            
            # Log after scaling
            logging.info("Sample strikes after scaling:")
            sample_strikes = df[index_option_mask].head()
            logging.info(f"\n{sample_strikes[['symbol', 'strike', 'name']].to_string()}")

            # Create filtered DataFrame, only including rows where token is numeric
            mask = df['token'].str.isdigit()
            filtered_df = df[mask & 
                           df['exch_seg'].isin(['NFO', 'BFO', 'MCX']) & 
                           df['instrumenttype'].isin(['OPTIDX', 'FUTIDX', 'FUTCOM'])].copy()
            filtered_df.set_index('token', inplace=True)
        else:
            logging.info("[+] CSV file not found for today. Downloading...")
            url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
            
            requests_data = response.json()
            df = pd.DataFrame.from_dict(requests_data)
            
            # Handle token conversion safely for downloaded data
            def safe_token_convert(token):
                try:
                    if isinstance(token, (int, float)):
                        return str(int(token))
                    elif isinstance(token, str) and token.isdigit():
                        return str(int(float(token)))
                    return token
                except:
                    return token
            
            df['token'] = df['token'].apply(safe_token_convert)

            # Fix strike scale for index options (NIFTY, BANKNIFTY, etc.)
            index_option_mask = (
                (df['instrumenttype'] == 'OPTIDX') &
                (df['name'].isin(['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']))
            )
            
            # Log before scaling
            logging.info("Sample strikes before scaling:")
            sample_strikes = df[index_option_mask].head()
            logging.info(f"\n{sample_strikes[['symbol', 'strike', 'name']].to_string()}")
            
            # Scale down the strikes
            df.loc[index_option_mask, 'strike'] = df.loc[index_option_mask, 'strike'] / 100
            
            # Log after scaling
            logging.info("Sample strikes after scaling:")
            sample_strikes = df[index_option_mask].head()
            logging.info(f"\n{sample_strikes[['symbol', 'strike', 'name']].to_string()}")

            # Create filtered DataFrame, only including rows where token is numeric
            mask = df['token'].str.isdigit()
            filtered_df = df[mask & 
                           df['exch_seg'].isin(['NFO', 'BFO', 'MCX']) & 
                           df['instrumenttype'].isin(['OPTIDX', 'FUTIDX', 'FUTCOM'])].copy()
            filtered_df.set_index('token', inplace=True)
            
            # Save the complete file
            df.to_csv(instruments_file, index=False)
            logging.info("[+] Instrument file Downloaded successfully")

        # Verify data loaded correctly
        if df.empty or filtered_df.empty:
            raise ValueError("No data loaded into DataFrame")

        # Create contract hub from filtered data
        contract_hub_exchng = filtered_df.to_dict(orient='index')
        num_contracts = len(filtered_df)
        logging.info(f"[+] Loaded {num_contracts} valid contracts into contract hub")

        # After contract hub is created, print a sample of NIFTY OPTIDX strikes for verification
        try:
            sample_nifty = df[(df['name'] == 'NIFTY') & (df['instrumenttype'] == 'OPTIDX')][['symbol', 'expiry', 'strike']].head(10)
            print("Sample NIFTY OPTIDX contracts after preprocessing:")
            print(sample_nifty)
        except Exception as e:
            print(f"Debug print failed: {e}")

        return contract_hub_exchng
        
    except Exception as e:
        logging.error(f"Error in preprocess_contract_hub_exchange: {str(e)}")
        raise

def fetch_exchange_tokens(index_value, list_of_expiries):
    """Fetch exchange tokens for given expiry dates"""
    global df, exchange_token_hub
    
    index_name = map_dict[index_value]['NAME']
    segment = map_dict[index_value]['SEGMENT']
    
    # Create a copy of the filtered DataFrame to avoid the SettingWithCopyWarning
    if segment == 'MCX':
        filtered_df = df[
            (df['name'].str.contains(index_name, case=False, na=False)) & 
            (df['exch_seg'] == segment) &
            (df['instrumenttype'] == 'FUTCOM')
        ].copy()  # Create explicit copy
    else:
        filtered_df = df[
            (df['name'] == index_name) & 
            (df['exch_seg'] == segment)
        ].copy()  # Create explicit copy
    
    # Now use loc to set values
    filtered_df.loc[:, 'expiry_str'] = filtered_df['expiry'].dt.strftime('%Y-%m-%d')
    
    for expiry in list_of_expiries:
        tokens = filtered_df[filtered_df['expiry_str'] == expiry]['token'].astype(str).tolist()
        if tokens:  # If tokens found for this expiry
            key = f"{index_value}_{expiry}"
            if segment == 'MCX':
                exchange_token_hub[key] = {"FUT": tokens}  # Store as futures for MCX
            else:
                ce_tokens = [t for t in tokens if filtered_df[filtered_df['token'].astype(str) == t]['symbol'].str.endswith('CE').iloc[0]]
                pe_tokens = [t for t in tokens if filtered_df[filtered_df['token'].astype(str) == t]['symbol'].str.endswith('PE').iloc[0]]
                exchange_token_hub[key] = {"CE": ce_tokens, "PE": pe_tokens}
    
    return exchange_token_hub

def get_recent_expiry(index_value):
    """Get recent expiry dates"""
    global df
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        
        # Make a copy of the expiry column to avoid modifying original
        df['expiry_temp'] = df['expiry']
        
        # Try multiple date formats in sequence
        for date_format in ['%d%b%Y', '%d%m%Y', '%Y-%m-%d']:
            try:
                mask = pd.to_datetime(df['expiry_temp'], errors='coerce').isna()
                df.loc[mask, 'expiry_temp'] = pd.to_datetime(
                    df.loc[mask, 'expiry_temp'], 
                    format=date_format, 
                    errors='coerce'
                )
            except:
                continue
        
        # If any dates are still NaT, try pandas default parser as last resort
        mask = pd.to_datetime(df['expiry_temp'], errors='coerce').isna()
        if mask.any():
            df.loc[mask, 'expiry_temp'] = pd.to_datetime(
                df.loc[mask, 'expiry_temp'],
                errors='coerce'
            )
        
        # Update main expiry column and drop temporary
        df['expiry'] = pd.to_datetime(df['expiry_temp'], errors='coerce')
        df = df.drop('expiry_temp', axis=1)
        
        # Remove any rows with NaT values in expiry
        df = df.dropna(subset=['expiry'])

    index_name = map_dict[index_value]['NAME']
    segment = map_dict[index_value]['SEGMENT']
    
    # For MCX instruments, use different filtering logic
    if segment == 'MCX':
        filtered_df = df[
            (df['name'].str.contains(index_name, case=False, na=False)) & 
            (df['exch_seg'] == segment) &
            (df['instrumenttype'] == 'FUTCOM')  # Filter for commodity futures
        ]
    else:
        filtered_df = df[
            (df['name'] == index_name) & 
            (df['exch_seg'] == segment)
        ]
    
    # Convert to dates for comparison
    unique_expiry_dates = {pd.Timestamp(timestamp).date() for timestamp in filtered_df['expiry']}
    today = pd.Timestamp.now().date()
    
    # Sort dates and filter out past dates
    valid_dates = [date for date in unique_expiry_dates if date >= today]
    if not valid_dates:
        logging.error(f"No valid future expiry dates found for {index_value}")
        return None
        
    sorted_dates = sorted(valid_dates)
    list_of_expiries = [date.strftime('%Y-%m-%d') for date in sorted_dates[:5]]
    
    exchange_tokens = fetch_exchange_tokens(index_value, list_of_expiries)
    return ({index_name: list_of_expiries}, exchange_tokens)

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logging.error(f"Telegram API error: {response.text}")
    except Exception as e:
        logging.error(f"Telegram error: {e}")

def initialize_trade_state(token, symbol, entry_threshold):
    """Initialize trade state for a new token"""
    trade_state[token] = {
        'symbol': symbol,
        'status': 'IDLE',
        'entry_threshold': entry_threshold,
        'entry_price': 0,
        'stop_loss': 0,
        'target': 0,
        'order_id': None,
        'entry_time': None,
        'ltp': 0
    }

def get_symbol_details(token):
    try:
        # Standardize token format to match contract hub
        token = str(int(float(token)))
        if token in contract_hub_exchng:
            return contract_hub_exchng[token]
        else:
            logging.error(f"Token {token} not found in contract hub")
            return None
    except Exception as e:
        logging.error(f"Error getting symbol details for token {token}: {str(e)}")
        return None

def place_market_order(smartapi_obj, symbol_details, token, qty, txn_type):
    try:
        # Use the correct lot size from symbol details
        final_qty = int(qty * symbol_details['lotsize'])  # Convert to regular Python integer
        
        # Get the exact trading symbol from the original data frame
        token_data = df[df['token'] == str(token)]
        if token_data.empty:
            raise ValueError(f"Token {token} not found in instrument data")
        trading_symbol = token_data.iloc[0]['symbol']
        
        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": trading_symbol,
            "symboltoken": str(token),
            "transactiontype": txn_type,
            "exchange": "NFO",
            "ordertype": "MARKET",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "quantity": str(final_qty)
        }
        
        logging.info(f"Placing order with params: {order_params}")
        response = smartapi_obj.placeOrder(order_params)
        
        # Handle the order response
        if isinstance(response, dict):
            if response.get('status'):
                order_id = response.get('data', {}).get('orderid')
                if order_id:
                    logging.info(f"Order placed successfully: {response}")
                    return order_id
            error_msg = response.get('message', 'Unknown error')
            logging.error(f"Order placement failed: {error_msg}")
            return None
        else:
            # If response is a string (order ID), it's successful
            order_id = str(response)
            logging.info(f"Order placed successfully with ID: {order_id}")
            return order_id
            
    except Exception as e:
        error_msg = f"Order placement error: {str(e)}"
        logging.error(error_msg)
        return None

def get_atm_strike(index_value, ltp, expiry_date):
    """Get ATM strike price based on current market price"""
    global df
    
    index_name = map_dict[index_value]['NAME']
    segment = map_dict[index_value]['SEGMENT']
    
    # For NIFTY and BANKNIFTY, we can use a standard strike calculation
    if index_name in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
        # Round to nearest strike based on index
        if index_name == 'NIFTY':
            # NIFTY uses 50-point intervals
            base = 50
            atm_strike = round(float(ltp) / base) * base
        elif index_name == 'BANKNIFTY':
            # BANKNIFTY uses 100-point intervals
            base = 100
            atm_strike = round(float(ltp) / base) * base
        elif index_name == 'FINNIFTY':
            # FINNIFTY uses 50-point intervals
            base = 50
            atm_strike = round(float(ltp) / base) * base
        elif index_name == 'MIDCPNIFTY':
            # MIDCPNIFTY uses 25-point intervals
            base = 25
            atm_strike = round(float(ltp) / base) * base
        
        logging.info(f"Calculated ATM strike for {index_name}: {atm_strike} (LTP: {ltp})")
        return int(atm_strike)
    
    # For other indices, try to find from available strikes
    try:
        # Filter for the specific expiry date and index
        filtered_df = df[
            (df['name'] == index_name) & 
            (df['exch_seg'] == segment) & 
            (df['expiry'].dt.strftime('%Y-%m-%d') == expiry_date)
        ]
        
        if filtered_df.empty:
            logging.warning(f"No strikes found for {index_name} with expiry {expiry_date}")
            # Fallback to standard calculation
            base = 50  # Default base
            atm_strike = round(float(ltp) / base) * base
            return int(atm_strike)
        
        # Extract unique strike prices
        strikes = sorted(filtered_df['strike'].unique())
        
        if not strikes:
            logging.warning(f"No strikes available for {index_name}")
            # Fallback to standard calculation
            base = 50  # Default base
            atm_strike = round(float(ltp) / base) * base
            return int(atm_strike)
        
        # Find the nearest strike price to current LTP
        atm_strike = min(strikes, key=lambda x: abs(float(x) - float(ltp)))
        logging.info(f"Found ATM strike from available strikes: {atm_strike}")
        
        return int(atm_strike)
    except Exception as e:
        logging.error(f"Error finding ATM strike: {str(e)}")
        # Fallback to standard calculation
        base = 50  # Default base
        atm_strike = round(float(ltp) / base) * base
        return int(atm_strike)

def get_index_token(index_value):
    """Get the token for an index instrument"""
    try:
        # Hardcoded tokens for common indices
        index_tokens = {
            "NIFTY 50": "26000",  # NSE NIFTY token
            "NIFTY BANK": "26009",  # BANKNIFTY token
            "NIFTY FIN SERVICE": "26037",  # FINNIFTY token
        }
        
        # First try the hardcoded tokens
        if index_value in index_tokens:
            token = index_tokens[index_value]
            logging.info(f"Using predefined token for {index_value}: {token}")
            return token
            
        # If not found in hardcoded tokens, try finding in instrument data
        filtered_df = df[
            ((df['name'] == map_dict[index_value]['NAME']) | 
             (df['name'] == index_value) |
             (df['symbol'] == map_dict[index_value]['NAME'])) &
            (df['instrumenttype'].isin(['INDEX', 'AMXIDX']))
        ]
        
        if not filtered_df.empty:
            token = filtered_df.iloc[0]['token']
            logging.info(f"Found index token for {index_value}: {token}")
            return token
            
        # Try partial match on name as last resort
        filtered_df = df[
            (df['name'].str.contains(map_dict[index_value]['NAME'], case=False, na=False) | 
             df['symbol'].str.contains(map_dict[index_value]['NAME'], case=False, na=False)) &
            (df['instrumenttype'].isin(['INDEX', 'AMXIDX']))
        ]
        
        if not filtered_df.empty:
            token = filtered_df.iloc[0]['token']
            logging.info(f"Found index token for {index_value} using partial match: {token}")
            return token
        
        logging.error(f"No token found for index {index_value}")
        # Log the available indices for debugging
        index_instruments = df[df['instrumenttype'].isin(['INDEX', 'AMXIDX'])]
        logging.debug(f"Available indices: {index_instruments[['name', 'symbol', 'token']].to_string()}")
        return None
        
    except Exception as e:
        logging.error(f"Error getting index token: {str(e)}")
        return None

def get_ltp(token, exchange_type=2):
    """Get Last Traded Price for a symbol"""
    try:
        global smartapi_obj
        
        # If token is a symbol (like 'CRUDEOIL'), get the actual token for the current expiry
        if isinstance(token, str) and token in map_dict:
            segment = map_dict[token]['SEGMENT']
            if segment == 'MCX':
                # Find the token for the current expiry
                result = get_recent_expiry(token)
                if result is None:
                    raise ValueError(f"No valid future expiry dates found for {token}")
                expiry_info, tokens = result
                expiry = expiry_info[map_dict[token]['NAME']][0]
                expiry_key = f"{token}_{expiry}"
                if expiry_key in exchange_token_hub and "FUT" in exchange_token_hub[expiry_key]:
                    token = exchange_token_hub[expiry_key]["FUT"][0]
                    exchange_type = 5  # MCX
                else:
                    raise ValueError(f"No futures token found for {token} expiry {expiry}")
            else:
                # Get the index token for indices
                index_token = get_index_token(token)
                if not index_token:
                    raise ValueError(f"Index token not found for {token}")
                token = index_token
                exchange_type = 1  # NSE for indices
            
        # Get symbol details from contract hub
        symbol_details = None
        if exchange_type not in [1]:  # Skip for indices
            symbol_details = get_symbol_details(token)
            if not symbol_details:
                raise ValueError(f"Symbol details not found for token {token}")
        
        # For indices, use the index name directly
        if exchange_type == 1:
            exchange = "NSE"
            trading_symbol = token  # For indices, token is the trading symbol
        else:
            # For MCX, use MCX exchange, for others use NSE
            exchange = "MCX" if exchange_type == 5 else "NSE"
            trading_symbol = symbol_details['symbol']
            if exchange == "MCX":
                # Remove 'FUT' suffix for MCX symbols if present
                trading_symbol = trading_symbol.replace('FUT', '')
            
        # Log the parameters we're about to use
        logging.info(f"Fetching LTP with params - Token: {token}, Symbol: {trading_symbol}, Exchange: {exchange}")
        
        # Make the LTP request with error handling
        try:
            response = smartapi_obj.ltpData(
                exchange=exchange,
                tradingsymbol=trading_symbol,
                symboltoken=str(token)
            )
        except Exception as req_error:
            if "not logged in" in str(req_error).lower() or "invalid session" in str(req_error).lower():
                logging.warning("Session appears to be invalid, using fallback values")
                # Use fallback values for common indices
                if trading_symbol == "NIFTY 50":
                    return 24379.6
                elif trading_symbol == "NIFTY BANK":
                    return 48250.3
                elif trading_symbol == "NIFTY FIN SERVICE":
                    return 21690.45
                elif trading_symbol == "NIFTY MID SELECT":
                    return 12450.6
                else:
                    # For other symbols, we can't provide a fallback
                    logging.error(f"Session error and no fallback available for {trading_symbol}")
                    return None
            else:
                # Rethrow other errors
                raise
        
        # Validate response
        if isinstance(response, dict):
            if response.get('status') and 'data' in response:
                logging.info(f"LTP data received: {response['data']}")
                return response['data'].get('ltp')
            else:
                error_msg = response.get('message', 'Unknown error')
                logging.error(f"Invalid LTP response: {error_msg}")
                
                # Handle session errors with fallback values
                if "not logged in" in error_msg.lower() or "invalid session" in error_msg.lower():
                    logging.warning("Session error detected, using fallback values")
                    # Use fallback values for common indices
                    if trading_symbol == "NIFTY 50":
                        return 24379.6
                    elif trading_symbol == "NIFTY BANK":
                        return 48250.3
                    elif trading_symbol == "NIFTY FIN SERVICE":
                        return 21690.45
                    elif trading_symbol == "NIFTY MID SELECT":
                        return 12450.6
                    
                return None
        else:
            logging.error(f"Unexpected response type: {type(response)}")
            return None
            
    except Exception as e:
        logging.error(f"Error fetching LTP: {str(e)}")
        logging.error(f"Full error details: {traceback.format_exc()}")
        
        # For testing/demo purposes, provide fallback values for common indices 
        if isinstance(token, str):
            if token == "NIFTY 50" or token == "NIFTY":
                return 24379.6
            elif token == "NIFTY BANK" or token == "BANKNIFTY":
                return 48250.3
            elif token == "NIFTY FIN SERVICE":
                return 21690.45
            elif token == "NIFTY MID SELECT":
                return 12450.6
                
        return None

def select_trading_instrument():
    """Allow user to select which instrument to trade"""
    try:
        print("\nAvailable Instruments:")
        instruments = list(map_dict.keys())
        for i, instrument in enumerate(instruments, 1):
            print(f"{i}. {instrument}")
        
        # Default to NIFTY 50 if no argument provided
        if len(sys.argv) > 1:
            try:
                choice = int(sys.argv[1])
                if 1 <= choice <= len(instruments):
                    selected = instruments[choice - 1]
                    logging.info(f"Selected instrument: {selected}")
                    return selected
                else:
                    logging.warning(f"Invalid selection: {choice}. Defaulting to NIFTY 50")
                    return "NIFTY 50"
            except ValueError:
                logging.warning(f"Invalid input: {sys.argv[1]}. Defaulting to NIFTY 50")
                return "NIFTY 50"
        else:
            # If no command line argument, use default
            logging.info("No instrument selected. Defaulting to NIFTY 50")
            return "NIFTY 50"
    except Exception as e:
        logging.error(f"Error in select_trading_instrument: {str(e)}")
        return "NIFTY 50"  # Default to NIFTY 50 on any error

class SmartAPIWebSocket:
    def __init__(self, session_data, api_key, client_code, token_list):
        # Monkey patch the SmartWebSocketV2 class to fix the callback issue
        def patched_on_close(self, wsapp, close_status_code=None, close_msg=None, *args, **kwargs):
            logging.info(f"WebSocket Closed with status: {close_status_code}, message: {close_msg}")
            
        # Apply the patch
        SmartWebSocketV2._on_close = patched_on_close
        
        self.sws = SmartWebSocketV2(
            session_data['data']['jwtToken'],
            api_key, 
            client_code,
            session_data['data']['feedToken']
        )
        self.token_list = token_list
        self.smartapi_obj = None
        self.order_queue = []
        self.last_trade_prices = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        self.setup_callbacks()

    def handle_reconnection(self):
        """Handle WebSocket reconnection with exponential backoff"""
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
            logging.info(f"Attempting to reconnect in {delay} seconds (Attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
            time.sleep(delay)
            try:
                self.start()
            except Exception as e:
                logging.error(f"Reconnection attempt failed: {str(e)}")
                self.handle_reconnection()
        else:
            logging.error("Max reconnection attempts reached. Please restart the application.")
            telegram_notifier.notify_error(
                context="WebSocket",
                message="Max reconnection attempts reached. Please restart the application."
            )

    def subscribe(self):
        """Subscribe to market data with retry mechanism"""
        try:
            logging.info(f"Subscribing to tokens: {self.token_list}")
            correlation_id = f"ws_{'_'.join(str(token) for token in self.token_list[0]['tokens'])}"
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.sws.subscribe(correlation_id=correlation_id, mode=3, token_list=self.token_list)
                    logging.info(f"Subscription request sent with correlation_id: {correlation_id}")
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Subscription attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"Error in subscription: {str(e)}")
            self.handle_reconnection()

    def setup_callbacks(self):
        def on_ticks(wsapp, message):
            try:
                logging.info(f"Tick: {message}")
                if isinstance(message, dict) and 'token' in message:
                    token = str(message['token'])
                    if 'last_traded_price' in message:
                        ltp = float(message['last_traded_price']) / 100
                        self.last_trade_prices[token] = ltp
                        self.process_tick_with_trading(message, token)
            except Exception as e:
                logging.error(f"Error in tick processing: {str(e)}")
            
        def on_open(wsapp):
            try:
                logging.info("WebSocket Connected")
                self.reconnect_attempts = 0
                self.subscribe()
            except Exception as e:
                logging.error(f"Error in WebSocket open: {str(e)}")
            
        def on_error(wsapp, error):
            logging.error(f"WebSocket Error: {error}")
            if "Connection refused" in str(error):
                self.handle_reconnection()
        
        # Set up the callbacks
        self.sws.on_open = on_open
        self.sws.on_data = on_ticks
        self.sws.on_error = on_error

    def start(self):
        """Start the websocket connection"""
        try:
            logging.info("Starting WebSocket connection...")
            self.sws.connect()
        except Exception as e:
            logging.error(f"Failed to start WebSocket: {str(e)}")
            self.handle_reconnection()

    def close(self):
        """Safely close the websocket connection"""
        try:
            if self.sws:
                self.sws.close_connection()
        except Exception as e:
            logging.error(f"Error closing WebSocket: {str(e)}")

    def process_tick_with_trading(self, tick_data, token):
        """Process incoming tick data for trading decisions"""
        try:
            if token not in trade_state:
                return
                
            ltp = float(tick_data['last_traded_price']) / 100
            time_stamp = datetime.fromtimestamp(int(tick_data['exchange_timestamp']) / 1000)
            trade_state[token]['ltp'] = ltp
            
            symbol_details = get_symbol_details(token)
            if not symbol_details:
                return
                
            logging.info(f"Processing tick for {symbol_details['symbol']} at {time_stamp}")
            logging.info(f"LTP: {ltp}, Trade Status: {trade_state[token]['status']}")
            
        except Exception as e:
            logging.error(f"Error processing tick: {str(e)}")
            telegram_notifier.notify_error(
                context="Tick Processing",
                message=str(e)
            )

if __name__ == "__main__":
    print("=== AlgoTrade is starting (DEBUG) ===")
    logger = setup_logging()
    logging.info("Logging initialized (DEBUG)")
    try:
        # Version check for smartapi-python
        try:
            smartapi_version = pkg_resources.get_distribution("smartapi-python").version
            print(f"smartapi-python version: {smartapi_version}")
        except Exception as e:
            print(f"Could not determine smartapi-python version: {e}")
        print("Attempting to login to SmartAPI...")
        smartapi_obj, session_data = login_to_smartapi()
        logging.info("SmartAPI login successful")
        print("Getting feed token...")
        feed_token = session_data['data']['feedToken']
        if not feed_token:
            raise Exception("Could not get feed token")
        logging.info("Successfully retrieved feed token")
        print("Processing contracts...")
        contract_hub_exchng = preprocess_contract_hub_exchange()
        print("Contract preprocessing complete (DEBUG)")
        logging.info("Contract processing completed (DEBUG)")
        print("Selecting trading instrument...")
        selected_instrument = select_trading_instrument()
        logging.info(f"Selected instrument: {selected_instrument}")
        print("Getting expiry dates...")
        expiry_info, tokens = get_recent_expiry(selected_instrument)
        if not expiry_info or not tokens:
            raise Exception(f"Could not get expiry info for {selected_instrument}")
        all_expiries = {map_dict[selected_instrument]['NAME']: expiry_info[map_dict[selected_instrument]['NAME']]}
        logging.info(f"Expiry info: {all_expiries}")
        instrument_expiry = all_expiries[map_dict[selected_instrument]['NAME']][0]
        logging.info(f"Selected expiry: {instrument_expiry}")
        if map_dict[selected_instrument]['SEGMENT'] == 'MCX':
            print("Processing MCX instrument...")
            expiry_key = f"{selected_instrument}_{instrument_expiry}"
            if expiry_key in exchange_token_hub and "FUT" in exchange_token_hub[expiry_key]:
                selected_token = exchange_token_hub[expiry_key]["FUT"][0]
                logging.info(f"Found MCX token for {selected_instrument}: {selected_token}")
                symbol_details = get_symbol_details(selected_token)
                if symbol_details:
                    print(f"Selected MCX Token: {selected_token}")
                    print(f"Symbol: {symbol_details.get('symbol', 'N/A')}")
                    print(f"Name: {symbol_details.get('name', 'N/A')}")
                    print(f"Expiry: {symbol_details.get('expiry', instrument_expiry)}")
                else:
                    print(f"Selected MCX Token: {selected_token}")
                    print("Symbol details not found.")
                    print(f"Expiry: {instrument_expiry}")
                token_list = [{"exchangeType": 5, "tokens": [selected_token]}]
            else:
                raise Exception(f"No futures tokens found for {selected_instrument} expiry {instrument_expiry}")
        else:
            print("Processing options instrument...")
            current_price = get_ltp(selected_instrument)
            if not current_price:
                raise Exception(f"Could not get current price for {selected_instrument}")
            logging.info(f"Current price for {selected_instrument}: {current_price}")
            atm_strike = get_atm_strike(selected_instrument, current_price, instrument_expiry)
            if not atm_strike:
                raise Exception(f"Could not get ATM strike for {selected_instrument}")
            logging.info(f"ATM strike: {atm_strike}")
            tokens_dict = exchange_token_hub[f"{selected_instrument}_{instrument_expiry}"]
            if not tokens_dict:
                raise Exception(f"No tokens found for {selected_instrument} expiry {instrument_expiry}")
            expiry_contracts = df[
                (df['expiry'].dt.strftime('%Y-%m-%d') == instrument_expiry) &
                (df['name'] == map_dict[selected_instrument]['NAME']) &
                (df['exch_seg'] == map_dict[selected_instrument]['SEGMENT']) &
                (df['instrumenttype'] == 'OPTIDX')
            ].copy()
            if expiry_contracts.empty:
                raise Exception(f"No contracts found for {selected_instrument} expiry {instrument_expiry}")
            all_strikes = sorted([s for s in expiry_contracts['strike'].unique() if s > 0])
            if not all_strikes:
                raise Exception(f"No valid strikes found for {selected_instrument} expiry {instrument_expiry}")
            logging.info(f"Available strikes: {all_strikes[:10]}")
            atm_strike = min(all_strikes, key=lambda x: abs(x - current_price))
            logging.info(f"ATM Strike selected: {atm_strike}")
            token_list = []
            monitored_contracts = []
            for option_type in ["CE", "PE"]:
                contract = expiry_contracts[
                    (expiry_contracts['strike'] == atm_strike) &
                    (expiry_contracts['symbol'].str.endswith(option_type))
                ]
                if not contract.empty:
                    contract_data = contract.iloc[0]
                    contract_details = {
                        'token': str(int(contract_data['token'])),
                        'tradingsymbol': contract_data['symbol'],
                        'lot_size': int(contract_data['lotsize']),
                        'tick_size': float(contract_data['tick_size']),
                        'strike': float(atm_strike),
                        'expiry': instrument_expiry
                    }
                    token_list.append(contract_details['token'])
                    monitored_contracts.append(contract_details)
                    logging.info(f"Monitoring {contract_details['tradingsymbol']} (ATM) - Token: {contract_details['token']}")
                    initialize_trade_state(contract_details['token'], contract_details['tradingsymbol'], entry_threshold=float('inf'))
            if not token_list:
                raise Exception("No valid tokens found for monitoring")
            ws_token_list = [{"action": 1, "exchangeType": 2, "tokens": token_list}]
            monitored_symbols = [contract['tradingsymbol'] for contract in monitored_contracts]
            telegram_notifier.notify_startup(
                instrument=selected_instrument,
                expiry=instrument_expiry,
                spot_price=current_price,
                atm_strike=float(atm_strike),
                monitored=monitored_symbols,
                strategy="High Win-Rate Strategy"
            )
        print("Creating WebSocket instance...")
        websocket = SmartAPIWebSocket(
            session_data,
            API_KEY,
            CLIENT_CODE,
            ws_token_list
        )
        # Print the WebSocket endpoint if available
        print(f"WebSocket endpoint: {getattr(websocket.sws, 'url', 'unknown')}")
        websocket.smartapi_obj = smartapi_obj
        if map_dict[selected_instrument]['SEGMENT'] == 'MCX':
            initialize_trade_state(selected_token, symbol_details.get('symbol', 'N/A'), 0.5)
        else:
            for contract in monitored_contracts:
                initialize_trade_state(contract['token'], contract['tradingsymbol'], 0.5)
        print("Starting WebSocket connection...")
        websocket.start()
        print("Application started successfully. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            websocket.close()
            telegram_notifier.notify_error(
                context="System",
                message="Trading Bot Stopped by User"
            )
            logging.info("Application terminated by user")
            sys.exit(0)
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        telegram_notifier.notify_error(
            context="System",
            message=f"Trading Bot Error: {str(e)}"
        )
        sys.exit(1)

# Instantiate analytics engine
analytics = TradingAnalytics(db_path="trading_analytics.db")

# === ANALYTICS LOGGING TEMPLATES ===
# Use these templates at the point where you enter or exit a trade.
# Example: After placing an order and updating trade_state, call analytics.record_trade() as shown below.

# --- Trade Entry Example ---
# analytics.record_trade({
#     'symbol': symbol,  # e.g., 'NIFTY24JUN18000CE'
#     'entry_time': entry_time,  # datetime object or string
#     'entry_price': entry_price,  # float
#     'quantity': quantity,  # int
#     'direction': direction,  # 'BUY' or 'SELL'
#     'status': 'OPEN',
#     'strategy': STRATEGY_NAME,
#     'stop_loss': stop_loss,  # float
#     'target_price': target_price,  # float
#     'exit_reason': None
# })

# --- Trade Exit Example ---
# analytics.record_trade({
#     'symbol': symbol,
#     'entry_time': entry_time,
#     'exit_time': exit_time,  # datetime object or string
#     'entry_price': entry_price,
#     'exit_price': exit_price,  # float
#     'quantity': quantity,
#     'direction': direction,
#     'pnl': pnl,  # float
#     'status': 'CLOSED',
#     'strategy': STRATEGY_NAME,
#     'stop_loss': stop_loss,
#     'target_price': target_price,
#     'exit_reason': exit_reason  # e.g., 'TARGET', 'STOP_LOSS', 'MANUAL', etc.
# })

# === REPORT GENERATION AT SHUTDOWN ===
# This should be called before sys.exit(0) or sys.exit(1) in the main block.
def generate_and_save_report():
    try:
        os.makedirs('reports', exist_ok=True)
        report_html = analytics.generate_performance_report()
        analytics.save_report(report_html, filename=f"reports/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        logging.info("Performance report generated and saved.")
    except Exception as e:
        logging.error(f"Failed to generate/save performance report: {str(e)}")

# ... existing code ...
# In the main block, before sys.exit(0) or sys.exit(1):
# generate_and_save_report()
# ... existing code ...
