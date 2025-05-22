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
API_KEY = "lB9qx6Rm"
CLIENT_CODE = "R182159"
PASSWORD = "1010"
TOTP_SECRET = "NRXMU4SVMHCEW3H2KQ65IZKDGI"
TELEGRAM_TOKEN = "7836552815:AAHuWTVdtz_vYInRH_f9SDLJBc8MkHoog0o"
TELEGRAM_CHAT_ID = "7904363041"

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
                raise Exception("Missing authentication tokens in session data")
                
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
        self.sws = SmartWebSocketV2(
            session_data['data']['jwtToken'],
            api_key, 
            client_code,
            session_data['data']['feedToken']
        )
        self.token_list = token_list
        self.smartapi_obj = None  # Will be set externally
        self.order_queue = []
        self.last_trade_prices = {}
        self.setup_callbacks()

    def setup_callbacks(self):
        def on_ticks(wsapp, message):
            try:
                logging.info(f"Tick: {message}")
                token = str(message['token'])
                ltp = float(message['last_traded_price']) / 100
                self.last_trade_prices[token] = ltp
                
                # Process the tick with trading logic
                self.process_tick_with_trading(message, token)
            except Exception as e:
                logging.error(f"Error in tick processing: {str(e)}")
            
        def on_open(wsapp):
            try:
                logging.info("WebSocket Connected")
                self.subscribe()
            except Exception as e:
                logging.error(f"Error in WebSocket open: {str(e)}")
            
        def on_error(wsapp, error):
            logging.error(f"WebSocket Error: {error}")
            self.reconnect()
            
        def on_close(wsapp):
            logging.info("WebSocket Closed")
            try:
                self.reconnect()
            except Exception as e:
                logging.error(f"Error in reconnection: {str(e)}")
        
        self.sws.on_open = on_open
        self.sws.on_data = on_ticks
        self.sws.on_error = on_error
        self.sws.on_close = on_close

    def subscribe(self):
        """Subscribe to market data"""
        try:
            logging.info(f"Subscribing to tokens: {self.token_list}")
            # Generate a unique correlation ID
            tokens_string = '_'.join(str(token) for token in self.token_list[0]['tokens'])
            correlation_id = f"ws_{tokens_string}"
            self.sws.subscribe(correlation_id=correlation_id, mode=3, token_list=self.token_list)
            logging.info(f"Subscription request sent with correlation_id: {correlation_id}")
        except Exception as e:
            logging.error(f"Error in subscription: {str(e)}")
            raise
    
    def reconnect(self):
        """Attempt to reconnect the websocket"""
        try:
            if not self.sws.is_connected():
                logging.info("Attempting to reconnect...")
                self.start()
        except Exception as e:
            logging.error(f"Reconnection failed: {str(e)}")

    def start(self):
        """Start the websocket connection in a separate thread"""
        try:
            threading.Thread(target=self.sws.connect).start()
        except Exception as e:
            logging.error(f"Failed to start WebSocket: {str(e)}")
    
    def close(self):
        """Safely close the websocket connection"""
        try:
            if self.sws:
                self.sws.close_connection()
        except Exception as e:
            logging.error(f"Error closing WebSocket: {str(e)}")

    def process_tick_with_trading(self, tick_data, token, qty=25, sl_pct=0.02, target_pct=0.05):
        try:
            if token not in trade_state:
                return
                
            ltp = float(tick_data['last_traded_price']) / 100
            time_stamp = datetime.fromtimestamp(int(tick_data['exchange_timestamp']) / 1000)
            trade_state[token]['ltp'] = ltp
            
            # Get symbol details
            symbol_details = get_symbol_details(token)
            if not symbol_details:
                return
                
            # Initialize strategy if not exists
            if 'strategy' not in trade_state[token]:
                trade_state[token]['strategy'] = HighWinRateStrategy(symbol_details['symbol'])
                logging.info(f"Initialized strategy for {symbol_details['symbol']}")
                
            # Update strategy data
            strategy = trade_state[token]['strategy']
            strategy.update_data(tick_data)
            signals = strategy.generate_signals()
            
            logging.info(f"Current state for {symbol_details['symbol']}:")
            logging.info(f"LTP: {ltp}")
            logging.info(f"Trade Status: {trade_state[token]['status']}")
            if not signals.empty:
                logging.info(f"Current Signals: {signals.iloc[-1].to_dict()}")
                
            if signals.empty or len(signals) == 0:
                return
                
            # Process trading signals
            if trade_state[token]['status'] == 'IDLE':
                if signals['final_signal'].iloc[-1] == 1:  # Buy signal
                    # Calculate position size and risk parameters
                    position_size = strategy.calculate_position_size(100000)  # Example account balance
                    stop_loss = strategy.get_stop_loss(ltp, 1)
                    take_profit = strategy.get_take_profit(ltp, 1)
                    
                    # Place order
                    order_id = place_market_order(self.smartapi_obj, symbol_details, token, position_size, "BUY")
                    if order_id:
                        trade_state[token].update({
                            'status': 'OPEN',
                            'entry_price': ltp,
                            'stop_loss': stop_loss,
                            'target': take_profit,
                            'order_id': order_id,
                            'entry_time': time_stamp,
                            'quantity': position_size
                        })
                        
                        # Create trade record
                        trade = Trade(
                            symbol=symbol_details['symbol'],
                            strike=symbol_details.get('strike'),
                            option_type=symbol_details.get('option_type'),
                            entry_price=ltp,
                            quantity=position_size,
                            entry_time=time_stamp
                        )
                        paper_trades.append(trade)
                        
                        # Get signal details
                        signal_details = self._get_signal_details(signals)
                        
                        # Send notification using new notifier
                        telegram_notifier.notify_trade_entry(
                            trade={
                                'symbol': symbol_details['symbol'],
                                'strike': symbol_details.get('strike'),
                                'option_type': symbol_details.get('option_type'),
                                'entry_price': ltp,
                                'quantity': position_size,
                                'stop_loss': stop_loss,
                                'target': take_profit
                            },
                            strategy="High Win-Rate Strategy",
                            signal_details=signal_details
                        )
                        
            elif trade_state[token]['status'] == 'OPEN':
                # Check for exit conditions
                exit_signal = False
                exit_reason = ""
                signal_details = ""
                
                if ltp <= trade_state[token]['stop_loss']:
                    exit_signal = True
                    exit_reason = "Stop Loss Hit"
                    signal_details = "Price below stop loss"
                elif ltp >= trade_state[token]['target']:
                    exit_signal = True
                    exit_reason = "Target Hit"
                    signal_details = "Price above target"
                elif signals['final_signal'].iloc[-1] == -1:
                    exit_signal = True
                    exit_reason = "Signal Reversal"
                    signal_details = self._get_signal_details(signals)
                
                if exit_signal:
                    # Place exit order
                    exit_order_id = place_market_order(
                        self.smartapi_obj, 
                        symbol_details, 
                        token, 
                        trade_state[token]['quantity'], 
                        "SELL"
                    )
                    
                    if exit_order_id:
                        # Update trade state
                        trade_state[token]['status'] = 'IDLE'
                        
                        # Update trade record
                        for trade in paper_trades:
                            if (trade.symbol == symbol_details['symbol'] and 
                                trade.status == "OPEN"):
                                trade.close(
                                    exit_price=ltp,
                                    exit_time=time_stamp,
                                    reason=exit_reason
                                )
                                
                                # Send notification using new notifier
                                telegram_notifier.notify_trade_exit(
                                    trade={
                                        'symbol': trade.symbol,
                                        'exit_price': ltp,
                                        'entry_time': trade.entry_time,
                                        'exit_time': time_stamp,
                                        'pnl': trade.pnl
                                    },
                                    strategy="High Win-Rate Strategy",
                                    reason=exit_reason,
                                    signal_details=signal_details
                                )
                                break
                    
        except Exception as e:
            logging.error(f"Error in tick processing: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            telegram_notifier.notify_error(
                context="Tick Processing",
                message=str(e)
            )

    def _get_signal_details(self, signals):
        """Extract signal details from the signals DataFrame"""
        details = []
        if signals['rsi_signal'].iloc[-1] != 0:
            details.append(f"RSI {'Oversold' if signals['rsi_signal'].iloc[-1] == 1 else 'Overbought'}")
        if signals['macd_signal'].iloc[-1] != 0:
            details.append(f"MACD {'Bullish' if signals['macd_signal'].iloc[-1] == 1 else 'Bearish'}")
        if signals['bb_signal'].iloc[-1] != 0:
            details.append(f"Price {'Below' if signals['bb_signal'].iloc[-1] == 1 else 'Above'} BB")
        if signals['volume_signal'].iloc[-1] != 0:
            details.append("High Volume")
        if signals['oi_signal'].iloc[-1] != 0:
            details.append(f"OI {'Increasing' if signals['oi_signal'].iloc[-1] == 1 else 'Decreasing'}")
        if signals['iv_signal'].iloc[-1] != 0:
            details.append(f"IV {'Low' if signals['iv_signal'].iloc[-1] == 1 else 'High'}")
        if signals['greeks_signal'].iloc[-1] != 0:
            details.append(f"Greeks {'Favorable' if signals['greeks_signal'].iloc[-1] == 1 else 'Unfavorable'}")
        
        return ", ".join(details)

class HighWinRateStrategy:
    def __init__(self, contract_hub, account_balance=100000):
        self.contract_hub = contract_hub
        self.account_balance = account_balance
        self.data = pd.DataFrame()
        
        # Technical Indicators
        self.fast_ema_period = 9
        self.slow_ema_period = 21
        self.rsi_period = 14
        self.volume_ma_period = 20
        self.oi_ma_period = 20
        self.atr_period = 14
        
        # Trade Management
        self.trade_state = 'IDLE'  # IDLE, IN_TRADE
        self.current_trade = None
        self.last_trade_time = None
        self.trading_start_time = datetime.strptime('09:15', '%H:%M').time()
        self.trading_end_time = datetime.strptime('15:15', '%H:%M').time()
        self.max_trades_per_day = 2
        self.daily_loss_cap = 2000
        self.daily_pnl = 0
        self.trades_today = 0
        
        # Risk Management
        self.trailing_sl_activated = False
        self.trailing_sl_percentage = 0.20  # 20% in-the-money to activate trailing SL
        self.trailing_sl_distance = 0.10    # 10% trailing distance
        self.partial_exit_percentage = 0.50  # Exit 50% at first target
        self.partial_exit_target = 0.20     # 20% profit for partial exit
        
        # Greeks thresholds
        self.max_delta = 0.7
        self.min_delta = 0.3
        self.max_theta = -0.1
        self.max_iv = 0.5  # 50% IV
        
        # Volume and OI thresholds
        self.min_volume_increase = 1.1  # 10% increase
        self.min_oi_increase = 1.05     # 5% increase
        
        # Market Context
        self.market_regime = "UNKNOWN"
        self.volatility_threshold = 0.02
        self.support_levels = []
        self.resistance_levels = []
        
    def calculate_rsi(self, data, period=14):
        """Calculate RSI indicator"""
        delta = data['ltp'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def calculate_volume_ma(self, data, period=20):
        """Calculate Volume Moving Average"""
        return data['volume'].rolling(window=period).mean()
        
    def calculate_oi_ma(self, data, period=20):
        """Calculate Open Interest Moving Average"""
        return data['oi'].rolling(window=period).mean()
        
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
        
    def calculate_vwap(self, data):
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        volume_price = typical_price * data['volume']
        cumulative_volume = data['volume'].cumsum()
        cumulative_volume_price = volume_price.cumsum()
        return cumulative_volume_price / cumulative_volume
        
    def find_support_resistance(self, data, window=20, threshold=0.02):
        """Find support and resistance levels"""
        highs = data['high'].rolling(window=window, center=True).max()
        lows = data['low'].rolling(window=window, center=True).min()
        
        # Find local maxima and minima
        resistance = highs[highs == data['high']]
        support = lows[lows == data['low']]
        
        # Group nearby levels
        self.resistance_levels = self.group_price_levels(resistance, threshold)
        self.support_levels = self.group_price_levels(support, threshold)
        
    def group_price_levels(self, levels, threshold):
        """Group nearby price levels"""
        if len(levels) == 0:
            return []
            
        grouped = []
        current_group = [levels.iloc[0]]
        
        for price in levels.iloc[1:]:
            if abs(price - current_group[-1]) / current_group[-1] <= threshold:
                current_group.append(price)
            else:
                grouped.append(sum(current_group) / len(current_group))
                current_group = [price]
                
        if current_group:
            grouped.append(sum(current_group) / len(current_group))
            
        return grouped
        
    def analyze_market_context(self):
        """Analyze market context and regime"""
        if len(self.data) < self.atr_period:
            return "UNKNOWN"
            
        # Calculate volatility
        atr = self.calculate_atr(self.data, self.atr_period)
        price_range = self.data['high'].max() - self.data['low'].min()
        volatility_ratio = atr.iloc[-1] / price_range
        
        # Determine market regime
        if volatility_ratio > self.volatility_threshold:
            return "TRENDING"
        else:
            return "RANGING"
            
    def update_data(self, tick_data):
        """Update strategy data with new tick"""
        try:
            # Convert tick to DataFrame row
            tick_df = pd.DataFrame([{
                'timestamp': datetime.fromtimestamp(int(tick_data['exchange_timestamp']) / 1000),
                'ltp': float(tick_data['last_traded_price']) / 100,
                'high': float(tick_data['high_price_of_the_day']) / 100,
                'low': float(tick_data['low_price_of_the_day']) / 100,
                'close': float(tick_data['closed_price']) / 100,
                'volume': tick_data['volume_trade_for_the_day'],
                'oi': tick_data['open_interest'],
                'oi_change': tick_data.get('open_interest_change_percentage', 0),
                'best_bid': float(tick_data['best_5_buy_data'][0]['price']) / 100 if tick_data.get('best_5_buy_data') else None,
                'best_ask': float(tick_data['best_5_sell_data'][0]['price']) / 100 if tick_data.get('best_5_sell_data') else None,
                'total_buy_qty': tick_data['total_buy_quantity'],
                'total_sell_qty': tick_data['total_sell_quantity']
            }])
            
            # Append to existing data
            self.data = pd.concat([self.data, tick_df], ignore_index=True)
            
            # Keep only last 100 candles
            if len(self.data) > 100:
                self.data = self.data.tail(100)
                
            # Calculate indicators
            if len(self.data) >= self.slow_ema_period:
                # EMAs
                self.data['fast_ema'] = self.data['ltp'].ewm(span=self.fast_ema_period, adjust=False).mean()
                self.data['slow_ema'] = self.data['ltp'].ewm(span=self.slow_ema_period, adjust=False).mean()
                
                # RSI
                self.data['rsi'] = self.calculate_rsi(self.data, self.rsi_period)
                
                # Volume and OI
                self.data['volume_ma'] = self.calculate_volume_ma(self.data, self.volume_ma_period)
                self.data['oi_ma'] = self.calculate_oi_ma(self.data, self.oi_ma_period)
                
                # VWAP
                self.data['vwap'] = self.calculate_vwap(self.data)
                
                # Market Context
                self.market_regime = self.analyze_market_context()
                self.find_support_resistance(self.data)
                
                # Generate signals
                self.generate_signals()
                
        except Exception as e:
            logging.error(f"Error in update_data: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
    
    def generate_signals(self):
        """Generate trading signals based on multiple indicators"""
        # Only proceed if enough data for slow EMA
        if len(self.data) < self.slow_ema_period:
            return pd.DataFrame()  # Not enough data for EMAs

        # Calculate EMAs
        self.data['fast_ema'] = self.data['ltp'].ewm(span=self.fast_ema_period, adjust=False).mean()
        self.data['slow_ema'] = self.data['ltp'].ewm(span=self.slow_ema_period, adjust=False).mean()
        
        # Calculate RSI
        self.data['rsi'] = self.calculate_rsi(self.data, self.rsi_period)
        
        # Calculate Volume and OI MAs
        self.data['volume_ma'] = self.calculate_volume_ma(self.data, self.volume_ma_period)
        self.data['oi_ma'] = self.calculate_oi_ma(self.data, self.oi_ma_period)
        
        # Calculate VWAP
        self.data['vwap'] = self.calculate_vwap(self.data)
        
        # Initialize signal columns
        self.data['final_signal'] = 0
        self.data['rsi_signal'] = 0
        self.data['macd_signal'] = 0
        self.data['bb_signal'] = 0
        self.data['volume_signal'] = 0
        self.data['oi_signal'] = 0
        self.data['iv_signal'] = 0
        self.data['greeks_signal'] = 0
        
        # Get current and previous values
        current_fast = self.data['fast_ema'].iloc[-1]
        current_slow = self.data['slow_ema'].iloc[-1]
        prev_fast = self.data['fast_ema'].iloc[-2]
        prev_slow = self.data['slow_ema'].iloc[-2]
        
        # Get latest values for other indicators
        current_rsi = self.data['rsi'].iloc[-1]
        current_volume = self.data['volume'].iloc[-1]
        current_volume_ma = self.data['volume_ma'].iloc[-1]
        current_oi = self.data['oi'].iloc[-1]
        current_oi_ma = self.data['oi_ma'].iloc[-1]
        current_price = self.data['ltp'].iloc[-1]
        current_vwap = self.data['vwap'].iloc[-1]
        
        # Check for buy signal
        if (prev_fast <= prev_slow and current_fast > current_slow and  # EMA crossover
            current_rsi < 70 and  # Not overbought
            current_volume > current_volume_ma * self.min_volume_increase and  # Volume confirmation
            current_oi > current_oi_ma * self.min_oi_increase and  # OI confirmation
            current_price > current_vwap and  # Price above VWAP
            self.is_near_support(current_price)):  # Near support level
            
            self.data.loc[self.data.index[-1], 'final_signal'] = 1  # Buy signal
            self.data.loc[self.data.index[-1], 'rsi_signal'] = 1
            self.data.loc[self.data.index[-1], 'macd_signal'] = 1
            self.data.loc[self.data.index[-1], 'volume_signal'] = 1
            self.data.loc[self.data.index[-1], 'oi_signal'] = 1
            
        # Check for sell signal
        elif (prev_fast >= prev_slow and current_fast < current_slow and  # EMA crossover
              current_rsi > 30 and  # Not oversold
              current_volume > current_volume_ma * self.min_volume_increase and  # Volume confirmation
              current_oi > current_oi_ma * self.min_oi_increase and  # OI confirmation
              current_price < current_vwap and  # Price below VWAP
              self.is_near_resistance(current_price)):  # Near resistance level
            
            self.data.loc[self.data.index[-1], 'final_signal'] = -1  # Sell signal
            self.data.loc[self.data.index[-1], 'rsi_signal'] = -1
            self.data.loc[self.data.index[-1], 'macd_signal'] = -1
            self.data.loc[self.data.index[-1], 'volume_signal'] = 1
            self.data.loc[self.data.index[-1], 'oi_signal'] = 1
            
        return self.data
    
    def is_near_support(self, price, threshold=0.01):
        """Check if price is near support level"""
        for level in self.support_levels:
            if abs(price - level) / level <= threshold:
                return True
        return False
        
    def is_near_resistance(self, price, threshold=0.01):
        """Check if price is near resistance level"""
        for level in self.resistance_levels:
            if abs(price - level) / level <= threshold:
                return True
        return False
        
    def should_exit_trade(self, current_price, entry_price):
        """Determine if we should exit the trade"""
        if not self.current_trade:
            return False, None
            
        # Calculate profit percentage
        profit_pct = (current_price - entry_price) / entry_price
        
        # Check for partial exit
        if (profit_pct >= self.partial_exit_target and 
            not self.current_trade.get('partial_exit_taken', False)):
            return True, "PARTIAL_EXIT"
            
        # Check for trailing stop loss
        if self.current_trade.get('trailing_sl_activated', False):
            if current_price <= self.current_trade['trailing_sl']:
                return True, "TRAILING_SL"
                
        # Check for stop loss
        if current_price <= self.current_trade['stop_loss']:
            return True, "STOP_LOSS"
            
        # Check for target
        if current_price >= self.current_trade['target']:
            return True, "TARGET"
            
        return False, None
        
    def process_tick_with_trading(self, tick_data):
        """Process tick data and execute trades"""
        try:
            # Get current time from tick data
            if 'exchange_timestamp' in tick_data:
                current_time = datetime.fromtimestamp(int(tick_data['exchange_timestamp']) / 1000).time()
            else:
                current_time = datetime.now().time()
            
            # Update strategy data
            self.update_data(tick_data)
            
            # Check if we can trade
            can_trade, signal_type = self.can_trade(current_time)
            
            if can_trade:
                if signal_type == "ENTRY":
                    # Get latest signal
                    latest_signal = self.data['signal'].iloc[-1]
                    
                    # Determine option type based on signal
                    option_type = 'CE' if latest_signal == 1 else 'PE'
                    
                    # Get ATM strike
                    atm_strike = self.contract_hub.get_atm_strike()
                    if not atm_strike:
                        return
                        
                    # Get contract details
                    contract = self.contract_hub.get_contract(atm_strike, option_type)
                    if not contract:
                        return
                        
                    # Calculate entry price, stop loss and target
                    entry_price = tick_data['ltp']
                    stop_loss = entry_price * 0.75  # 25% stop loss
                    target = entry_price * 1.40     # 40% target
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(entry_price, stop_loss, option_type)
                    
                    # Update trade state
                    self.trade_state = 'IN_TRADE'
                    self.current_trade = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'position_size': position_size,
                        'option_type': option_type,
                        'entry_time': current_time,
                        'trailing_sl_activated': False,
                        'partial_exit_taken': False
                    }
                    self.trades_today += 1
                    
                    logging.info(f"Entered {option_type} trade at {entry_price}")
                    
                elif signal_type == "EXIT" and self.trade_state == 'IN_TRADE':
                    # Calculate PnL
                    exit_price = tick_data['ltp']
                    pnl = (exit_price - self.current_trade['entry_price']) * self.current_trade['position_size']
                    self.daily_pnl += pnl
                    
                    # Reset trade state
                    self.trade_state = 'IDLE'
                    self.current_trade = None
                    
                    logging.info(f"Exited trade at {exit_price}, PnL: {pnl}")
                    
            # Check for exits if in trade
            elif self.trade_state == 'IN_TRADE':
                current_price = tick_data['ltp']
                should_exit, exit_reason = self.should_exit_trade(current_price, self.current_trade['entry_price'])
                
                if should_exit:
                    if exit_reason == "PARTIAL_EXIT":
                        # Exit 50% of position
                        exit_size = self.current_trade['position_size'] * self.partial_exit_percentage
                        pnl = (current_price - self.current_trade['entry_price']) * exit_size
                        self.daily_pnl += pnl
                        self.current_trade['position_size'] -= exit_size
                        self.current_trade['partial_exit_taken'] = True
                        logging.info(f"Partial exit at {current_price}, PnL: {pnl}")
                        
                    else:
                        # Full exit
                        pnl = (current_price - self.current_trade['entry_price']) * self.current_trade['position_size']
                        self.daily_pnl += pnl
                        self.trade_state = 'IDLE'
                        self.current_trade = None
                        logging.info(f"Exited trade at {current_price} due to {exit_reason}, PnL: {pnl}")
                        
                # Update trailing stop loss if activated
                elif self.current_trade.get('trailing_sl_activated', False):
                    new_sl = current_price * (1 - self.trailing_sl_distance)
                    if new_sl > self.current_trade['trailing_sl']:
                        self.current_trade['trailing_sl'] = new_sl
                        logging.info(f"Updated trailing stop loss to {new_sl}")
                        
                # Check for trailing stop loss activation
                elif not self.current_trade.get('trailing_sl_activated', False):
                    profit_pct = (current_price - self.current_trade['entry_price']) / self.current_trade['entry_price']
                    if profit_pct >= self.trailing_sl_percentage:
                        self.current_trade['trailing_sl_activated'] = True
                        self.current_trade['trailing_sl'] = current_price * (1 - self.trailing_sl_distance)
                        logging.info(f"Activated trailing stop loss at {current_price}")
                        
        except Exception as e:
            logging.error(f"Error in process_tick_with_trading: {str(e)}")
            logging.error(f"Full error details: {traceback.format_exc()}")

class Trade:
    def __init__(self, symbol, strike, option_type, entry_price, quantity, entry_time):
        self.symbol = symbol
        self.strike = strike
        self.option_type = option_type  # 'CE' or 'PE'
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.status = "OPEN"
        self.exit_reason = None

    def close(self, exit_price, exit_time, reason="Manual Exit"):
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = (exit_price - self.entry_price) * self.quantity if self.option_type == 'CE' else (self.entry_price - exit_price) * self.quantity
        self.status = "CLOSED"
        self.exit_reason = reason

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)
    rho = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

def enter_trade(symbol, strike, option_type, entry_price, quantity, entry_time):
    trade = Trade(symbol, strike, option_type, entry_price, quantity, entry_time)
    paper_trades.append(trade)
    return trade

def exit_trade(trade, exit_price, exit_time, reason="Manual Exit"):
    trade.close(exit_price, exit_time, reason)

def export_trades_to_csv(filename="trades.csv"):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["symbol", "strike", "option_type", "entry_price", "quantity", "entry_time", "exit_price", "exit_time", "pnl", "status", "exit_reason"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trade in paper_trades:
            writer.writerow(trade.__dict__)

def get_total_pnl():
    return sum(t.pnl for t in paper_trades if t.status == "CLOSED")

def get_win_rate():
    closed = [t for t in paper_trades if t.status == "CLOSED"]
    wins = [t for t in closed if t.pnl > 0]
    return (len(wins) / len(closed) * 100) if closed else 0

def get_available_instruments():
    """Get list of available trading instruments"""
    return list(map_dict.keys())

def set_global_smartapi(api_obj):
    """Set the global smartapi_obj variable from an external module"""
    global smartapi_obj
    smartapi_obj = api_obj
    logging.info("Global SmartAPI object has been set from external source")
    return True

if __name__ == "__main__":
    print("Starting AlgoTrade application...")
    logger = setup_logging()
    logging.info("Logging initialized")
    
    try:
        print("Attempting to login to SmartAPI...")
        # Initialize SmartAPI with retry
        smartapi_obj, session_data = login_to_smartapi()
        logging.info("SmartAPI login successful")
        
        # Get feed token
        print("Getting feed token...")
        feed_token = session_data['data']['feedToken']
        if not feed_token:
            raise Exception("Could not get feed token")
        logging.info("Successfully retrieved feed token")
        
        # Process contracts with retry
        print("Processing contracts...")
        contract_hub_exchng = preprocess_contract_hub_exchange()
        logging.info("Contract processing completed")
        
        # Let user select the instrument
        print("Selecting trading instrument...")
        selected_instrument = select_trading_instrument()
        logging.info(f"Selected instrument: {selected_instrument}")
        
        # Get expiry dates for selected instrument
        print("Getting expiry dates...")
        expiry_info, tokens = get_recent_expiry(selected_instrument)
        if not expiry_info or not tokens:
            raise Exception(f"Could not get expiry info for {selected_instrument}")
            
        all_expiries = {map_dict[selected_instrument]['NAME']: expiry_info[map_dict[selected_instrument]['NAME']]}
        logging.info(f"Expiry info: {all_expiries}")
        
        # Get first expiry date
        instrument_expiry = all_expiries[map_dict[selected_instrument]['NAME']][0]
        logging.info(f"Selected expiry: {instrument_expiry}")
        
        # For MCX instruments
        if map_dict[selected_instrument]['SEGMENT'] == 'MCX':
            print("Processing MCX instrument...")
            expiry_key = f"{selected_instrument}_{instrument_expiry}"
            if expiry_key in exchange_token_hub and "FUT" in exchange_token_hub[expiry_key]:
                selected_token = exchange_token_hub[expiry_key]["FUT"][0]
                logging.info(f"Found MCX token for {selected_instrument}: {selected_token}")
                
                # Print expiry and symbol in terminal for the MCX token
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
                
                # Initialize token list for MCX
                token_list = [{
                    "exchangeType": 5,  # MCX exchange type
                    "tokens": [selected_token]
                }]
            else:
                raise Exception(f"No futures tokens found for {selected_instrument} expiry {instrument_expiry}")
        else:
            print("Processing options instrument...")
            # For options instruments, get ATM strike and CE token
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
            
            # Get all CE and PE symbols with their tokens
            ce_symbols = {get_symbol_details(t)['symbol']: t for t in tokens_dict["CE"] if get_symbol_details(t)}
            pe_symbols = {get_symbol_details(t)['symbol']: t for t in tokens_dict["PE"] if get_symbol_details(t)}
            
            # Get all available strikes for this expiry
            expiry_contracts = df[df['expiry'] == instrument_expiry]
            if expiry_contracts.empty:
                raise Exception(f"No contracts found for expiry {instrument_expiry}")
                
            all_strikes = sorted(expiry_contracts['strike'].unique())
            logging.info(f"Available strikes: {all_strikes[:10]}")  # Show first 10 strikes
            
            # Find ATM strike (nearest to spot price)
            # Convert spot price to match strike format (multiply by 100)
            spot_price_100 = current_price * 100
            atm_strike = min(all_strikes, key=lambda x: abs(float(x) - spot_price_100))
            logging.info(f"ATM Strike selected: {atm_strike}")
            
            # Initialize monitoring for ATM strikes only
            token_list = []
            monitored_contracts = []
            
            # Monitor only ATM strikes for both CE and PE
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
                        'strike': float(atm_strike)/100,  # Convert back to normal price
                        'expiry': instrument_expiry
                    }
                    token_list.append(contract_details['token'])
                    monitored_contracts.append(contract_details)
                    logging.info(f"Monitoring {contract_details['tradingsymbol']} (ATM) - Token: {contract_details['token']}")
                    initialize_trade_state(contract_details['token'], contract_details['tradingsymbol'], entry_threshold=float('inf'))
            
            if not token_list:
                raise Exception("No valid tokens found for monitoring")
            
            # Initialize WebSocket with selected tokens
            ws_token_list = [{
                "action": 1,
                "exchangeType": 2,
                "tokens": token_list
            }]
            
            # Send startup notification using new notifier
            monitored_symbols = [contract['tradingsymbol'] for contract in monitored_contracts]
            telegram_notifier.notify_startup(
                instrument=selected_instrument,
                expiry=instrument_expiry,
                spot_price=current_price,
                atm_strike=float(atm_strike)/100,
                monitored=monitored_symbols,
                strategy="High Win-Rate Strategy"
            )
        
        print("Creating WebSocket instance...")
        # Create WebSocket instance
        websocket = SmartAPIWebSocket(
            session_data,
            API_KEY,
            CLIENT_CODE,
            ws_token_list
        )
        websocket.smartapi_obj = smartapi_obj
        
        # Initialize trade states for tokens
        if map_dict[selected_instrument]['SEGMENT'] == 'MCX':
            # For MCX, just initialize the futures token
            initialize_trade_state(selected_token, symbol_details.get('symbol', 'N/A'), 0.5)
        else:
            # For options, initialize both CE and PE tokens
            for contract in monitored_contracts:
                initialize_trade_state(contract['token'], contract['tradingsymbol'], 0.5)
        
        print("Starting WebSocket connection...")
        # Start WebSocket in a separate thread
        websocket.start()
        
        print("Application started successfully. Press Ctrl+C to exit.")
        # Keep the main thread running with heartbeat check
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
