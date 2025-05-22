from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_login import UserMixin

db = SQLAlchemy()

# Define association table for many-to-many relationship between MarketScan and Instrument
scan_instruments = db.Table('scan_instruments',
    db.Column('scan_id', db.Integer, db.ForeignKey('market_scan.id'), primary_key=True),
    db.Column('instrument_id', db.Integer, db.ForeignKey('instrument.id'), primary_key=True)
)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    api_key = db.Column(db.Text)
    client_code = db.Column(db.String(50))
    telegram_chat_id = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    trades = db.relationship('Trade', backref='user', lazy=True)
    alerts = db.relationship('PriceAlert', backref='user', lazy=True)
    scans = db.relationship('MarketScan', backref='user', lazy=True)
    strategies = db.relationship('Strategy', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    instrument = db.Column(db.String(50), nullable=False)
    timeframe = db.Column(db.String(20), default='1d')
    entry_condition = db.Column(db.Text, nullable=False)
    exit_condition = db.Column(db.Text, nullable=False)
    position_size = db.Column(db.Float, default=1.0)
    stop_loss = db.Column(db.Float)
    take_profit = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    trades = db.relationship('Trade', backref='strategy', lazy=True)
    
    def __repr__(self):
        return f'<Strategy {self.name}>'

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50), nullable=False)
    strike = db.Column(db.Float)
    option_type = db.Column(db.String(10))
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Integer, nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)
    pnl = db.Column(db.Float, default=0)
    status = db.Column(db.String(20), default='OPEN')
    exit_reason = db.Column(db.String(100))
    trade_type = db.Column(db.String(20), default='PAPER')  # PAPER or LIVE
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'))
    
    def __repr__(self):
        return f'<Trade {self.symbol} {self.strike} {self.option_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'strike': self.strike,
            'option_type': self.option_type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M:%S') if self.exit_time else None,
            'pnl': self.pnl,
            'status': self.status,
            'exit_reason': self.exit_reason,
            'trade_type': self.trade_type
        }

class PriceAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50), nullable=False)
    alert_type = db.Column(db.String(20), nullable=False)  # ABOVE, BELOW, PERCENT_CHANGE
    price_target = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float)
    is_triggered = db.Column(db.Boolean, default=False)
    notification_sent = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    triggered_at = db.Column(db.DateTime)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<PriceAlert {self.symbol} {self.alert_type} {self.price_target}>'

class Instrument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, unique=True)
    name = db.Column(db.String(100))
    instrument_type = db.Column(db.String(20), default='STOCK')  # STOCK, ETF, etc.
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Instrument {self.symbol}>'

class MarketScan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    criteria = db.Column(db.JSON, nullable=False)  # JSON object of scan criteria
    is_active = db.Column(db.Boolean, default=True)
    last_run = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Many-to-many relationship with instruments
    instruments = db.relationship('Instrument', secondary=scan_instruments,
                                 lazy='subquery', backref=db.backref('scans', lazy=True))
    
    results = db.relationship('ScanResult', backref='scan', lazy=True)
    
    def __repr__(self):
        return f'<MarketScan {self.name}>'

class ScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(50), nullable=False)
    data = db.Column(db.Text, nullable=False)  # JSON data of the scan result
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scan_id = db.Column(db.Integer, db.ForeignKey('market_scan.id'), nullable=False)
    
    def __repr__(self):
        return f'<ScanResult {self.symbol}>'
    
    @property
    def data_dict(self):
        return json.loads(self.data)
    
    @data_dict.setter
    def data_dict(self, value):
        self.data = json.dumps(value)

class BacktestResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    total_trades = db.Column(db.Integer, default=0)
    winning_trades = db.Column(db.Integer, default=0)
    losing_trades = db.Column(db.Integer, default=0)
    total_pnl = db.Column(db.Float, default=0)
    max_drawdown = db.Column(db.Float, default=0)
    sharpe_ratio = db.Column(db.Float)
    trade_data = db.Column(db.Text)  # JSON string containing individual trade details
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<BacktestResult {self.strategy_id} {self.total_pnl}>'
    
    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100
    
    @property
    def trade_list(self):
        if not self.trade_data:
            return []
        return json.loads(self.trade_data)
    
    @trade_list.setter
    def trade_list(self, value):
        self.trade_data = json.dumps(value)

# New model for AngelOne's ScripMaster data
class ScripMaster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(50), index=True)  # Trading token
    symbol = db.Column(db.String(100), index=True) # Trading symbol
    name = db.Column(db.String(100))              # Company/Instrument name
    expiry = db.Column(db.Date, nullable=True)    # Expiry date for derivatives
    strike = db.Column(db.Float, nullable=True)   # Strike price for options
    option_type = db.Column(db.String(10), nullable=True)  # CE/PE for options
    exchange = db.Column(db.String(20), index=True)  # NSE/BSE/MCX
    segment = db.Column(db.String(30))  # EQ/FUT/OPT
    lot_size = db.Column(db.Integer)    # Contract/Lot size
    tick_size = db.Column(db.Float)     # Minimum tick size
    isin = db.Column(db.String(20), nullable=True)  # ISIN code
    trading_symbol = db.Column(db.String(100), index=True)  # Full trading symbol
    
    # Date tracking
    data_date = db.Column(db.Date, index=True)  # The date this data corresponds to
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ScripMaster {self.symbol} - {self.token}>'
        
    @property
    def as_dict(self):
        return {
            'token': self.token,
            'symbol': self.symbol,
            'name': self.name,
            'expiry': self.expiry.strftime('%Y-%m-%d') if self.expiry else None,
            'strike': self.strike,
            'option_type': self.option_type,
            'exchange': self.exchange,
            'segment': self.segment,
            'lot_size': self.lot_size,
            'tick_size': self.tick_size,
            'isin': self.isin,
            'trading_symbol': self.trading_symbol,
            'data_date': self.data_date.strftime('%Y-%m-%d')
        } 