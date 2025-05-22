import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional, Union
import sqlite3
from config import Config
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class TradingAnalytics:
    """Analytics engine for trading performance monitoring and analysis."""
    
    def __init__(self, db_path: str = "trading_analytics.db"):
        """Initialize analytics engine.
        
        Args:
            db_path: Path to SQLite database for storing analytics data
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    pnl REAL,
                    status TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    stop_loss REAL,
                    target_price REAL,
                    exit_reason TEXT
                )
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    avg_profit REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    daily_pnl REAL
                )
            """)
            
            conn.commit()
    
    def record_trade(self, trade_data: Dict):
        """Record a trade in the database.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Log the trade data being recorded
                logger.debug(f"Recording trade data: {json.dumps(trade_data, default=str)}")
                
                # Check if trade already exists
                cursor.execute("""
                    SELECT id FROM trades 
                    WHERE symbol = ? AND entry_time = ? AND entry_price = ?
                """, (
                    trade_data['symbol'],
                    trade_data['entry_time'],
                    trade_data['entry_price']
                ))
                
                existing_trade = cursor.fetchone()
                if existing_trade:
                    logger.warning(f"Trade already exists with ID: {existing_trade[0]}")
                    return
                
                # Insert new trade
                cursor.execute("""
                    INSERT INTO trades (
                        symbol, entry_time, exit_time, entry_price, exit_price,
                        quantity, direction, pnl, status, strategy,
                        stop_loss, target_price, exit_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['symbol'],
                    trade_data['entry_time'],
                    trade_data.get('exit_time'),
                    trade_data['entry_price'],
                    trade_data.get('exit_price'),
                    trade_data['quantity'],
                    trade_data['direction'],
                    trade_data.get('pnl'),
                    trade_data['status'],
                    trade_data['strategy'],
                    trade_data.get('stop_loss'),
                    trade_data.get('target_price'),
                    trade_data.get('exit_reason')
                ))
                
                # Get the ID of the inserted trade
                trade_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Successfully recorded trade {trade_id}: {trade_data['symbol']} at {trade_data['entry_time']}")
                
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            logger.error(f"Trade data: {json.dumps(trade_data, default=str)}")
            raise
    
    def update_trade(self, trade_id: int, update_data: Dict):
        """Update an existing trade record.
        
        Args:
            trade_id: ID of the trade to update
            update_data: Dictionary containing updated trade information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            set_clause = ", ".join(f"{k} = ?" for k in update_data.keys())
            cursor.execute(f"""
                UPDATE trades
                SET {set_clause}
                WHERE id = ?
            """, (*update_data.values(), trade_id))
            conn.commit()
    
    def calculate_performance_metrics(self, start_date: Optional[datetime] = None) -> Dict:
        """Calculate performance metrics for all trades.
        
        Args:
            start_date: Optional start date to filter trades from
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all trades
                query = """
                    SELECT * FROM trades 
                    ORDER BY entry_time
                """
                df = pd.read_sql_query(query, conn)
                
                logger.debug(f"Total trades in database: {len(df)}")
                
                if len(df) == 0:
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'win_rate': 0.0,
                        'avg_profit': 0.0,
                        'avg_loss': 0.0,
                        'profit_factor': 0.0,
                        'max_drawdown': 0.0,
                        'sharpe_ratio': 0.0,
                        'daily_pnl': 0.0
                    }
                
                # Convert entry_time to datetime
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                
                # Filter by start date if provided
                if start_date:
                    df = df[df['entry_time'] >= start_date]
                
                logger.debug(f"Total trades after filtering: {len(df)}")
                
                # Calculate metrics
                winning_trades = df[df['pnl'] > 0]
                losing_trades = df[df['pnl'] < 0]
                
                logger.debug(f"Winning trades: {len(winning_trades)}")
                logger.debug(f"Losing trades: {len(losing_trades)}")
                
                total_trades = len(df)
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                
                avg_profit = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
                
                total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
                total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Calculate daily PnL
                df['date'] = df['entry_time'].dt.date
                daily_pnl = df.groupby('date')['pnl'].sum().mean()
                
                # Calculate max drawdown
                cumulative_pnl = df['pnl'].cumsum()
                rolling_max = cumulative_pnl.expanding().max()
                drawdowns = (cumulative_pnl - rolling_max) / rolling_max * 100
                max_drawdown = abs(drawdowns.min())
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0)
                returns = df['pnl'].pct_change().dropna()
                sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 0 else 0
                
                metrics = {
                    'total_trades': total_trades,
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'daily_pnl': daily_pnl
                }
                
                logger.info("Performance metrics calculated:")
                for key, value in metrics.items():
                    logger.info(f"{key}: {value}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    
    def generate_performance_report(self, start_date: Optional[datetime] = None) -> str:
        """Generate a detailed performance report.
        
        Args:
            start_date: Start date for the report (default: 30 days ago)
            
        Returns:
            HTML string containing the performance report
        """
        metrics = self.calculate_performance_metrics(start_date)
        
        # Create performance dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Win Rate Distribution',
                'PnL Distribution',
                'Cumulative PnL',
                'Drawdown'
            )
        )
        
        with sqlite3.connect(self.db_path) as conn:
            trades_df = pd.read_sql_query("""
                SELECT * FROM trades
                WHERE entry_time >= ?
            """, conn, params=(start_date,))
            
            # Convert timestamp columns to datetime
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        
        if not trades_df.empty:
            # Win Rate Distribution
            fig.add_trace(
                go.Pie(
                    labels=['Winning Trades', 'Losing Trades'],
                    values=[metrics['winning_trades'], metrics['losing_trades']],
                    hole=0.3
                ),
                row=1, col=1
            )
            
            # PnL Distribution
            fig.add_trace(
                go.Histogram(
                    x=trades_df['pnl'],
                    nbinsx=50,
                    name='PnL Distribution'
                ),
                row=1, col=2
            )
            
            # Cumulative PnL
            cumulative_pnl = trades_df['pnl'].cumsum()
            fig.add_trace(
                go.Scatter(
                    x=trades_df['exit_time'],
                    y=cumulative_pnl,
                    name='Cumulative PnL'
                ),
                row=2, col=1
            )
            
            # Drawdown
            rolling_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - rolling_max) / rolling_max
            fig.add_trace(
                go.Scatter(
                    x=trades_df['exit_time'],
                    y=drawdown,
                    name='Drawdown'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Trading Performance Dashboard"
        )
        
        # Generate HTML report
        report = f"""
        <html>
        <head>
            <title>Trading Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ 
                    background: #f5f5f5;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    display: inline-block;
                    width: 200px;
                }}
                .metric-value {{ 
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{ 
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <h1>Trading Performance Report</h1>
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-value">{metrics['total_trades']}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['win_rate']:.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">₹{metrics['daily_pnl']:.2f}</div>
                    <div class="metric-label">Avg Daily PnL</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['profit_factor']:.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['max_drawdown']:.1%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics['sharpe_ratio']:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
            </div>
            {fig.to_html(full_html=False)}
        </body>
        </html>
        """
        
        return report
    
    def save_report(self, report: str, filename: str = None):
        """Save the performance report to a file.
        
        Args:
            report: HTML report string
            filename: Output filename (default: performance_report_YYYYMMDD.html)
        """
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d')}.html"
        
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        with open(reports_dir / filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Performance report saved to {filename}")
    
    def get_recent_trades(self, limit: int = 10) -> pd.DataFrame:
        """Get the most recent trades.
        
        Args:
            limit: Number of trades to retrieve
            
        Returns:
            DataFrame containing recent trades
        """
        with sqlite3.connect(self.db_path) as conn:
            trades_df = pd.read_sql_query("""
                SELECT * FROM trades
                ORDER BY entry_time DESC
                LIMIT ?
            """, conn, params=(limit,))
            
            # Convert timestamp columns to datetime
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            return trades_df
    
    def get_trade_statistics(self, symbol: Optional[str] = None) -> Dict:
        """Get trade statistics for a specific symbol or all symbols.
        
        Args:
            symbol: Optional symbol to filter trades
            
        Returns:
            Dictionary containing trade statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    symbol,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_profit,
                    AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss,
                    SUM(pnl) as total_pnl
                FROM trades
            """
            
            if symbol:
                query += " WHERE symbol = ?"
                params = (symbol,)
            else:
                params = ()
            
            query += " GROUP BY symbol"
            
            return pd.read_sql_query(query, conn, params=params).to_dict('records')
    
    def cleanup(self):
        """Clear all data from the database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM performance_metrics")
            conn.commit()
            logger.info("Database tables cleared")

    def plot_trade_distribution(self, save_path: Optional[str] = None) -> None:
        """Plot trade distribution by symbol and direction.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades", conn)
                
                # Create figure with subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Trade Distribution by Symbol',
                        'Trade Distribution by Direction',
                        'PnL Distribution by Symbol',
                        'Win Rate by Symbol'
                    )
                )
                
                # Trade count by symbol
                symbol_counts = df['symbol'].value_counts()
                fig.add_trace(
                    go.Bar(x=symbol_counts.index, y=symbol_counts.values, name='Trades'),
                    row=1, col=1
                )
                
                # Trade count by direction
                direction_counts = df['direction'].value_counts()
                fig.add_trace(
                    go.Bar(x=direction_counts.index, y=direction_counts.values, name='Direction'),
                    row=1, col=2
                )
                
                # PnL distribution by symbol
                for symbol in df['symbol'].unique():
                    symbol_data = df[df['symbol'] == symbol]
                    fig.add_trace(
                        go.Box(y=symbol_data['pnl'], name=symbol),
                        row=2, col=1
                    )
                
                # Win rate by symbol
                win_rates = []
                for symbol in df['symbol'].unique():
                    symbol_data = df[df['symbol'] == symbol]
                    win_rate = len(symbol_data[symbol_data['pnl'] > 0]) / len(symbol_data)
                    win_rates.append(win_rate)
                
                fig.add_trace(
                    go.Bar(x=df['symbol'].unique(), y=win_rates, name='Win Rate'),
                    row=2, col=2
                )
                
                fig.update_layout(
                    height=800,
                    width=1200,
                    title_text="Trade Analysis Dashboard",
                    showlegend=False
                )
                
                if save_path:
                    fig.write_html(save_path)
                else:
                    fig.show()
                
        except Exception as e:
            logger.error(f"Error creating trade distribution plot: {str(e)}")
            raise

    def plot_performance_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot performance metrics over time.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades", conn)
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                
                # Calculate cumulative PnL
                df['cumulative_pnl'] = df['pnl'].cumsum()
                
                # Calculate drawdown
                df['rolling_max'] = df['cumulative_pnl'].expanding().max()
                df['drawdown'] = (df['cumulative_pnl'] - df['rolling_max']) / df['rolling_max'] * 100
                
                # Create figure with subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        'Cumulative PnL',
                        'Drawdown',
                        'Daily PnL'
                    )
                )
                
                # Cumulative PnL
                fig.add_trace(
                    go.Scatter(x=df['entry_time'], y=df['cumulative_pnl'], name='Cumulative PnL'),
                    row=1, col=1
                )
                
                # Drawdown
                fig.add_trace(
                    go.Scatter(x=df['entry_time'], y=df['drawdown'], name='Drawdown %'),
                    row=2, col=1
                )
                
                # Daily PnL
                daily_pnl = df.groupby(df['entry_time'].dt.date)['pnl'].sum()
                fig.add_trace(
                    go.Bar(x=daily_pnl.index, y=daily_pnl.values, name='Daily PnL'),
                    row=3, col=1
                )
                
                fig.update_layout(
                    height=1200,
                    width=1200,
                    title_text="Performance Metrics Over Time",
                    showlegend=True
                )
                
                if save_path:
                    fig.write_html(save_path)
                else:
                    fig.show()
                
        except Exception as e:
            logger.error(f"Error creating performance metrics plot: {str(e)}")
            raise

    def analyze_trade_patterns(self) -> Dict:
        """Analyze trading patterns to identify potential issues.
        
        Returns:
            Dictionary containing pattern analysis results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades", conn)
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                
                # Time-based analysis
                df['hour'] = df['entry_time'].dt.hour
                df['day_of_week'] = df['entry_time'].dt.day_name()
                
                # Analyze patterns
                patterns = {
                    'time_distribution': {
                        'hourly': df.groupby('hour')['pnl'].agg(['count', 'mean', 'sum']).to_dict(),
                        'daily': df.groupby('day_of_week')['pnl'].agg(['count', 'mean', 'sum']).to_dict()
                    },
                    'symbol_analysis': {
                        symbol: {
                            'total_trades': len(df[df['symbol'] == symbol]),
                            'winning_trades': len(df[(df['symbol'] == symbol) & (df['pnl'] > 0)]),
                            'avg_profit': df[df['symbol'] == symbol]['pnl'].mean(),
                            'max_profit': df[df['symbol'] == symbol]['pnl'].max(),
                            'max_loss': df[df['symbol'] == symbol]['pnl'].min(),
                            'profit_factor': (
                                df[(df['symbol'] == symbol) & (df['pnl'] > 0)]['pnl'].sum() /
                                abs(df[(df['symbol'] == symbol) & (df['pnl'] < 0)]['pnl'].sum())
                                if len(df[(df['symbol'] == symbol) & (df['pnl'] < 0)]) > 0 else float('inf')
                            )
                        }
                        for symbol in df['symbol'].unique()
                    },
                    'consecutive_trades': {
                        'max_consecutive_wins': self._calculate_max_consecutive(df, True),
                        'max_consecutive_losses': self._calculate_max_consecutive(df, False)
                    }
                }
                
                # Log the analysis results
                logger.info("Trade Pattern Analysis Results:")
                for category, data in patterns.items():
                    logger.info(f"\n{category.upper()}:")
                    logger.info(data)
                
                return patterns
                
        except Exception as e:
            logger.error(f"Error analyzing trade patterns: {str(e)}")
            raise

    def _calculate_max_consecutive(self, df: pd.DataFrame, winning: bool) -> int:
        """Calculate maximum consecutive winning or losing trades.
        
        Args:
            df: DataFrame containing trades
            winning: True for winning trades, False for losing trades
            
        Returns:
            Maximum number of consecutive trades
        """
        if winning:
            condition = df['pnl'] > 0
        else:
            condition = df['pnl'] < 0
            
        consecutive = 0
        max_consecutive = 0
        
        for is_win in condition:
            if is_win:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
                
        return max_consecutive 