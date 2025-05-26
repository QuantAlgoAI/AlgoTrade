from flask import Flask, render_template_string
from models import db, Trade
from analytics import TradingAnalytics
import plotly.graph_objs as go
import pandas as pd
import os

app = Flask(__name__)
app.config.from_pyfile('config.py', silent=True)
db.init_app(app)

# Basic authentication placeholder (replace with real auth in production)
def is_authenticated():
    return True  # Replace with real user/session check

@app.route('/dashboard')
def dashboard():
    if not is_authenticated():
        return "Unauthorized", 401
    # Get recent trades
    trades = Trade.query.order_by(Trade.entry_time.desc()).limit(20).all()
    trades_df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()
    # Get performance metrics
    analytics = TradingAnalytics()
    metrics = analytics.calculate_performance_metrics()
    # Plotly PnL chart
    pnl_fig = None
    if not trades_df.empty:
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        trades_df = trades_df.sort_values('entry_time')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        pnl_fig = go.Figure()
        pnl_fig.add_trace(go.Scatter(x=trades_df['entry_time'], y=trades_df['cumulative_pnl'], mode='lines', name='Cumulative PnL'))
        pnl_fig.update_layout(title='Cumulative PnL', xaxis_title='Time', yaxis_title='PnL')
    # Render dashboard
    return render_template_string('''
    <html><head><title>Trading Dashboard</title></head><body>
    <h1>Trading Dashboard</h1>
    <h2>Performance Metrics</h2>
    <ul>
      {% for k, v in metrics.items() %}
        <li><b>{{ k }}:</b> {{ v }}</li>
      {% endfor %}
    </ul>
    <h2>Recent Trades</h2>
    {% if not trades_df.empty %}
      {{ trades_df.to_html(index=False) }}
    {% else %}
      <p>No trades yet.</p>
    {% endif %}
    <h2>Cumulative PnL</h2>
    {% if pnl_fig %}
      {{ pnl_fig.to_html(full_html=False) | safe }}
    {% endif %}
    </body></html>
    ''', trades_df=trades_df, metrics=metrics, pnl_fig=pnl_fig)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True) 