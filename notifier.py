import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class TelegramNotifier:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or TELEGRAM_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID

    def send(self, message: str, parse_mode: str = 'HTML'):
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        try:
            response = requests.post(url, data=payload)
            if response.status_code != 200:
                logging.error(f"Telegram API error: {response.text}")
        except Exception as e:
            logging.error(f"Telegram error: {e}")

    def notify_startup(self, instrument: str, expiry: str, spot_price: float, atm_strike: float, monitored: list, strategy: str):
        now = datetime.now().strftime('%H:%M:%S')
        msg = (
            f"🤖 <b>Trading Bot Started</b>\n"
            f"<b>Instrument:</b> {instrument}\n"
            f"<b>Expiry:</b> {expiry}\n"
            f"<b>Spot Price:</b> ₹{spot_price:,.2f}\n"
            f"<b>ATM Strike:</b> {atm_strike:,.0f}\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Monitoring:</b>\n"
        )
        for m in monitored:
            msg += f"• {m} (ATM)\n"
        msg += f"⏰ <i>{now}</i>"
        self.send(msg)

    def notify_trade_entry(self, trade: Dict[str, Any], strategy: str, signal_details: str):
        now = datetime.now().strftime('%H:%M:%S')
        msg = (
            f"🟢 <b>Trade Entry</b>\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Signal:</b> {signal_details}\n"
            f"<b>Symbol:</b> {trade['symbol']}\n"
            f"<b>Type:</b> {'Call Option (CE)' if trade.get('option_type') == 'CE' else 'Put Option (PE)'}\n"
            f"<b>Strike:</b> {trade['strike']}\n"
            f"<b>Entry Price:</b> ₹{trade['entry_price']:,.2f}\n"
            f"<b>Quantity:</b> {trade['quantity']}\n"
            f"<b>Stop Loss:</b> ₹{trade['stop_loss']:,.2f}\n"
            f"<b>Target:</b> ₹{trade['target']:,.2f}\n"
            f"⏰ <i>{now}</i>"
        )
        self.send(msg)

    def notify_trade_exit(self, trade: Dict[str, Any], strategy: str, reason: str, signal_details: str):
        now = datetime.now().strftime('%H:%M:%S')
        duration = self._format_duration(trade.get('entry_time'), trade.get('exit_time'))
        pnl = trade.get('pnl', 0)
        pnl_str = f"+₹{pnl:,.2f}" if pnl >= 0 else f"-₹{abs(pnl):,.2f}"
        msg = (
            f"🔴 <b>Trade Exit</b>\n"
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Symbol:</b> {trade['symbol']}\n"
            f"<b>Exit Price:</b> ₹{trade['exit_price']:,.2f}\n"
            f"<b>P&L:</b> {pnl_str}\n"
            f"<b>Duration:</b> {duration}\n"
            f"<b>Reason:</b> {reason} (Signal: {signal_details})\n"
            f"⏰ <i>{now}</i>"
        )
        self.send(msg)

    def notify_error(self, context: str, message: str):
        now = datetime.now().strftime('%H:%M:%S')
        msg = (
            f"⚠️ <b>Error</b>\n"
            f"<b>Context:</b> {context}\n"
            f"<b>Message:</b> {message}\n"
            f"⏰ <i>{now}</i>"
        )
        self.send(msg)

    @staticmethod
    def _format_duration(start: Optional[datetime], end: Optional[datetime]) -> str:
        if not start or not end:
            return "-"
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        delta = end - start
        return str(delta).split('.')[0]  # HH:MM:SS 