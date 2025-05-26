class BaseStrategy:
    def __init__(self, contract_hub, account_balance=100000):
        self.contract_hub = contract_hub
        self.account_balance = account_balance
        self.data = None

    def update_data(self, tick_data):
        """Update strategy data with new tick."""
        raise NotImplementedError

    def generate_signals(self):
        """Generate trading signals based on updated data."""
        raise NotImplementedError

    def should_enter_trade(self, *args, **kwargs):
        """Determine if a trade entry condition is met."""
        raise NotImplementedError

    def should_exit_trade(self, *args, **kwargs):
        """Determine if a trade exit condition is met."""
        raise NotImplementedError 