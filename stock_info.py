from alpha_vantage.timeseries import TimeSeries
import pandas as pd

def get_stock_info(symbol, api_key="0NWT0509VXGMKUSE"):
	"""returns csv data for the given stock ticker and time frame.

        Parameters
        ----------
        symbol : str
            The desired stock ticker
        """
	try:
  		ts = TimeSeries(key=api_key, output_format='pandas')
	except:
		raise ValueError("Invalid API key.")

	try:
  		data, metadata = ts.get_daily(symbol=symbol, outputsize="full")
	except:
  		raise ValueError("Invalid stock symbol.")

	data.reset_index(inplace=True)
	del data['5. volume']
	del data['date']
	return data