from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from sys import exit

# f=pd.read_csv(data)
# keep_col = ['day','month','lat','long']
# new_f = f[keep_col]
# new_f.to_csv("newFile.csv", index=False)

# del data['date']
# data.reset_index(inplace=True)
# del data['5. volume']
# del data['date']
# print(list(data))

# data.to_csv('stockoutput.csv', sep=',', header=False, index=False)
# print(list(data)


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
  		print("Invalid API key, exiting.")
  		exit()

	try:
  		data, metadata = ts.get_daily(symbol=symbol, outputsize="full")
	except:
  		print("Invalid stock symbol, exiting")
  		exit()

	data.reset_index(inplace=True)
	del data['5. volume']
	del data['date']
	return data