#Naming Convention
"""
>  First, let's talk about the naming convention. All functions perform an array operation on financial time series. To make life easier,
   the function name contains a shorthand notation of the array operation and the requested timeframe. Assume every function is executed
   once per bar close. If the timeframe portion of the name is omitted, then it needs CURRENT timeframe data.

   Example:    ema_4H()
      >  The first half of the name, 'ema', refers to exponential moving average operation.
      >  The second half of the name, '_4H', refers to the 4 Hour timeframe of the requested financial data.
      >  This function should run once per 4H bar close, which can be approximated to every 240 minutes.

   Next, the conventions for arguments. All functions require a source time series and often require the number of bars to perform the
   operation on. If the source is omitted, then the default time series is bar Close.

   Example: ema_4H(13,Low)
      >  This function performs an exponential moving average operation on the Low of the 13 most recent 4H bars.
   
   Example: sma(50)
      >  This function performs a simple moving average operation on the Close of the 50 most recent CURRENT bars.

   Finally, the convention for return types. Nearly all functions return a time series, though it may not be numerical. This means we're 
   able to chain functions to make compact expressions that perform complex operations.

   Example: ema_4H(13,Low).sma(5)
      >  This function returns a time series of transformed Low asset_prices from the 4H.
      >  First, the exponential moving average operation occurs, then we take the simple moving average of the result.
      >  Of course, we're talking about a discrete implementation here - easy to debug and follow along. 
         An SMA of an EMA can be represented using a difference equation, transfer function, or other continuous expression.
"""

#Reserved Words
"""
   Close
      >  Close price of the current bar when it has closed, or last traded price of a yet incomplete, realtime bar.
   
   Open, High, and Low
      >  Open, High, and Low asset_prices of the current bar

"""

#Functions in this Document:
"""
   ema_TF(N, source = close)
      >  Exponential Moving Average of length N on Close from timeframe _TF
      >  Alpha = 2/(N+1)

   rma_TF(N, source = close)
      >  Exponential Moving Average of length N on Close from timeframe _TF
      >  Alpha = 1/N

   sma_TF(N, source = close)
      >  Simple Moving Average of length N on Close from timeframe _TF

   rsi_TF(N, source = close)
      >  Relative Strength Index. Calculated by taking the RMA of price change

   crossover(A, B)
      >  True if series A crossed above series B on the Close of the current bar.

   crossunder(A, B)
      >  True if series A crossed below series B on the Close of the current bar.

   barssince(boolean series)
      >  Returns the number of bars since the last occurance of a given boolean series
"""

#Final Remarks
"""
Sometimes its simply easier to describe the contents of a variable or a function. In this case, I'll use 'strings' in place of pseudocode.
For example, I refer to 'Currently Active Trades' to describe an integer variable that stores the number of open trades.
"""

#Public Parameters	   (Default Value)   #these are the parameters that are currently exposed to the end user

LotSize =                0.25              #A 'standard lot' is equal to $100,000. default trade size is 0.25 standard lot, or $25,000
TakeProfit       =         500
StopLoss    =                500
TrendType    =             'T1'              #Can be 'T1' or 'T2', see Entry Logic below
CloseReversal    =         False
EnableBreakEven    =       True
BreakEvenTrigger1         =        300
BreakEvenDis1    =          200
BreakEvenTrigger2         =        400
BreakEvenDis2    =          300
EnableTrailing        =      False
EnableTrailing2    =       False
TrailingStopDis    =       80
EnableMaxTrades    =       False
MaxTrades        =         10
EnableSession    =          False
SessionStart    =          '07:00'
SessionEnd    =             '22:00'

#Private Parameters	   (Default Value)   #these are parameters that are hidden from the end user

rsi_period        =         13
rsi_timeframe    =          '4H'
sma_on_rsi_len1     =     2		            #N value in rsi_4H(13).sma(N)
sma_on_rsi_len2      =    7
sma_on_rsi_len3      =    34
T1_ema_len1      =       13
T1_ema_len2     =        50
T1_ema_timeframe      =    'CURRENT'           #refers to timeframe of loaded chart. Can be 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W, 1M
T2_ema_len1         =       13
T2_ema_len2        =        50
T2_ema_timeframe     =     '4H'

#Entry Conditions:
"""
   All conditions must be true in order for new trades to be opened
"""
    1. not(EnableMaxTrades) or ('Current Active Trades' < MaxTrades)		      #limit the total active trades
    2. not(EnableSession) or (SessionStart < 'Current Time' < SessionEnd) 		#restrict the time allowed to open trades

#Entry Logic:
"""
   Indicator logic for opening a new trade. If all conditions are met, then a market order is made.
"""
    Buy Entry:
        1. TrendType == 'T1' or (ema_4h(13) > ema_4h(50))
        2. crossover( ema(13) , ema(50) )
        3. barssince( crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) ) <
           barssince( crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) )
           #ask me about this one. It's tough to get in writing and I can easily show this visually

    Sell Entry:
        1. TrendType == 'T1' or (ema_4h(13) < ema_4h(50))
        2. crossunder( ema(13) , ema(50) )
        3. barssince( crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossover( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) ) >
           barssince( crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(34) ) or crossunder( rsi_4h(13).sma(2) , rsi_4h(13).sma(7) ) )

#Trade Management:
"""
   Rules for managing a trade once it has been opened
"""
    'All trades are Market Type and should be executed immediately`
    'TakeProfit and StopLoss values are set immediately'
    'Trade Size' = LotSize


    3. if( EnableBreakEven == true )			         #As the trade moves into profit, we move the original Stop Loss from -500 points to +200, then +300 points.
         if( 'Trade Points' > BreakEvenTrigger1 )
            'Trade StopLoss' = BreakEvenDis1 # only move stop_loss
         if( 'Trade Points' > BreakEvenTrigger2 )
            'Trade StopLoss' = BreakEvenDis2 # only move stop_loss
    4. if( EnableTrailing and EnableTrailing2 )		#Requires price data at point resolution, so ignore for now
           if( 'Trade Points' > BreakEvenDis2 )
               'Trail the StopLoss by TrailingStopDis'
    5. if( EnableTrailing and not(EnableTrailing2))	#Requires price data at point resolution, so ignore for now
        'Trail the StopLoss by TrailingStopDis'

#Exit Conditions:
"""
   Rules for exiting a trade, either in profit or loss.
"""
    1. 'Trade Points' == -StopLoss
    2. 'Trade Points' == TakeProfit
    3. ( CloseReversal == true ) and 'new entry signal occurs that is opposite of current active trade'	#i.e. a sell trade is active and a new buy signal occurs, exit the sell trade immediately
