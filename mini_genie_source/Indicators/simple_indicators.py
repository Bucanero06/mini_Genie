import vectorbtpro as vbt


EMA = vbt.IF.from_expr("ema=@talib_ema(close, window)\n ema")
ATR_EWM = vbt.IF.from_expr("""
                        ATR:
                        tr0 = abs(high - low)
                        tr1 = abs(high - fshift(close))
                        tr2 = abs(low - fshift(close))
                        tr = nanmax(column_stack((tr0, tr1, tr2)), axis=1)
                        atr = @talib_ema(tr, 2 * window - 1)  # Wilder's EMA
                        tr, atr
                        """)