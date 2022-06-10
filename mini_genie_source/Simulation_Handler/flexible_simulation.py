#!/usr/bin/env python3
import warnings
from time import perf_counter

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- ↓ Do not remove these libs ↓ -------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import vectorbtpro as vbt
from logger_tt import logger
from numba import njit
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.chunking import flex_array_gl_slicer
from vectorbtpro.portfolio.enums import Direction, NoOrder, OrderStatus, OrderSide

# --- ↑ Do not remove these libs ↑ -------------------------------------------------------------------------------------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# defines how the inputs will be chunked and to execute with 'dask'
chunked = dict(
    arg_take_spec=dict(
        flex_order_args=vbt.ArgsTaker(
            flex_array_gl_slicer,  # long_entries
            flex_array_gl_slicer,  # long_exits
            flex_array_gl_slicer,  # short_entries
            flex_array_gl_slicer,  # short_exits
            flex_array_gl_slicer,  # order_size
            flex_array_gl_slicer,  # size_type
            flex_array_gl_slicer,  # fees
            flex_array_gl_slicer,  # slippage
            flex_array_gl_slicer,  # tick_size
            flex_array_gl_slicer,  # type_percent
            flex_array_gl_slicer,  # breakeven_1_distance_points
            flex_array_gl_slicer,  # breakeven_2_distance_points
            flex_array_gl_slicer,  # long_progressive_condition
            flex_array_gl_slicer,  # short_progressive_condition
            flex_array_gl_slicer,  # progressive_bool
            flex_array_gl_slicer,  # allow_multiple_trade_from_entries
            flex_array_gl_slicer,  # exit_on_opposite_direction_entry
        ),
        post_order_args=vbt.ArgsTaker(
            flex_array_gl_slicer,  # tick_size
            flex_array_gl_slicer,  # take_profit_bool
            flex_array_gl_slicer,  # take_profit_points
            flex_array_gl_slicer,  # stop_loss_bool
            flex_array_gl_slicer,  # stop_loss_points
            flex_array_gl_slicer,  # type_percent
            flex_array_gl_slicer,  # breakeven_1_trigger_bool
            flex_array_gl_slicer,  # breakeven_1_trigger_points

            flex_array_gl_slicer,  # breakeven_2_trigger_bool
            flex_array_gl_slicer,  # breakeven_2_trigger_points
        )
    ),

    engine='dask',
    # # n_chunks='auto',
    # # chunk_len='auto',
    # init_kwargs={'num_cpus': 20},
    # show_progress=True
)

open_order_tracker_dtype = np.dtype([
    # Value changed during sim funciton to track direction during post_sim_function in the current idx
    ('current_order_placed_direction', 'i'),

    ('long_exit_price_now', 'f8'),
    ('short_exit_price_now', 'f8'),

    # If open_order_size in the given direction is nan then opening_order is true
    ('long_opening_order', np.bool_),
    ('short_opening_order', 'f8'),

    # Used to track the parent price for the current position which guides the take-profits and stop-losses
    # even when using progressive entries
    ('long_open_order_price', 'f8'),
    ('short_open_order_price', 'f8'),

    # Used to keep track of the aggregate of the trade sizes for a given position direction
    ('long_open_order_size', 'f8'),
    ('short_open_order_size', 'f8'),

    # Used to keep track of position where the take-profit condition would hit for a give position direction
    ('long_take_profit_price', 'f8'),
    ('short_take_profit_price', 'f8'),

    # Used to keep track of position where the stop-loss condition would hit for a give position direction
    ('long_stop_loss_price', 'f8'),
    ('short_stop_loss_price', 'f8'),

    # Used to keep track of position where the break even condition 1 would hit for a give position direction and
    # thus changing the stop-loss price
    ('long_break_even_trigger_1_price', 'f8'),
    ('short_break_even_trigger_1_price', 'f8'),

    # Used to keep track of position where the break even condition 2 would hit for a give position direction and
    # thus changing the stop-loss price
    ('long_break_even_trigger_2_price', 'f8'),
    ('short_break_even_trigger_2_price', 'f8'),
])
entry_and_exit_police_dtype = np.dtype([
    # Used to prevent entry during same candle, if multiple trades in the same direction are not allowed and
    # one is already opened, or if a stop loss or take profits are hit
    ('long_prevent_entry', np.bool_),
    ('short_prevent_entry', np.bool_),

    # Used to prevent exit during same candle
    ('long_prevent_exit', np.bool_),
    ('short_prevent_exit', np.bool_),

    # Used to force an exit when exit in opposite direction is hit or when a stop loss or take profits are hit
    ('long_force_exit', np.bool_),
    ('short_force_exit', np.bool_),
])


@njit
def pre_sim_func_nb(c):
    # Array containing information about the open orders, stop-losses, take-profits, and break-even conditions.
    # Has shape of number of assets
    open_order_tracker = np.empty(c.target_shape[1], dtype=open_order_tracker_dtype)

    # Array containing information that persists across multiple call_id in the same time-step.
    # Has shape of (number of time-steps, number of assets)
    entry_and_exit_police = np.empty(c.target_shape, dtype=entry_and_exit_police_dtype)

    # Work around because Numba did not like using the dtype for the structured array without knowing the
    # exact shape I could not use full to fill the arrays at once
    for i in range(c.target_shape[1]):
        open_order_tracker['current_order_placed_direction'][i] = -1
        open_order_tracker['long_exit_price_now'][i] = np.nan
        open_order_tracker['short_exit_price_now'][i] = np.nan

        open_order_tracker['long_opening_order'][i] = False
        open_order_tracker['long_open_order_price'][i] = np.nan
        open_order_tracker['long_open_order_size'][i] = np.nan
        open_order_tracker['long_take_profit_price'][i] = np.nan
        open_order_tracker['long_stop_loss_price'][i] = np.nan
        open_order_tracker['long_break_even_trigger_1_price'][i] = np.nan
        open_order_tracker['long_break_even_trigger_2_price'][i] = np.nan

        open_order_tracker['short_opening_order'][i] = np.nan
        open_order_tracker['short_open_order_price'][i] = np.nan
        open_order_tracker['short_open_order_size'][i] = np.nan
        open_order_tracker['short_take_profit_price'][i] = np.nan
        open_order_tracker['short_stop_loss_price'][i] = np.nan
        open_order_tracker['short_break_even_trigger_1_price'][i] = np.nan
        open_order_tracker['short_break_even_trigger_2_price'][i] = np.nan
        # Since entry_and_exit_police has same shape as c.target_shape it requires another iteration
        for j in range(c.target_shape[0]):
            entry_and_exit_police['long_prevent_entry'][j][i] = False
            entry_and_exit_police['long_prevent_exit'][j][i] = False
            entry_and_exit_police['long_force_exit'][j][i] = False
            #
            entry_and_exit_police['short_prevent_entry'][j][i] = False
            entry_and_exit_police['short_prevent_exit'][j][i] = False
            entry_and_exit_police['short_force_exit'][j][i] = False

    return (open_order_tracker, entry_and_exit_police,)


@vbt.jitted
def flex_order_func_nb(c, open_order_tracker, entry_and_exit_police,
                       # _____Passed From Settings_____
                       long_entries, long_exits, short_entries, short_exits, order_size, size_type, fees,
                       slippage, tick_size, type_percent,
                       breakeven_1_distance_points, breakeven_2_distance_points,
                       long_progressive_condition, short_progressive_condition,
                       progressive_bool, allow_multiple_trade_from_entries, exit_on_opposite_direction_entry):
    """Flexible Sim Function"""

    '''Current Price'''
    low_price_now = nb.flex_select_auto_nb(c.low, c.i, c.from_col, c.flex_2d)
    high_price_now = nb.flex_select_auto_nb(c.high, c.i, c.from_col, c.flex_2d)
    close_price_now = nb.flex_select_auto_nb(c.close, c.i, c.from_col, c.flex_2d)

    '''Sim Settings'''
    size_type_now = nb.flex_select_auto_nb(size_type, c.i, c.from_col, c.flex_2d)
    order_size_now = nb.flex_select_auto_nb(order_size, c.i, c.from_col, c.flex_2d)
    entry_size_now = round(order_size_now / close_price_now if size_type_now == 1 else order_size_now, 8)
    # entry_size_now = order_size_now / close_price_now if size_type_now == 1 else order_size_now
    #
    fees_now = nb.flex_select_auto_nb(fees, c.i, c.from_col, c.flex_2d)
    slippage_now = nb.flex_select_auto_nb(slippage, c.i, c.from_col, c.flex_2d)
    tick_size_now = nb.flex_select_auto_nb(tick_size, c.i, c.from_col, c.flex_2d)
    #
    # allow multiple trades in the same direction to aggregate, otherwise only the first counts
    allow_multiple_trade_from_entries_now = nb.flex_select_auto_nb(allow_multiple_trade_from_entries, c.i, c.from_col,
                                                                   c.flex_2d)
    # Exit position if entry in the opposite the direction is true, else hedge
    exit_on_opposite_direction_entry_now = nb.flex_select_auto_nb(exit_on_opposite_direction_entry, c.i, c.from_col,
                                                                  c.flex_2d)

    '''Long Order Settings'''
    long_entries_now = nb.flex_select_auto_nb(long_entries, c.i, c.from_col, c.flex_2d)
    long_exits_now = nb.flex_select_auto_nb(long_exits, c.i, c.from_col, c.flex_2d)
    #
    long_open_order_price = open_order_tracker['long_open_order_price'][c.from_col]
    long_open_order_size = open_order_tracker['long_open_order_size'][c.from_col]
    long_take_profit_price = open_order_tracker['long_take_profit_price'][c.from_col]
    long_stop_loss_price = open_order_tracker['long_stop_loss_price'][c.from_col]
    long_break_even_trigger_1_price = open_order_tracker['long_break_even_trigger_1_price'][c.from_col]
    long_break_even_trigger_2_price = open_order_tracker['long_break_even_trigger_2_price'][c.from_col]

    '''Short Order Settings'''
    short_entries_now = nb.flex_select_auto_nb(short_entries, c.i, c.from_col, c.flex_2d)
    short_exits_now = nb.flex_select_auto_nb(short_exits, c.i, c.from_col, c.flex_2d)
    #
    short_open_order_price = open_order_tracker['short_open_order_price'][c.from_col]
    short_open_order_size = open_order_tracker['short_open_order_size'][c.from_col]
    short_take_profit_price = open_order_tracker['short_take_profit_price'][c.from_col]
    short_stop_loss_price = open_order_tracker['short_stop_loss_price'][c.from_col]
    short_break_even_trigger_1_price = open_order_tracker['short_break_even_trigger_1_price'][c.from_col]
    short_break_even_trigger_2_price = open_order_tracker['short_break_even_trigger_2_price'][c.from_col]

    '''Break Even Distances Settings'''
    break_even_dist_1_now = nb.flex_select_auto_nb(breakeven_1_distance_points, c.i, c.from_col, c.flex_2d)
    break_even_dist_2_now = nb.flex_select_auto_nb(breakeven_2_distance_points, c.i, c.from_col, c.flex_2d)

    '''Progressive Entries Settings'''
    '''FOR NATE's STRATEGY'''
    long_progressive_condition_now = nb.flex_select_auto_nb(long_progressive_condition, c.i, c.from_col, c.flex_2d)
    short_progressive_condition_now = nb.flex_select_auto_nb(short_progressive_condition, c.i, c.from_col, c.flex_2d)
    progressive_bool_now = nb.flex_select_auto_nb(progressive_bool, c.i, c.from_col, c.flex_2d)

    '''Check If Break Even Condition has Triggered (if so, adjust stop loss)'''
    if high_price_now >= long_break_even_trigger_1_price and not np.isnan(long_open_order_size):
        # long_break_even_trigger_1_price and a long position is opened then redefine stop_loss
        long_stop_loss_price = open_order_tracker['long_stop_loss_price'][c.from_col] = long_open_order_price * (
                1 + break_even_dist_1_now) if type_percent \
            else long_open_order_price + (break_even_dist_1_now * tick_size_now)
    if high_price_now >= long_break_even_trigger_2_price and not np.isnan(long_open_order_size):
        # long_break_even_trigger_2_price and a long position is opened then redefine stop_loss
        long_stop_loss_price = open_order_tracker['long_stop_loss_price'][c.from_col] = long_open_order_price * (
                1 + break_even_dist_2_now) if type_percent \
            else long_open_order_price + (break_even_dist_2_now * tick_size_now)

    if low_price_now <= short_break_even_trigger_1_price and not np.isnan(short_open_order_size):
        # short_break_even_trigger_1_price and a short position is opened then redefine stop_loss
        short_stop_loss_price = open_order_tracker['short_stop_loss_price'][c.from_col] = short_open_order_price * (
                1 - break_even_dist_2_now) if type_percent \
            else short_open_order_price - (break_even_dist_1_now * tick_size_now)
    if low_price_now <= short_break_even_trigger_2_price and not np.isnan(short_open_order_size):
        # short_break_even_trigger_2_price and a short position is opened then redefine stop_loss
        short_stop_loss_price = open_order_tracker['short_stop_loss_price'][c.from_col] = short_open_order_price * (
                1 - break_even_dist_2_now) if type_percent \
            else short_open_order_price - (break_even_dist_2_now * tick_size_now)

    '''If Progressive entry determine whether any type C entry conditions have been met'''
    '''FOR NATE's STRATEGY'''
    # If met then change entry now to true and exit now to false
    if (long_progressive_condition_now and progressive_bool_now) and not np.isnan(
            open_order_tracker['long_open_order_size'][c.from_col]):
        long_entries_now = True
        long_exits_now = False
    if (short_progressive_condition_now and progressive_bool_now) and not np.isnan(
            open_order_tracker['short_open_order_size'][c.from_col]):
        short_entries_now = True
        short_exits_now = False

    # print("Outside", low_price_now, high_price_now, close_price_now, long_stop_loss_price, long_take_profit_price)
    '''Determine whether any type B exit conditions have been met'''
    # if exit trigger and nothing preventing it and there is a position opened in the other direction then prevent
    # entry and force exit in other direction, change entry now to false and exit now to true
    if (high_price_now >= long_take_profit_price or low_price_now <= long_stop_loss_price) \
            and not entry_and_exit_police['long_prevent_exit'][c.i][c.from_col]:
        long_entries_now = False
        long_exits_now = True
        entry_and_exit_police['long_prevent_entry'][c.i][c.from_col] = True
        entry_and_exit_police['long_force_exit'][c.i][c.from_col] = True
        # print("Long", low_price_now, high_price_now, close_price_now, long_stop_loss_price, long_take_profit_price)

    if (low_price_now <= short_take_profit_price or high_price_now >= short_stop_loss_price) \
            and not entry_and_exit_police['short_prevent_exit'][c.i][c.from_col]:
        short_entries_now = False
        short_exits_now = True
        entry_and_exit_police['short_prevent_entry'][c.i][c.from_col] = True
        entry_and_exit_police['short_force_exit'][c.i][c.from_col] = True

    '''Exit position if entry in the opposite the direction is true'''
    if exit_on_opposite_direction_entry_now:
        # if entry and nothing preventing it and there is a position opened in the other direction then prevent
        # entry and force exit in other direction, change entry now to false and exit now to true
        if long_entries_now and not entry_and_exit_police['long_prevent_entry'][c.i][c.from_col] and \
                not np.isnan(open_order_tracker['short_open_order_size'][c.from_col]):
            short_entries_now = False
            short_exits_now = True
            entry_and_exit_police['short_prevent_entry'][c.i][c.from_col] = True
            entry_and_exit_police['short_force_exit'][c.i][c.from_col] = True
        #
        if short_entries_now and not entry_and_exit_police['short_prevent_entry'][c.i][c.from_col] and \
                not np.isnan(open_order_tracker['long_open_order_size'][c.from_col]):
            long_entries_now = False
            long_exits_now = True
            entry_and_exit_police['long_prevent_entry'][c.i][c.from_col] = True
            entry_and_exit_police['long_force_exit'][c.i][c.from_col] = True

    '''Prevent From Opening Multiple Trades if allow_multiple_trade_from_entries_now is False'''
    if not allow_multiple_trade_from_entries_now:
        # entry now and there is already a position open in the same direction then set entry now to false and prevent
        # entry
        if long_entries_now and not np.isnan(open_order_tracker['long_open_order_size'][c.from_col]):
            long_entries_now = False
            entry_and_exit_police['long_prevent_entry'][c.i][c.from_col] = True
        if short_entries_now and not np.isnan(open_order_tracker['short_open_order_size'][c.from_col]):
            short_entries_now = False
            entry_and_exit_police['short_prevent_entry'][c.i][c.from_col] = True

    '''Set entry_and_exit_police values for readability'''
    # Used to prevent entry during same candle, if multiple trades in the same direction are not allowed and
    # one is already opened, or if a stop loss or take profits are hit
    long_prevent_entry = entry_and_exit_police['long_prevent_entry'][c.i][c.from_col]
    short_prevent_entry = entry_and_exit_police['short_prevent_entry'][c.i][c.from_col]

    # Used to prevent exit during same candle
    long_prevent_exit = entry_and_exit_police['long_prevent_exit'][c.i][c.from_col]
    short_prevent_exit = entry_and_exit_police['short_prevent_exit'][c.i][c.from_col]

    # Used to force an exit when exit in opposite direction is hit or when a stop loss or take profits are hit
    long_force_exit = entry_and_exit_police['long_force_exit'][c.i][c.from_col]
    short_force_exit = entry_and_exit_police['short_force_exit'][c.i][c.from_col]

    '''Long Orders Logic'''
    if long_entries_now and not (long_force_exit or long_prevent_entry):

        # Prevent entry and exit during the following call_id
        entry_and_exit_police['long_prevent_entry'][c.i][c.from_col] = True
        entry_and_exit_police['long_prevent_exit'][c.i][c.from_col] = True

        # Track direction during post_sim_function this call_id
        open_order_tracker['current_order_placed_direction'][c.from_col] = 0

        # If long_open_order_size is nan then long_opening_order is true
        open_order_tracker['long_opening_order'][c.from_col] = True if np.isnan(
            open_order_tracker['long_open_order_size'][c.from_col]) else False

        # Aggregate the long trade sizes add size to ongoing long position else start a new one
        open_order_tracker['long_open_order_size'][c.from_col] = entry_size_now + \
                                                                 open_order_tracker['long_open_order_size'][
                                                                     c.from_col] if not np.isnan(
            open_order_tracker['long_open_order_size'][c.from_col]) else entry_size_now

        # Place Order
        return c.from_col, nb.order_nb(
            size=entry_size_now,
            price=close_price_now,
            direction=Direction.Both,
            fees=fees_now,
            slippage=(tick_size_now * slippage_now) / close_price_now
        )
    #
    elif (long_exits_now or long_force_exit) and not long_prevent_exit:

        # Prevent exit during the following call_id (entry is already false --> elif)
        entry_and_exit_police['long_prevent_exit'][c.i][c.from_col] = True

        # Track direction during post_sim_function this call_id
        open_order_tracker['current_order_placed_direction'][c.from_col] = 0

        # Exit size for long position is equal to the aggregate sum of long sizes
        long_exit_size = open_order_tracker['long_open_order_size'][c.from_col]

        # If long_open_order_size is nan then there is nothing to exit thus miscellaneous long exit
        if np.isnan(long_exit_size): return -1, NoOrder

        # Exiting long position thus set long_open_order_size to nan
        open_order_tracker['long_open_order_size'][c.from_col] = np.nan

        return c.from_col, nb.order_nb(
            size=-long_exit_size,
            price=close_price_now,
            direction=Direction.Both,
            fees=fees_now,
            slippage=(tick_size_now * slippage_now) / close_price_now
        )

    '''Short Orders Logic'''
    if short_entries_now and not (short_force_exit or short_prevent_entry):

        # Prevent entry and exit during the following call_id
        entry_and_exit_police['short_prevent_entry'][c.i][c.from_col] = True
        entry_and_exit_police['short_prevent_exit'][c.i][c.from_col] = True

        # Track direction during post_sim_function this call_id
        open_order_tracker['current_order_placed_direction'][c.from_col] = 1

        # If short_open_order_size is nan then short_opening_order is true
        open_order_tracker['short_opening_order'][c.from_col] = True if np.isnan(
            open_order_tracker['short_open_order_size'][c.from_col]) else False

        # Aggregate the short trade sizes add size to ongoing short position else start a new one
        open_order_tracker['short_open_order_size'][c.from_col] = open_order_tracker['short_open_order_size'][
                                                                      c.from_col] - entry_size_now if not np.isnan(
            open_order_tracker['short_open_order_size'][c.from_col]) else -entry_size_now

        # Place Order
        return c.from_col, nb.order_nb(
            size=-entry_size_now,
            price=close_price_now,
            direction=Direction.Both,
            fees=fees_now,
            slippage=(tick_size_now * slippage_now) / close_price_now
        )
    #
    elif (short_exits_now or short_force_exit) and not short_prevent_exit:

        # Prevent exit during the following call_id (entry is already false --> elif)
        entry_and_exit_police['short_prevent_exit'][c.i][c.from_col] = True

        # Track direction during post_sim_function this call_id
        open_order_tracker['current_order_placed_direction'][c.from_col] = 1

        # Exit size for short position is equal to the aggregate sum of short sizes
        short_exit_size = open_order_tracker['short_open_order_size'][c.from_col]

        # If short_open_order_size is nan then there is nothing to exit thus miscellaneous short exit
        if np.isnan(short_exit_size): return -1, NoOrder

        # Exiting short position thus set short_open_order_size to nan
        open_order_tracker['short_open_order_size'][c.from_col] = np.nan

        return c.from_col, nb.order_nb(
            size=-short_exit_size,
            price=close_price_now,
            direction=Direction.Both,
            fees=fees_now,
            slippage=(tick_size_now * slippage_now) / close_price_now
        )
    return -1, NoOrder  # No Order Placed After All This Chaos


@vbt.jitted
def post_order_func_nb(c, open_order_tracker, entry_and_exit_police,
                       # _____Passed From Settings_____
                       tick_size, typepercent,
                       breakeven_1_trigger_bool, breakeven_1_trigger_points,
                       breakeven_2_trigger_bool, breakeven_2_trigger_points,
                       take_profit_bool, take_profit_points,
                       stop_loss_bool, stop_loss_points):
    """Post Order Function"""

    '''User Settings'''
    tick_size_now = nb.flex_select_auto_nb(tick_size, c.i, c.from_col)
    take_profit_bool_now = nb.flex_select_auto_nb(take_profit_bool, c.i, c.from_col)
    take_profit_trigger_now = nb.flex_select_auto_nb(take_profit_points, c.i, c.from_col)
    stop_loss_bool_now = nb.flex_select_auto_nb(stop_loss_bool, c.i, c.from_col)
    stop_loss_trigger_now = nb.flex_select_auto_nb(stop_loss_points, c.i, c.from_col)
    breakeven_1_trigger_bool_now = nb.flex_select_auto_nb(breakeven_1_trigger_bool, c.i, c.from_col)
    break_even_trigger_1_now = nb.flex_select_auto_nb(breakeven_1_trigger_points, c.i, c.from_col)
    breakeven_2_trigger_bool_now = nb.flex_select_auto_nb(breakeven_2_trigger_bool, c.i, c.from_col)
    break_even_trigger_2_now = nb.flex_select_auto_nb(breakeven_2_trigger_points, c.i, c.from_col)

    if c.order_result.status == OrderStatus.Filled:
        '''An Entry or Exit Ordered Was Filled'''
        direction_now = open_order_tracker['current_order_placed_direction'][c.from_col]
        long_open_order_size = open_order_tracker['long_open_order_size'][c.from_col]
        short_open_order_size = open_order_tracker['short_open_order_size'][c.from_col]
        long_opening_order = open_order_tracker['long_opening_order'][c.from_col]
        short_opening_order = open_order_tracker['short_opening_order'][c.from_col]

        '''Long Orders'''
        if direction_now == 0 and c.order_result.side == OrderSide.Buy and long_opening_order:
            # Used to track the parent price for the current position which guides the take-profits and stop-losses
            # even when using progressive entries
            long_open_order_price = open_order_tracker['long_open_order_price'][c.from_col] = c.order_result.price

            # Set take_profit condition as a percentage of order price or set amount
            if take_profit_bool_now:
                open_order_tracker['long_take_profit_price'][c.from_col] = (1 + (
                    take_profit_trigger_now)) * long_open_order_price if typepercent \
                    else (take_profit_trigger_now * tick_size_now) + long_open_order_price

            # Set stop loss condition as a percentage of order price or set amount
            if stop_loss_bool_now:
                open_order_tracker['long_stop_loss_price'][c.from_col] = (1 + (
                    stop_loss_trigger_now)) * long_open_order_price if typepercent \
                    else (stop_loss_trigger_now * tick_size_now) + long_open_order_price

            # Set break even point 1 condition as a percentage of order price or set amount
            if breakeven_1_trigger_bool_now:
                open_order_tracker['long_break_even_trigger_1_price'][c.from_col] = (1 + (
                    break_even_trigger_1_now)) * long_open_order_price if typepercent \
                    else (break_even_trigger_1_now * tick_size_now) + long_open_order_price

            # Set break even point 2 condition as a percentage of order price or set amount
            if breakeven_2_trigger_bool_now:
                open_order_tracker['long_break_even_trigger_2_price'][c.from_col] = (1 + (
                    break_even_trigger_2_now)) * long_open_order_price if typepercent \
                    else (break_even_trigger_2_now * tick_size_now) + long_open_order_price

        elif direction_now == 0 and np.isnan(long_open_order_size) and c.order_result.side == OrderSide.Sell:
            # Remove open_oder_price and open_order_direction as well as take_profit, stop_loss, break even conditions
            open_order_tracker['long_open_order_price'][c.from_col] = \
                open_order_tracker['long_open_order_size'][c.from_col] = \
                open_order_tracker['long_take_profit_price'][c.from_col] = \
                open_order_tracker['long_stop_loss_price'][c.from_col] = \
                open_order_tracker['long_break_even_trigger_1_price'][c.from_col] = \
                open_order_tracker['long_break_even_trigger_2_price'][c.from_col] = np.nan

        '''Short Orders'''
        if direction_now == 1 and c.order_result.side == OrderSide.Sell and short_opening_order:
            # print('''Short Position Entered''')
            # open_order_price = open_order_tracker[0][c.from_col] = c.order_result.price
            open_order_price = open_order_tracker['short_open_order_price'][c.from_col] = c.order_result.price

            # Set take_profit condition as a percentage of order price or set amount
            if take_profit_bool_now:
                open_order_tracker['short_take_profit_price'][c.from_col] = (1 - (
                    take_profit_trigger_now)) * open_order_price if typepercent \
                    else -(take_profit_trigger_now * tick_size_now) + open_order_price

            # Set stop loss condition as a percentage of order price or set amount
            if stop_loss_bool_now:
                open_order_tracker['short_stop_loss_price'][c.from_col] = (1 - (
                    stop_loss_trigger_now)) * open_order_price if typepercent \
                    else -(stop_loss_trigger_now * tick_size_now) + open_order_price

            # Set break even point 1 condition as a percentage of order price or set amount
            if breakeven_1_trigger_bool_now:
                open_order_tracker['short_break_even_trigger_1_price'][c.from_col] = (1 - (
                    break_even_trigger_1_now)) * open_order_price if typepercent \
                    else -(break_even_trigger_1_now * tick_size_now) + open_order_price

            # Set break even point 2 condition as a percentage of order price or set amount
            if breakeven_2_trigger_bool_now:
                open_order_tracker['short_break_even_trigger_2_price'][c.from_col] = (1 - (
                    break_even_trigger_2_now)) * open_order_price if typepercent \
                    else -(break_even_trigger_2_now * tick_size_now) + open_order_price

        elif direction_now == 1 and np.isnan(short_open_order_size) and c.order_result.side == OrderSide.Buy:
            '''Position Exited'''
            # Remove open_oder_price and open_order_direction as well as take_profit, stop_loss, break even conditions
            open_order_tracker['short_open_order_price'][c.from_col] = \
                open_order_tracker['short_open_order_size'][c.from_col] = \
                open_order_tracker['short_take_profit_price'][c.from_col] = \
                open_order_tracker['short_stop_loss_price'][c.from_col] = \
                open_order_tracker['short_break_even_trigger_1_price'][c.from_col] = \
                open_order_tracker['short_break_even_trigger_2_price'][c.from_col] = np.nan


def Flexible_Simulation_Backtest(runtime_settings, open_data, low_data, high_data, close_data, parameter_data):
    '''What do I do'''
    '''Prepare entries and exits'''
    logger.info('Preparing entries and exits')
    Start_Timer = perf_counter()
    long_entries, long_exits, short_entries, short_exits, strategy_specific_kwargs = \
        runtime_settings["Strategy_Settings"]["Strategy_Name"](open_data, low_data, high_data, close_data,
                                                               parameter_data)
    logger.info(f'Time to Prepare Entries and Exits Signals {perf_counter() - Start_Timer}')

    '''Run Simulation'''
    Start_Timer = perf_counter()
    pf = vbt.Portfolio.from_order_func(
        close_data,
        flex_order_func_nb,
        vbt.Rep('long_entries'), vbt.Rep('long_exits'), vbt.Rep('short_entries'), vbt.Rep('short_exits'),
        vbt.Rep('order_size'),
        vbt.Rep('size_type'), vbt.Rep('fees'), vbt.Rep('slippage'), vbt.Rep('tick_size'), vbt.Rep('type_percent'),
        vbt.Rep('breakeven_1_distance_points'),
        vbt.Rep('breakeven_2_distance_points'),
        #
        vbt.Rep('long_progressive_condition'), vbt.Rep('short_progressive_condition'),
        #
        vbt.Rep('progressive_bool'),
        vbt.Rep('allow_multiple_trade_from_entries'),
        vbt.Rep('exit_on_opposite_direction_entry'),
        high=high_data,
        low=low_data,
        pre_sim_func_nb=pre_sim_func_nb,
        post_order_func_nb=post_order_func_nb,
        post_order_args=(vbt.Rep('tick_size'),
                         vbt.Rep('type_percent'),
                         vbt.Rep('breakeven_1_trigger_bool'),
                         vbt.Rep('breakeven_1_trigger_points'),

                         vbt.Rep('breakeven_2_trigger_bool'),
                         vbt.Rep('breakeven_2_trigger_points'),

                         vbt.Rep('take_profit_bool'), vbt.Rep('take_profit_points'),
                         vbt.Rep('stop_loss_bool'), vbt.Rep('stop_loss_points'),),
        broadcast_named_args=dict(  # broadcast against each other
            #
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,

            order_size=runtime_settings["Portfolio_Settings"]['size'],
            size_type=1 if runtime_settings["Portfolio_Settings"]['size_type'] == 'cash' else 0,
            fees=runtime_settings["Portfolio_Settings"]['trading_fees'],
            slippage=runtime_settings["Portfolio_Settings"]['slippage'],
            tick_size=runtime_settings["Data_Settings"]['tick_size'],
            type_percent=runtime_settings["Portfolio_Settings"]['type_percent'],

            # strategy Specific Kwargs
            long_progressive_condition=strategy_specific_kwargs['long_progressive_condition'],
            short_progressive_condition=strategy_specific_kwargs['short_progressive_condition'],
            #
            progressive_bool=strategy_specific_kwargs['progressive_bool'],
            allow_multiple_trade_from_entries=strategy_specific_kwargs['allow_multiple_trade_from_entries'],
            exit_on_opposite_direction_entry=strategy_specific_kwargs['exit_on_opposite_direction_entry'],

            #
            breakeven_1_trigger_bool=strategy_specific_kwargs['breakeven_1_trigger_bool'],
            breakeven_1_trigger_points=strategy_specific_kwargs['breakeven_1_trigger_points'],
            breakeven_1_distance_points=strategy_specific_kwargs['breakeven_1_distance_points'],
            #
            breakeven_2_trigger_bool=strategy_specific_kwargs['breakeven_2_trigger_bool'],
            breakeven_2_trigger_points=strategy_specific_kwargs['breakeven_2_trigger_points'],
            breakeven_2_distance_points=strategy_specific_kwargs['breakeven_2_distance_points'],
            #
            take_profit_bool=strategy_specific_kwargs['take_profit_bool'],
            take_profit_points=strategy_specific_kwargs['take_profit_points'],
            #
            stop_loss_bool=strategy_specific_kwargs['stop_loss_bool'],
            stop_loss_points=-abs(strategy_specific_kwargs['stop_loss_points']),

        ),
        flexible=True,
        max_orders=close_data.shape[0] * 2,  # do not change
        freq=runtime_settings["Data_Settings"]['timeframe'],
        init_cash=runtime_settings["Portfolio_Settings"]['init_cash'],
        # cash_sharing=runtime_settings["Portfolio_Settings"]['cash_sharing'],
        # group_by=runtime_settings["Portfolio_Settings"]['group_by'],
        chunked=chunked,
    )
    logger.info(f'Time to Run Portfolio Simulation {perf_counter() - Start_Timer}')

    extra_info_dtype = np.dtype([
        ('risk_reward_ratio', 'f8'),  # metric for loss function
        ('init_cash_div_order_size', 'f8'),  # needed to compute risk adjusted returns
        ('risk_adjusted_return', 'f8'),  # metric for loss function
    ])

    number_of_parameter_comb = parameter_data.shape[0]
    number_of_assets = len(close_data.keys())
    extra_info = np.empty(number_of_parameter_comb, dtype=extra_info_dtype)
    extra_info['risk_reward_ratio'] = np.divide(strategy_specific_kwargs['take_profit_point_parameters'],
                                                abs(strategy_specific_kwargs['stop_loss_points_parameters'])) if \
        strategy_specific_kwargs['take_profit_bool'] and strategy_specific_kwargs['stop_loss_bool'] else np.zeros(
        shape=np.array(strategy_specific_kwargs['take_profit_point_parameters']).shape)
    extra_info['init_cash_div_order_size'] = [np.divide(
        np.multiply(runtime_settings['Portfolio_Settings']['init_cash'], number_of_assets),
        runtime_settings['Portfolio_Settings']['size'])] * number_of_parameter_comb
    # """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
    # extra_info = np.empty(number_of_parameter_comb * number_of_assets, dtype=extra_info_dtype)
    # take_profit_point_parameters = [680, 480]
    # stop_loss_points_parameters = [-380, -280]
    # from genie_trader.utility_modules.Utils import slow_add_flatten_lists, append_flatten_lists
    #
    # extra_info['risk_reward_ratio'] = append_flatten_lists([np.divide(take_profit_point_parameters,
    #                                                                   stop_loss_points_parameters)] * number_of_assets)
    # extra_info['init_cash_div_order_size'] = append_flatten_lists(
    #     [[np.divide(runtime_settings['Portfolio_Settings']['init_cash'],
    #                 runtime_settings['Portfolio_Settings'][
    #                     'size'])] * number_of_parameter_comb] * number_of_assets)
    # for i in extra_info:
    #     print(
    #         f"{i}\n"
    #     )
    # # exit()
    # """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
    # """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
    #
    # portfolio_combined = pf.stats(agg_func=None).replace(
    #     [np.inf, -np.inf], np.nan, inplace=False)
    # print(f'\n{portfolio_combined = }')  # For Debugging    # For Debugging    # For Debugging
    #
    portfolio_grouped = pf.stats(agg_func=None, group_by=runtime_settings["Portfolio_Settings"][
        'group_by']).replace(
        [np.inf, -np.inf], np.nan, inplace=False)
    print(f'\n{portfolio_grouped = }')  # For Debugging    # For Debugging    # For Debugging
    #
    # """'''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
    return pf, extra_info


def Flexible_Simulation_Optimization(runtime_settings,
                                     open_data, low_data, high_data, close_data,
                                     long_entries, long_exits, short_entries, short_exits,
                                     strategy_specific_kwargs, number_of_parameter_comb):
    # close_data = pd.DataFrame({
    #     #     0  1  2  3  4  5  6  7
    #     'a': [1, 2, 3, 4, 3, 4, 5, 1],
    #     # 'b': [10, 11, 15, 12, 10, 5, 9, 9]
    # })
    # low_data = pd.DataFrame({
    #     #     0  1  2  3  4  5  6  7
    #     'a': [i - 1 for i in close_data['a'].to_numpy()],
    #     # 'b': [10, 11, 15, 12, 10, 5, 9, 9]
    # })
    # high_data = pd.DataFrame({
    #     #     0  1  2  3  4  5  6  7
    #     'a': [i + 1 for i in close_data['a'].to_numpy()],
    #     # 'b': [10, 11, 15, 12, 10, 5, 9, 9]
    # })
    #
    # long_entries = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [True, False, False, False, True, False, False, False],
    #     # 'b': [True, False, False, False, True, False, False, False]
    # })
    # long_exits = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [False, False, False, True, False, False, True, False],
    #     # 'b': [False, False, False, True, False, False, False, True]
    # })
    # #
    # short_entries = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [False, False, True, False, False, False, False, False],
    #     # 'b': [True, False, False, False, True, False, False, False]
    # })
    # short_exits = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [False, False, False, False, False, True, False, False],
    #     # 'b': [False, False, False, True, False, False, False, True]
    # })
    #
    # strategy_specific_kwargs['take_profit_bool'] = True
    # strategy_specific_kwargs['take_profit_points'] = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [1, 1, 1, 1, 1, 1, 1, 1],
    #     # 'b': [False, False, False, True, False, False, False, True]
    # })
    # #
    # strategy_specific_kwargs['stop_loss_bool'] = True
    # strategy_specific_kwargs['stop_loss_points'] = pd.DataFrame({
    #     #       0     1      2      3      4      5      6      7
    #     'a': [-1, -1, -1, -1, -1, -1, -1, -1],
    #     # 'b': [False, False, False, True, False, False, False, True]
    # })

    """What do I do"""
    '''Run Simulation'''
    Start_Timer = perf_counter()
    pf = vbt.Portfolio.from_order_func(
        close_data,
        flex_order_func_nb,
        vbt.Rep('long_entries'), vbt.Rep('long_exits'), vbt.Rep('short_entries'), vbt.Rep('short_exits'),
        vbt.Rep('order_size'),
        vbt.Rep('size_type'), vbt.Rep('fees'), vbt.Rep('slippage'), vbt.Rep('tick_size'), vbt.Rep('type_percent'),
        vbt.Rep('breakeven_1_distance_points'),
        vbt.Rep('breakeven_2_distance_points'),
        #
        vbt.Rep('long_progressive_condition'), vbt.Rep('short_progressive_condition'),
        #
        vbt.Rep('progressive_bool'),
        vbt.Rep('allow_multiple_trade_from_entries'),
        vbt.Rep('exit_on_opposite_direction_entry'),
        high=high_data,
        low=low_data,
        pre_sim_func_nb=pre_sim_func_nb,
        post_order_func_nb=post_order_func_nb,
        post_order_args=(vbt.Rep('tick_size'),
                         vbt.Rep('type_percent'),
                         vbt.Rep('breakeven_1_trigger_bool'),
                         vbt.Rep('breakeven_1_trigger_points'),

                         vbt.Rep('breakeven_2_trigger_bool'),
                         vbt.Rep('breakeven_2_trigger_points'),

                         vbt.Rep('take_profit_bool'), vbt.Rep('take_profit_points'),
                         vbt.Rep('stop_loss_bool'), vbt.Rep('stop_loss_points'),),
        broadcast_named_args=dict(  # broadcast against each other
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            #
            order_size=runtime_settings["Portfolio_Settings.size"],
            size_type=1 if runtime_settings["Portfolio_Settings.size_type"] == 'cash' else 0,
            fees=runtime_settings["Portfolio_Settings.trading_fees"],
            slippage=runtime_settings["Portfolio_Settings.slippage"],
            tick_size=runtime_settings["Data_Settings.tick_size"],
            type_percent=runtime_settings["Portfolio_Settings.type_percent"],

            # strategy Specific Kwargs
            long_progressive_condition=strategy_specific_kwargs['long_progressive_condition'],
            short_progressive_condition=strategy_specific_kwargs['short_progressive_condition'],
            #
            progressive_bool=strategy_specific_kwargs['progressive_bool'],
            allow_multiple_trade_from_entries=strategy_specific_kwargs['allow_multiple_trade_from_entries'],
            exit_on_opposite_direction_entry=strategy_specific_kwargs['exit_on_opposite_direction_entry'],

            #
            breakeven_1_trigger_bool=strategy_specific_kwargs['breakeven_1_trigger_bool'],
            breakeven_1_trigger_points=strategy_specific_kwargs['breakeven_1_trigger_points'],
            breakeven_1_distance_points=strategy_specific_kwargs['breakeven_1_distance_points'],
            #
            breakeven_2_trigger_bool=strategy_specific_kwargs['breakeven_2_trigger_bool'],
            breakeven_2_trigger_points=strategy_specific_kwargs['breakeven_2_trigger_points'],
            breakeven_2_distance_points=strategy_specific_kwargs['breakeven_2_distance_points'],
            #
            take_profit_bool=strategy_specific_kwargs["take_profit_bool"],
            take_profit_points=strategy_specific_kwargs["take_profit_points"],
            #
            stop_loss_bool=strategy_specific_kwargs["stop_loss_bool"],
            stop_loss_points=-abs(strategy_specific_kwargs["stop_loss_points"]),

        ),
        flexible=True,
        max_orders=close_data.shape[0] * 2,  # do not change
        freq=runtime_settings["Portfolio_Settings.sim_timeframe"],
        init_cash=runtime_settings["Portfolio_Settings.init_cash"],
        # cash_sharing=True,
        # group_by=runtime_settings["Portfolio_Settings"]['group_by'],
        chunked=chunked,
    )
    logger.info(f'Time to Run Portfolio Simulation {perf_counter() - Start_Timer}')

    extra_info_dtype = np.dtype([
        ("risk_reward_ratio", 'f8'),  # metric for loss function
        ("init_cash_div_order_size", 'f8'),  # needed to compute risk adjusted returns
        ("risk_adjusted_return", 'f8'),  # metric for loss function
    ])

    '''Extra Information Needed For Loss Function'''
    number_of_assets = len(close_data.keys())
    extra_info = np.empty(number_of_parameter_comb, dtype=extra_info_dtype)

    extra_info["risk_reward_ratio"] = np.divide(strategy_specific_kwargs["take_profit_point_parameters"],
                                                abs(strategy_specific_kwargs["stop_loss_points_parameters"])) if \
        strategy_specific_kwargs["take_profit_bool"] and strategy_specific_kwargs["stop_loss_bool"] else np.zeros(
        shape=np.array(strategy_specific_kwargs["take_profit_point_parameters"]).shape)
    extra_info["init_cash_div_order_size"] = [np.divide(
        np.multiply(runtime_settings["Portfolio_Settings.init_cash"], number_of_assets),
        runtime_settings["Portfolio_Settings.size"])] * number_of_parameter_comb

    # from genie_trader.data_n_analysis_modules.Analyze import Compute_Stats
    # Portfolio_Stats = Compute_Stats(pf)[
    #     ['Total Trades', 'Win Rate [%]', 'Avg Winning Trade [%]', 'Avg Losing Trade [%]']]
    # print(f'{Portfolio_Stats}')
    #
    # # print(f'\n\n '
    # #       # f'{pf.orders.records_readable}'
    # #       f'\n\n '
    # #       f'{pf.positions.records_readable[["Position Id", "Size", "Direction", "Entry Timestamp", "Exit Timestamp"]].head()} '
    # #       f'\n{pf.positions.records_readable[["Avg Entry Price", "Avg Exit Price", "PnL", "Return"]].head(200)} '
    # #       f'\n\n')
    #
    # pnl = []
    # for value in pf.positions.records_readable["PnL"]:
    #     if value > 0:
    #         pnl.append(value)
    # print(f'{pnl = }')
    # exit()
    return pf, extra_info
