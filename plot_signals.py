from common_imports import df, df_original
import pickle

# Show learned buy and sell signals

import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

def plot_signals(buy_signals, sell_signals, title):
    # Filter signals to only include those within the DataFrame's length
    buy_signals = [i for i in buy_signals if i >= 0 and i < len(df_original)]
    sell_signals = [i for i in sell_signals if i >= 0 and i < len(df_original)]

    # Convert step indices to datetime indices
    buy_signals = [df_original.index[i] for i in buy_signals]
    sell_signals = [df_original.index[i] for i in sell_signals]

    # Create addplots for buy and sell signals
    addplots = []
    if buy_signals:
        buy_signal_prices = df_original['close'].copy()
        buy_signal_prices[~df_original.index.isin(buy_signals)] = float('nan')
        buy_signal_plot = mpf.make_addplot(buy_signal_prices, type='scatter', markersize=40, marker='^', color='green')
        addplots.append(buy_signal_plot)
    
    if sell_signals:
        sell_signal_prices = df_original['close'].copy()
        sell_signal_prices[~df_original.index.isin(sell_signals)] = float('nan')
        sell_signal_plot = mpf.make_addplot(sell_signal_prices, type='scatter', markersize=40, marker='v', color='red')
        addplots.append(sell_signal_plot)

    # Plot with the added buy and sell signals
    fig, axes = mpf.plot(
        df_original,
        type='ohlc',
        datetime_format='%Y-%m-%d %H:%M',
        addplot=addplots,
        returnfig=True,
        figsize=(16, 8),
        warn_too_much_data=10000,
        title=title
    )

    return fig, axes

def main():
    # Plot signals for both models
    model_names = ['best_model', 'final_model']
    for model_name in model_names:
        print(f"\nPlotting signals for {model_name}...")
        try:
            # Load signals from CSV file
            df_signals = pd.read_csv(f'trade_signals_{model_name}.csv')
            
            # Get signals
            buy_signals = eval(df_signals['buy_signals'][0])
            sell_signals = eval(df_signals['sell_signals'][0])
            
            # Plot the signals
            fig, axes = plot_signals(
                buy_signals,
                sell_signals,
                title=f"RL Model Buy/Sell Signals - {model_name}"
            )
            
            # Save the plot
            plt.savefig(f'trade_signals_{model_name}.png')
            print(f"Saved plot to trade_signals_{model_name}.png")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Plot signals for both models
    model_names = ['best_model', 'final_model']
    for model_name in model_names:
        print(f"\nPlotting signals for {model_name}...")
        try:
            # Load signals from CSV file
            df_signals = pd.read_csv(f'trade_signals_{model_name}.csv')
            
            # Get signals
            buy_signals = eval(df_signals['buy_signals'][0])
            sell_signals = eval(df_signals['sell_signals'][0])
            
            # Plot the signals
            fig, axes = plot_signals(
                buy_signals,
                sell_signals,
                title=f"RL Model Buy/Sell Signals - {model_name}"
            )
            
            # Save the plot
            plt.savefig(f'trade_signals_{model_name}.png')
            print(f"Saved plot to trade_signals_{model_name}.png")
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")
            continue