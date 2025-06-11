import importlib
import common_imports
importlib.reload(common_imports)
from common_imports import *
import pickle

# Show learned buy and sell signals

import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_signals(csv_file, df, output_dir=None, debug=False):
    try:

        print(df)
        # Load signals from CSV file
        df_signals = pd.read_csv(csv_file)
        
        # Get signals and title
        buy_signals = eval(df_signals['buy_signals'][0])
        sell_signals = eval(df_signals['sell_signals'][0])
        model_name = df_signals['model'][0]
        reward = df_signals['reward'][0]
        title = f"RL Model Buy/Sell Signals - {model_name} (Reward: {reward:.2f})"
        
        # Get max_steps from CSV
        max_steps = int(df_signals['max_steps'][0])
        
        # Convert signals to integers if they're strings
        if isinstance(buy_signals[0], str):
            buy_signals = [int(sig) for sig in buy_signals]
        if isinstance(sell_signals[0], str):
            sell_signals = [int(sig) for sig in sell_signals]
        

        # Get the original DataFrame from the common_imports
        df = df_original.copy()

        # Ensure we have a proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Get the first and last timestamps
        start_time = df.index[0]
        end_time = df.index[max_steps-1]
        
        # Ensure we have a proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Reset index to get datetime as a column
        df = df.reset_index()
        
        # Filter DataFrame to only include tested steps using integer index
        df_tested = df.iloc[:max_steps]
        
        # Set datetime as index again
        df_tested.set_index('datetime', inplace=True)
        
        # Debug print the datetime values
        print("\nDebug datetime info:")
        print(f"First few datetime values: {df_tested.index[:5].tolist()}")
        print(f"First few original datetime values: {df.index[:5].tolist()}")
        print(f"First few tested datetime values: {df_tested.index[:5].tolist()}")
        print(f"Index type: {type(df_tested.index[0])}")
        print(f"Index year: {df_tested.index[0].year}")
        print(f"Index month: {df_tested.index[0].month}")
        print(f"Index day: {df_tested.index[0].day}")
        
        # Filter signals to only include those within the tested range
        buy_signals = [i for i in buy_signals if 0 <= i < max_steps]
        sell_signals = [i for i in sell_signals if 0 <= i < max_steps]

        if debug:
            print(f"\nDebug information for {csv_file}:")
            print(f"Max steps: {max_steps}")
            print(f"Buy signals: {buy_signals}")
            print(f"Sell signals: {sell_signals}")
            print(f"Data range: {df_tested.index[0]} to {df_tested.index[-1]}")
            print(f"Number of data points: {len(df_tested)}")
            print(f"Number of buy signals: {len(buy_signals)}")
            print(f"Number of sell signals: {len(sell_signals)}")
            print(f"Original DataFrame length: {len(df)}")
            print(f"Filtered DataFrame length to plot: {len(df_tested)}")
            print(f"Number of unique dates: {len(df_tested.index.unique())}")
            print(f"Date frequency: {pd.infer_freq(df_tested.index)}")
            print(f"First few datetime values: {df_tested.index[:5].tolist()}")
            print(f"First few original datetime values: {df['datetime'].head().tolist()}")
            print(f"First few tested datetime values: {df_tested.index[:5].tolist()}")
            print(f"Index type: {type(df_tested.index[0])}")
            print(f"Index year: {df_tested.index[0].year}")
            print(f"Index month: {df_tested.index[0].month}")
            print(f"Index day: {df_tested.index[0].day}")
        
        # Filter signals to only include those within the tested range
        buy_signals = [i for i in buy_signals if 0 <= i < max_steps]
        sell_signals = [i for i in sell_signals if 0 <= i < max_steps]

        # Create addplots for buy and sell signals
        addplots = []
        if buy_signals:
            # Create a DataFrame with all points, setting non-signal points to NaN
            buy_signals_df = pd.DataFrame(
                data={'close': np.nan},
                index=df_tested.index
            )
            buy_signals_df.loc[df_tested.index[buy_signals], 'close'] = df_tested.iloc[buy_signals]['close']
            buy_signal_plot = mpf.make_addplot(buy_signals_df['close'], type='scatter', markersize=40, marker='^', color='green')
            addplots.append(buy_signal_plot)
        
        if sell_signals:
            # Create a DataFrame with all points, setting non-signal points to NaN
            sell_signals_df = pd.DataFrame(
                data={'close': np.nan},
                index=df_tested.index
            )
            sell_signals_df.loc[df_tested.index[sell_signals], 'close'] = df_tested.iloc[sell_signals]['close']
            sell_signal_plot = mpf.make_addplot(sell_signals_df['close'], type='scatter', markersize=40, marker='v', color='red')
            addplots.append(sell_signal_plot)

        if debug:
            print(f"Addplots information:")
            for i, plot in enumerate(addplots):
                print(f"Plot {i+1}:")
                print(f"  Type: {plot['type']}")
                print(f"  Marker: {plot['marker']}")
                print(f"  Color: {plot['color']}")
                print(f"  Number of points: {len(plot['data'])}")
                if len(plot['data']) > 0:
                    print(f"  First point: {plot['data'].iloc[0]}")
                    print(f"  Last point: {plot['data'].iloc[-1]}")
        
        # Plot with the added buy and sell signals
        fig = mpf.plot(
            df_tested,
            type='ohlc',  # Changed from ohlc to line for better performance with large data
            style='yahoo',
            title=title,
            addplot=addplots,
            volume=True,
            show_nontrading=True,
            tight_layout=True,
            figsize=(15, 10),
            datetime_format='%Y-%m-%d %H:%M',
            returnfig=True,
            warn_too_much_data=10000,  # Set high threshold to avoid warning
            xrotation=0,  # Prevent x-axis labels from overlapping
            scale_width_adjustment=dict(volume=0.9)  # Adjust volume width
        )
        
        # Determine output filename
        base_name = os.path.basename(csv_file).replace('.csv', '.png')
        if output_dir is None:
            output_dir = os.path.dirname(csv_file)
        png_file = os.path.join(output_dir, base_name)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot
        plt.savefig(png_file)
        print(f"Saved plot to {png_file}")
        
        plt.close(fig[0])  # Close the figure to free up memory
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        raise

def find_csv_files(input_dir):
    """Find all CSV files in the input directory."""
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith('trade_signals_') and file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)
    return directory

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot trading signals from CSV files')
    parser.add_argument('--input', type=str, default='test_results',
                      help='Input directory containing test results (default: test_results)')
    parser.add_argument('--output', type=str, default='plot_results',
                      help='Output directory for plots (default: plot_results)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    args = parser.parse_args()
    
    # Find all CSV files in the input directory
    csv_files = find_csv_files(args.input)
    
    if not csv_files:
        print(f"No CSV files found in {args.input}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Get the input directory name to use as a subdirectory
            input_dir_name = os.path.basename(os.path.normpath(args.input))
            
            # Create output directory with input directory name as subdirectory
            output_dir = ensure_dir(os.path.join(args.output, input_dir_name))
            
            print(f"\nProcessing: {csv_file}")
            print(f"Output directory: {output_dir}")
            
            # Plot signals and save directly to output directory
            plot_signals(csv_file, df, output_dir=output_dir, debug=args.debug)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue