import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_pseudo_dice(file_path: str) -> dict:
    epoch_data = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Match epoch number
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                epoch_number = int(epoch_match.group(1))
                # Initialize the entry for this epoch
                epoch_data[epoch_number] = {'Pseudo Dice': None}

            # Match Pseudo dice values
            pseudo_dice_match = re.search(r'Pseudo dice \[(.*?)\]', line)
            if pseudo_dice_match and epoch_number in epoch_data:
                # Extract the float values, removing the 'np.float32(' and ')' parts
                values_str = pseudo_dice_match.group(1)
                # Parse the values, convert to float and handle the np.float32 notation
                values = [float(v.split('np.float32(')[-1].split(')')[0]) for v in values_str.split(',')]
                epoch_data[epoch_number]['Pseudo Dice'] = values

    # Filter for the first 50 epochs and return the results
    return {k: v for k, v in epoch_data.items() if k < 50}

def plot_pseudo_dice_scores(epoch_results: dict, dest: str = None, headless: bool = False) -> None:
    # Extract epoch numbers and Pseudo Dice values
    epochs = list(epoch_results.keys())
    pseudo_dice_values = [data['Pseudo Dice'] for data in epoch_results.values()]

    # Create a DataFrame to store the results
    num_classes = len(pseudo_dice_values[0]) if pseudo_dice_values else 0
    dice_df = pd.DataFrame(pseudo_dice_values, columns=[f'K={i+1}' for i in range(num_classes)])
    dice_df['Epoch'] = epochs

    # Calculate the mean (average) of Pseudo Dice values across all classes
    dice_df['All Classes'] = dice_df[[f'K={i+1}' for i in range(num_classes)]].mean(axis=1)

    # Save Pseudo Dice scores to a CSV file
    dice_df.to_csv('pseudo_dice_scores.csv', index=False)
    print("Pseudo Dice scores saved to pseudo_dice_scores.csv")

    # Plotting
    fig, ax = plt.subplots()
    ax.set_title('Pseudo Dice Scores Over Epochs')

    # Plot the Pseudo Dice values for each class (K=1, K=2, etc.)
    for class_index in range(num_classes):
        ax.plot(epochs, dice_df[f'K={class_index+1}'], label=f'k={class_index+1}', linewidth=2)

    # Plot the average (all classes)
    ax.plot(epochs, dice_df['All Classes'], label='All Classes', linestyle='-', linewidth=2, color='purple')

    ax.legend()
    fig.tight_layout()

    if dest:
        fig.savefig(dest)
        print(f"Plot saved to {dest}")

    if not headless:
        plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Process text file and plot EMA Pseudo Dice Scores')
    parser.add_argument('--file', type=str, required=False, default="c:/Users/varda/Desktop/training_log.txt",
                        help="The text file to process.")
    parser.add_argument('--dest', type=str, default="PLOT.png",
                        help="Optional: save the plot to a .png file")
    parser.add_argument("--headless", action="store_true",
                        help="Does not display the plot and save it directly (implies --dest to be provided.")

    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = get_args()
    # Extract Pseudo Dice data from the specified text file
    epoch_results = extract_pseudo_dice(args.file)
    # Generate the plot using the extracted data
    plot_pseudo_dice_scores(epoch_results, dest=args.dest, headless=args.headless)




