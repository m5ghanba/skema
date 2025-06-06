import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [ 8.18228524e+02,  5.89239079e+02,  3.35254818e+02,  8.89031938e+02,
  4.24965675e+02,  1.38826094e+00, -4.63517145e+00,  4.03730054e-02,
  1.79598210e-01, -1.79598210e-01,  3.03750709e-01,  4.52992752e-02]

    std_per_channel = [1.20862186e+02, 1.53052653e+02, 1.56862150e+02, 1.04906862e+03,
 2.87962356e+02, 1.35024083e+00, 2.10757242e+02, 4.78087799e-01,
 5.16086996e-01, 5.16086996e-01, 1.44287829e+00, 1.86513087e-01]

    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()