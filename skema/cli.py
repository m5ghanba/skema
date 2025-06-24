import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [1.55891195e+02, 2.12970584e+02, 2.04725875e+02, 1.27900457e+03,
 3.78005877e+02, 1.33968945e+00, 2.13826603e+02, 6.74924441e-01,
 7.32946216e-01, 7.32946216e-01, 2.26199765e+10, 4.11327859e-01]

    std_per_channel = [1.55891195e+02, 2.12970584e+02, 2.04725875e+02, 1.27900457e+03,
 3.78005877e+02, 1.33968945e+00, 2.13826603e+02, 6.74924441e-01,
 7.32946216e-01, 7.32946216e-01, 2.26199765e+10, 4.11327859e-01]


    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()