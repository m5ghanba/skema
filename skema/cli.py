import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [ 8.24242699e+02,  5.94452579e+02,  3.40587609e+02,  9.08875000e+02,
  4.29146186e+02,  1.13989937e+00, -1.49963112e+01,  4.25161337e-02,
  1.75753714e-01, -1.75753714e-01,  3.27247919e-01,  4.44581977e-02]

    std_per_channel = [1.25663058e+02, 1.55674389e+02, 1.64844461e+02, 1.07801420e+03,
 2.91511819e+02, 1.32835895e+00, 2.19753140e+02, 4.76854661e-01,
 5.16788840e-01, 5.16788840e-01, 1.48334234e+00, 1.85546163e-01]

    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()