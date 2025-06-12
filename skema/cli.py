import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [ 8.23703725e+02,  5.95571813e+02,  3.41566639e+02,  9.07368874e+02,
  4.33726076e+02,  1.36830877e+00, -4.04446852e+00,  4.73336378e-02,
  1.71122954e-01, -1.71122954e-01,  1.98505740e+06,  4.72605017e-02]

    std_per_channel = [1.29613479e+02, 1.60648936e+02, 1.67238334e+02, 1.05605174e+03,
 2.94992568e+02, 1.35160295e+00, 2.12205155e+02, 4.77127918e-01,
 5.15707643e-01, 5.15707643e-01, 7.86553735e+09, 1.86447174e-01]

    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()