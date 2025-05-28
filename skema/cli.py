import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [ 8.19832209e+02  5.92348544e+02  3.37563632e+02  9.07100408e+02
  4.31277521e+02  1.36518478e+00 -4.12282944e+00  4.95029905e-02
  1.69760541e-01 -1.69760541e-01  3.26501326e-01  4.90320325e-02]

    std_per_channel = [1.20800603e+02 1.52516683e+02 1.55277863e+02 1.05305242e+03
 2.89233658e+02 1.34969885e+00 2.17514203e+02 4.79392376e-01
 5.17476406e-01 5.17476406e-01 1.44774196e+00 1.87423892e-01]


    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()