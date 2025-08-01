import click
from skema.lib import classify


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Input directory with image and mask files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF.')
def main(input_dir, output_filename):
    """Classify a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats (same as in lib.py)
    mean_per_channel = [ 1.93357159e+02,  2.53693333e+02,  1.41648022e+02,  9.99292362e+02,
  3.21693919e+02,  1.30867292e+00,  2.63550136e+00,  6.49704998e-02,
  1.57273007e-01  , -1.57273007e-01,  1.82229161e+07,  1.09806622e-01]

    std_per_channel = [1.55697494e+02, 2.12700364e+02, 2.04018106e+02, 1.27588129e+03,
 3.77324432e+02, 1.33938435e+00, 2.14640498e+02, 6.75251176e-01,
 7.32966188e-01, 7.32966188e-01, 2.16768826e+10, 4.11232123e-01]

    classify(input_dir, output_filename, mean_per_channel, std_per_channel)

if __name__ == '__main__':
    main()