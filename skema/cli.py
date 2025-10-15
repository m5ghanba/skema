import click
from skema.lib import segment


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Full path to the .SAFE file, e.g., C:\...\S2C_MSIL2A_20250715T194921_N0511_R085_T09UUU_20250716T001356.SAFE.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF, e.g., output.tif.')
@click.option('--model-type', 
              type=click.Choice(['model_full', 'model_s2bandsandindices_only'], case_sensitive=False),
              default='model_full',
              help='Model type to use: "model_full" (with substrate/bathymetry) or "model_s2bandsandindices_only" (S2 bands and indices only). Default: model_full.')
def main(input_dir, output_filename, model_type):
    """Segment a Sentinel-2 scene and output a kelp mask."""
    # Define the normalization stats based on model type
    if model_type == 'model_full':
        mean_per_channel = [1.93357159e+02, 2.53693333e+02, 1.41648022e+02, 9.99292362e+02,
                           3.21693919e+02, 1.30867292e+00, 2.63550136e+00, 6.49704998e-02,
                           1.57273007e-01, -1.57273007e-01, 1.82229161e+07, 1.09806622e-01]

        std_per_channel = [1.55697494e+02, 2.12700364e+02, 2.04018106e+02, 1.27588129e+03,
                          3.77324432e+02, 1.33938435e+00, 2.14640498e+02, 6.75251176e-01,
                          7.32966188e-01, 7.32966188e-01, 2.16768826e+10, 4.11232123e-01]
    else:  # model_s2bandsandindices_only
        mean_per_channel = [1.93357159e+02, 2.53693333e+02, 1.41648022e+02, 9.99292362e+02,
                           3.21693919e+02, 6.49704998e-02, 1.57273007e-01, -1.57273007e-01,
                           1.82229161e+07, 1.09806622e-01]

        std_per_channel = [1.55697494e+02, 2.12700364e+02, 2.04018106e+02, 1.27588129e+03,
                          3.77324432e+02, 6.75251176e-01, 7.32966188e-01, 7.32966188e-01,
                          2.16768826e+10, 4.11232123e-01]

    segment(input_dir, output_filename, mean_per_channel, std_per_channel, model_type)

if __name__ == '__main__':
    main()