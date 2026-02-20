import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

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
    
    # Print configuration info
    click.echo("\n" + "="*60)
    click.echo("SKEMA - Kelp Classification Tool")
    click.echo("="*60)
    click.echo(f"Input Directory:  {input_dir}")
    click.echo(f"Output Filename:  {output_filename}")
    click.echo(f"Model Type:       {model_type}")
    
    # Check GPU availability
    import torch
    device = "GPU" if torch.cuda.is_available() else "CPU"
    click.echo(f"Computing Device: {device}")
    click.echo("="*60 + "\n")
    
    # Define the normalization stats based on model type
    if model_type == 'model_full':
        mean_per_channel = [2.02127847e+02, 2.64991799e+02, 1.45913497e+02, 9.57456953e+02,
                            3.20302883e+02, 1.37548690e+00, -3.29322396e-01, 3.47405153e+01,
                            3.66107438e-02, 1.84492036e-01, -1.84492036e-01, 2.22893410e+07,
                            9.99078992e-02]
        std_per_channel = [1.61504107e+02, 2.22303637e+02, 2.03997451e+02, 1.26105656e+03,
                            3.79069759e+02, 1.36767747e+00, 2.02345453e+02, 2.60894243e+01,
                            6.71879776e-01, 7.23202999e-01, 7.23202999e-01, 2.25708723e+10,
                            4.06695959e-01]

    else:  # model_s2bandsandindices_only                       
        mean_per_channel = [2.08900522e+02, 2.70272557e+02, 1.52312422e+02, 9.87182507e+02,
                            3.26650321e+02, 5.65647882e-02, 1.62913280e-01, -1.62913280e-01,
                            1.90832012e+07, 1.09887867e-01]
        std_per_channel = [1.59021908e+02, 2.18393833e+02, 2.08355086e+02, 1.29667310e+03,
                            3.79794112e+02, 6.53314129e-01, 7.11820223e-01, 7.11820223e-01,
                            2.07060299e+10, 3.90619966e-01]


    segment(input_dir, output_filename, mean_per_channel, std_per_channel, model_type)

if __name__ == '__main__':
    main()