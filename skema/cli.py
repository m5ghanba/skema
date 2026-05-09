import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import click
from skema.lib import segment, create_mosaic


@click.command(name="classify")
@click.option('--input-dir', required=True, help='Full path to a single .SAFE file, e.g., C:\\...\\S2C_...SAFE., OR (when using --batch-dir) to a directory containing multiple unzipped .SAFE files.')
@click.option('--output-filename', required=True, help='Filename for the output prediction TIFF, e.g., output.tif. In batch mode, this is used as a postfix appended to each scene name.')
@click.option('--model-type', 
              type=click.Choice(['model_full', 'model_s2bandsandindices_only', 'model_ensemble'], case_sensitive=False),
              default='model_s2bandsandindices_only',
              help='Model type to use: "model_full" (with substrate, bathymetry, and slope), "model_s2bandsandindices_only" (S2 bands and indices only), or "model_ensemble" (ensemble of both models). Default: model_s2bandsandindices_only.')
@click.option('--batch-dir', is_flag=True, default=False,
              help='When set, --input-dir is treated as a directory containing multiple unzipped .SAFE folders. '
                   'Each scene is processed individually and a mosaic (mosaic_kelp_map.tif) is created in --input-dir afterwards.')
@click.option('--soft-substrate-masking', is_flag=True, default=False,
              help='When set, an additional substrate-masked prediction is saved alongside each output, '
                   'where kelp pixels overlapping substrate classes 3 or 4 are set to 0 (no kelp). '
                   'Only valid for model_full and model_ensemble. '
                   'In batch mode, a second substrate-masked mosaic is also created.')
@click.option('--use-bops-substrate', is_flag=True, default=False,
              help='When set, use BoPs substrate files and the BoPs-trained model weights. '
                   'When not set (default), RF substrate files and RF-trained model weights are used. '
                   'Only valid for model_full and model_ensemble.')
@click.option('--eelgrass-masking', is_flag=True, default=False,
              help='When set, kelp pixels that fall within eelgrass polygons (BCMCA eelgrass dataset) '
                   'are reclassified to 0 (no kelp) in a combined _masked.tif output. '
                   'Valid for all model types. Can be combined with --soft-substrate-masking.')
def main(input_dir, output_filename, model_type, batch_dir, soft_substrate_masking, use_bops_substrate, eelgrass_masking):
    """Segment a Sentinel-2 scene and output a kelp mask.

    Single scene:
        skema --input-dir "path/to/scene.SAFE" --output-filename output.tif

    Batch (multiple scenes + mosaic):
        skema --input-dir "path/to/safe_files/" --output-filename output.tif --batch-dir
    """

    # Print configuration info
    click.echo("\n" + "="*60)
    click.echo("SKEMA - Kelp Classification Tool")
    click.echo("="*60)
    click.echo(f"Input:            {input_dir}")
    click.echo(f"Output Filename:  {output_filename}")
    click.echo(f"Model Type:       {model_type}")
    click.echo(f"Batch Mode:       {'Yes' if batch_dir else 'No'}")

    # Check GPU availability
    import torch
    device = "GPU" if torch.cuda.is_available() else "CPU"
    click.echo(f"Computing Device: {device}")
    click.echo(f"Soft Substrate Masking: {'Yes' if soft_substrate_masking else 'No'}")
    click.echo(f"Substrate Source:       {'None' if (model_type == 'model_s2bandsandindices_only' and not soft_substrate_masking) else ('BoPs' if use_bops_substrate else 'RF Model')}")
    click.echo(f"Eelgrass Masking:       {'Yes' if eelgrass_masking else 'No'}")
    click.echo("="*60 + "\n")

    # # Validate --soft-substrate-masking is not used with band-only model
    # if soft_substrate_masking and model_type == 'model_s2bandsandindices_only':
    #     raise click.UsageError(
    #         "--soft-substrate-masking requires a substrate channel and cannot be used with "
    #         "'model_s2bandsandindices_only'. "
    #         "Please use --model-type model_full or --model-type model_ensemble."
    #     )

    # # Validate --use-bops-substrate is not used with band-only model
    # if use_bops_substrate and model_type == 'model_s2bandsandindices_only':
    #     raise click.UsageError(
    #         "--use-bops-substrate requires a substrate channel and cannot be used with "
    #         "'model_s2bandsandindices_only'. "
    #         "Please use --model-type model_full or --model-type model_ensemble."
    #     )

    # Define the normalization stats based on model type and substrate source
    if model_type == 'model_full' or model_type == 'model_ensemble':
        if use_bops_substrate:
            # BoPs substrate stats (6th value differs: substrate channel mean/std)
            mean_per_channel = [2.02127847e+02, 2.64991799e+02, 1.45913497e+02, 9.57456953e+02,
                                3.20302883e+02, 8.15331190e-01, -6.30723576e+00, 7.60650406e+00,
                                3.66107438e-02, 1.84492036e-01, -1.84492036e-01, 1.56750584e+00,
                                9.99078992e-02]
            std_per_channel = [1.61504107e+02, 2.22303637e+02, 2.03997451e+02, 1.26105656e+03,
                                3.79069759e+02, 1.09273473e+00, 2.35930351e+02, 1.14107889e+01,
                                6.71879776e-01, 7.23202999e-01, 7.23202999e-01, 3.86945642e+00,
                                4.06695959e-01]
        else:
            # RF substrate stats (default)
            mean_per_channel = [2.02127847e+02, 2.64991799e+02, 1.45913497e+02, 9.57456953e+02,
                                3.20302883e+02, 1.37548690e+00, -6.30723576e+00, 7.60650406e+00,
                                3.66107438e-02, 1.84492036e-01, -1.84492036e-01, 1.56750584e+00,
                                9.99078992e-02]
            std_per_channel = [1.61504107e+02, 2.22303637e+02, 2.03997451e+02, 1.26105656e+03,
                                3.79069759e+02, 1.36767732e+00, 2.35930351e+02, 1.14107889e+01,
                                6.71879776e-01, 7.23202999e-01, 7.23202999e-01, 3.86945642e+00,
                                4.06695959e-01]

    else:  # model_s2bandsandindices_only                       
        mean_per_channel = [2.08900522e+02, 2.70272557e+02, 1.52312422e+02, 9.87182507e+02,
                            3.26650321e+02, 5.65647882e-02, 1.62913280e-01, -1.62913280e-01,
                            1.63743045e+00, 1.09887867e-01]
        std_per_channel = [1.59021908e+02, 2.18393833e+02, 2.08355086e+02, 1.29667310e+03,
                            3.79794112e+02, 6.53314129e-01, 7.11820223e-01, 7.11820223e-01,
                            3.84363017e+00, 3.90619966e-01]

    # ------------------------------------------------------------------ #
    #  BATCH MODE                                                          #
    # ------------------------------------------------------------------ #
    if batch_dir:
        if not os.path.isdir(input_dir):
            raise click.BadParameter(
                f"--input-dir must be an existing directory when --batch-dir is set. Got: {input_dir}"
            )

        # Find all .SAFE folders directly inside input_dir
        safe_folders = sorted([
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if d.upper().endswith('.SAFE') and os.path.isdir(os.path.join(input_dir, d))
        ])

        if not safe_folders:
            raise click.ClickException(f"No .SAFE folders found in {input_dir}")

        click.echo(f"Found {len(safe_folders)} .SAFE folder(s) to process.\n")

        processed_output_paths = []

        for idx, safe_path in enumerate(safe_folders, start=1):
            safe_basename = os.path.basename(safe_path).replace('.SAFE', '')
            # Build per-scene output filename: <SAFE_basename>_<output_filename>
            scene_output_filename = f"{safe_basename}_{output_filename}"

            click.echo(f"[{idx}/{len(safe_folders)}] Processing: {safe_basename}")

            segment(safe_path, scene_output_filename, mean_per_channel, std_per_channel, model_type, soft_substrate_masking, use_bops_substrate, eelgrass_masking)

            # The segment() function saves the result inside a folder named after the SAFE basename,
            # located next to the .SAFE file (i.e. inside input_dir).
            output_folder = os.path.join(input_dir, safe_basename)
            output_tif_path = os.path.join(output_folder, scene_output_filename)
            processed_output_paths.append(output_tif_path)


        # Create mosaic from all processed outputs
        click.echo("="*60)
        click.echo("All scenes processed. Creating mosaic...")
        mosaic_path = os.path.join(input_dir, "mosaic_kelp_map.tif")
        create_mosaic(processed_output_paths, mosaic_path, soft_substrate_masking=soft_substrate_masking, eelgrass_masking=eelgrass_masking)
        click.echo("="*60 + "\n")

    # ------------------------------------------------------------------ #
    #  SINGLE-SCENE MODE (original behaviour)                             #
    # ------------------------------------------------------------------ #
    else:
        if not input_dir.upper().endswith('.SAFE'):
            raise click.BadParameter(
                f"--input-dir must point to a .SAFE folder for single-scene mode. Got: {input_dir}\n"
                f"  Tip: If you want to process multiple scenes at once, use the --batch-dir flag."
            )
        if not os.path.isdir(input_dir):
            raise click.BadParameter(
                f"The path '{input_dir}' does not exist or is not a directory."
            )
        segment(input_dir, output_filename, mean_per_channel, std_per_channel, model_type, soft_substrate_masking, use_bops_substrate, eelgrass_masking)


if __name__ == '__main__':
    main()