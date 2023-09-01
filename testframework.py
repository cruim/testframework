import sys

import click

import utils


@click.group(help="TestFramework")
def cli():
    pass


@cli.command(name="available_models", help="List of available models for auto wrapping")
def available_models():
    res = "\n".join("- "+name for name in sorted(utils.model_config().keys()))
    click.echo(f"Available models for automatic wrapping:\n\n{res}")
    return res


@cli.command(name="available_tests", help="List of available tests")
def available_tests():
    message = "Available tests: ift"
    click.echo(message)
    return message


@cli.command(name="ift", help="")
@click.option("--path_to_model", type=click.Path(exists=False), help="")
@click.option("--config", type=click.Path(exists=False), default=None,
              help="Config file for IFT. Available formats: csv, json. Each row have to contain two required columns:"
                   "'vector' and 'expected_result'. 'vector' will be passed to predict. 'expected_result' is result"
                   "of prediction which expected. When completed ")
@click.option("--acceptable_range", type=click.BOOL, default=False, help="Flag")
def ift(config, path_to_model, acceptable_range):
    click.echo("Start IFT")
    exit_code, obj = utils.make_ift(config=config, model=path_to_model, acceptable_range=acceptable_range)
    if exit_code:
        click.echo(f"{obj}")
        return exit_code
    click.echo(f"IFT results: {obj}")
    click.echo("Results saved into ift_report.json")
    sys.exit(0)


@cli.command(name="wrap_model")
@click.option("--model_type", type=click.Choice(utils.model_config().keys(), case_sensitive=False), required=True)
@click.option("--path_to_model", type=click.Path(exists=False), required=True)
@click.option("--wrapped_name", type=str, required=False, default="wrapped_model",
              help="Name for wrapped model, by default is wrapped_model")
def wrap_model(model_type: str, path_to_model, wrapped_name: str):
    click.echo(f"Start wrapping {model_type}")
    utils.wrap(model_type, path_to_model, wrapped_name)
    click.echo(f"Model {wrapped_name} wrapped")
    sys.exit(0)


@cli.command("validate_categorical_feature")
@click.option("--path_to_model", type=click.Path(exists=False), help="")
@click.option("--config", type=click.Path(exists=False), help="ratio[i] is coeff between value[i] and value[i+1]")
def validate_categorical_feature(path_to_model, config):
    click.echo("Start validate_categorical_feature test")
    exit_code, obj = utils.validate_categorical_feature(model=path_to_model, config=config)
    click.echo(obj)


if __name__ == '__main__':
    cli()
