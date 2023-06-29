"""Neocode cli."""
from pathlib import Path

import click

from neocode.data.iss import download_iss

data_modules_map = {
    "iss": download_iss,
}


def str2array(ctx: click.Context, _: str, value: str) -> list:
    """Convert string input by user into an array of values.

    Parameters
    ----------
    ctx : click.Context
        Click execution context.
    value : str
        Value received for the parameter.

    Returns
    -------
    list
        list of values.

    Raises
    ------
    click.BadParameter
        Raises an error for malformed input.
    """
    try:
        data_modules = [k for k in data_modules_map.keys()]
        if value == "":
            config_list = data_modules
        elif value.startswith("~"):
            value = value.lstrip("~")
            config_list = value.split(",")
            for val in config_list:
                if val not in data_modules:
                    raise ValueError
            config_list = list(set(data_modules) - set(config_list))
        else:
            config_list = value.split(",")
            for val in config_list:
                if val not in data_modules:
                    raise ValueError
        return config_list
    except ValueError:
        raise click.BadParameter(f"{val} is an unknown data module")


@click.command()
@click.option(
    "-o",
    "--out",
    type=click.Path(),
    help="Path to save downloaded files",
    required=True,
)
@click.option(
    "-c",
    "--config",
    type=click.UNPROCESSED,
    callback=str2array,
    help="""Comma separated list of data modules to select/de-select for download.
            Example: iss or ~iss""",
    default="",
    show_default=True,
)
@click.option("-f", "--force", is_flag=True, help="Force redownload all data")
@click.option("-d", "--debug", is_flag=True, help="Run in debug mode.")
def fetch(out: str, config: list, force: bool, debug: bool) -> None:
    """Fetch datasets."""
    for dataset in config:
        data_modules_map[dataset](Path(out), force, debug)


@click.group()
def data() -> None:
    """Neocode data toolkit."""
    pass


@click.group()
def main() -> None:
    """Neocode CLI."""
    pass


data.add_command(fetch)
main.add_command(data)
