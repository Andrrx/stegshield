from typer.testing import CliRunner

from stegshield.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Analyze image files" in result.output
