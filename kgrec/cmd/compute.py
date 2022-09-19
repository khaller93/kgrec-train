import typer

from kgrec.datasets import parse
from kgrec.metrics import ldsd


app = typer.Typer()


@app.command(name='ldsd', help='compute LDSD similarity metric')
def ldsd_cmd(model_out_directory: str = 'model', dataset: str = 'pokemon'):
    ldsd.train(parse(dataset), model_out_directory)


if __name__ == '__main__':
    app()
