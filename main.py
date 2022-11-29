import typer
from kgrec.cmd import train, compute

app = typer.Typer()
app.add_typer(train.app, name='train')
app.add_typer(compute.app, name='compute')

if __name__ == '__main__':
    app()
