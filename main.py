import typer
from kgrec.cmd import train

app = typer.Typer()
app.add_typer(train.app, name='train')

if __name__ == '__main__':
    app()
