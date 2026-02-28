"""CLI for nemotron-asr-mlx.

Commands:
    transcribe <file> [--chunk-ms]   Transcribe an audio file.
    listen [--chunk-ms]              Stream from the microphone.
    convert <nemo-path> <output-dir> Convert .nemo weights to MLX safetensors.
"""

from pathlib import Path

import typer

app = typer.Typer(
    name="nemotron-asr",
    help="Nemotron ASR on Apple Silicon — streaming speech recognition.",
    add_completion=False,
)


@app.command()
def transcribe(
    file: Path = typer.Argument(..., help="Audio file to transcribe."),
    chunk_ms: int = typer.Option(
        0,
        "--chunk-ms",
        help="Chunk size in ms for streaming mode (0 = batch).",
    ),
    model_id: str = typer.Option(
        "dboris/nemotron-asr-mlx",
        "--model",
        help="HuggingFace model ID or local path.",
    ),
    beam_size: int = typer.Option(
        1,
        "--beam-size",
        help="Beam width for decoding. 1 = greedy (fastest).",
    ),
    lm: str = typer.Option(
        None,
        "--lm",
        help="Path to KenLM ARPA/binary model for shallow fusion.",
    ),
    lm_alpha: float = typer.Option(
        0.3,
        "--lm-alpha",
        help="LM interpolation weight (default 0.3).",
    ),
):
    """Transcribe an audio file."""
    from nemotron_asr_mlx.model import from_pretrained

    model = from_pretrained(model_id)

    if chunk_ms > 0:
        _streaming_transcribe(model, file, chunk_ms)
    else:
        result = model.transcribe(
            str(file),
            beam_size=beam_size,
            lm_path=lm,
            lm_alpha=lm_alpha,
        )
        typer.echo(result.text)


def _streaming_transcribe(model, file: Path, chunk_ms: int):
    """Transcribe a file in streaming mode, printing incremental output."""
    import numpy as np

    from nemotron_asr_mlx.audio import load_audio

    audio = load_audio(str(file))
    sample_rate = 16000
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    session = model.create_stream(chunk_ms=chunk_ms)

    import mlx.core as mx

    for start in range(0, len(audio), chunk_samples):
        chunk = mx.array(audio[start : start + chunk_samples])
        event = session.push(chunk)
        if event.text_delta:
            typer.echo(event.text_delta, nl=False)

    final = session.flush()
    typer.echo()  # newline


@app.command()
def listen(
    chunk_ms: int = typer.Option(
        160,
        "--chunk-ms",
        help="Chunk size in milliseconds (80, 160, 560, 1120).",
    ),
    model_id: str = typer.Option(
        "dboris/nemotron-asr-mlx",
        "--model",
        help="HuggingFace model ID or local path.",
    ),
):
    """Stream transcription from the microphone."""
    from nemotron_asr_mlx.model import from_pretrained

    model = from_pretrained(model_id)
    typer.echo("Listening... (Ctrl+C to stop)\n")

    try:
        with model.listen(chunk_ms=chunk_ms) as stream:
            for event in stream:
                if event.text_delta:
                    typer.echo(event.text_delta, nl=False)
    except KeyboardInterrupt:
        typer.echo("\nStopped.")


@app.command()
def convert(
    nemo_path: Path = typer.Argument(..., help="Path to .nemo checkpoint."),
    output_dir: Path = typer.Argument(..., help="Output directory."),
):
    """Convert a .nemo checkpoint to MLX safetensors."""
    from nemotron_asr_mlx.convert import convert_nemo_to_mlx

    convert_nemo_to_mlx(str(nemo_path), str(output_dir))


if __name__ == "__main__":
    app()
