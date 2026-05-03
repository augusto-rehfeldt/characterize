# CLI Validation And Output

Validate positive numeric CLI inputs in `parse_arguments` before collecting inputs or building character resources. Width, height, complexity, and framerate should fail with a clear `SystemExit` message instead of causing downstream divide-by-zero, empty character lists, or timing errors.

Text exports should write UTF-8 rows with `"".join(row)`, matching terminal rendering. Inserting spaces with `" ".join(row)` changes the output aspect and is inconsistent with video and preview rendering.

Avoid `shell=True` for executable integrations such as FileOptimizer. Pass an argument list to `subprocess.run` so paths containing spaces or quotes are handled by the subprocess API rather than shell parsing.
