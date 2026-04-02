# ASCII Video Playback Timeline Design

This document outlines the design for a real-time ASCII timeline beneath video frames in terminal mode.

## 1. Progress Bar Layout

- **Bar length**: 40 characters (configurable)  
- **Fill character**: `█` (U+2588)  
- **Empty character**: `–` (en dash)  
- **Enclosure**: `[` and `]`  

## 2. Time & Percentage Display

- **Format**:

  ```
  [██████████––––––––––––––––––––––––] 25% 0:15/1:20
  ```

- **Elements**:
  - 40-char bar with filled/empty segments  
  - Percentage (`25%`)  
  - Elapsed time `mm:ss` and total time `mm:ss`  

## 3. Update Frequency

- Update once per frame, immediately after printing the frame.  
- No separate buffer—simple print per loop iteration.  

## 4. Placement & Screen Management

1. Call `clear_screen()`.  
2. Print all ASCII frame lines.  
3. Print a single blank line (padding).  
4. Print the timeline line.  
5. Flush stdout.  

## 5. ANSI Color (Optional)

- If `--color` is enabled:  
  - Wrap filled portion in green: `\x1b[32m... \x1b[0m`  
  - Wrap empty portion in dim gray: `\x1b[2m... \x1b[0m`  

**Example**:

```
[\x1b[32m█████\x1b[0m\x1b[2m–––––\x1b[0m] 50% 0:40/1:20
```

## 6. Sample Frame + Timeline

```
<ASCII FRAME LINES...>

[██████████████––––––––––––––––––] 70% 0:56/1:20
```

## 7. Integration API Sketch

```python
def render_timeline(current_frame: int,
                    total_frames: int,
                    fps: float,
                    bar_length: int = 40,
                    use_color: bool = False) -> str:
    """
    Return an ASCII progress bar with percentage and elapsed/total times.
    """
    percent = current_frame / total_frames if total_frames else 0
    elapsed = current_frame / fps
    total = total_frames / fps

    def fmt(t):
        return f"{int(t)//60}:{int(t)%60:02d}"

    filled = int(bar_length * percent)
    empty = bar_length - filled
    bar = "█" * filled + "–" * empty

    if use_color:
        bar = (
            f"\x1b[32m{'█'*filled}\x1b[0m"
            f"\x1b[2m{'–'*empty}\x1b[0m"
        )

    return f"[{bar}] {percent*100:3.0f}% {fmt(elapsed)}/{fmt(total)}"
```

**Invocation**:

```python
clear_screen()
for line in out_txt:
    sys.stdout.write("".join(line) + "\n")
sys.stdout.write("\n")
sys.stdout.write(
    render_timeline(frame_count+1,
                    total_frames,
                    fps,
                    bar_length=40,
                    use_color=args.color)
    + "\n"
)
sys.stdout.flush()
```

## 8. High-Level Control Flow

```mermaid
flowchart LR
  A[Start Frame Loop] --> B{Paused?}
  B -- No --> C[Read Next Frame]
  B -- Yes --> D[Reuse Last Frame]
  C --> E[Convert to ASCII]
  D --> E
  E --> F[clear_screen()]
  F --> G[Print ASCII Lines]
  G --> H[Print Padding & Timeline]
  H --> I[Sleep to Sync FPS]
  I --> A