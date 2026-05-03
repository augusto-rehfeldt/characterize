# Terminal Video Audio

Terminal video playback should not rely on `pygame.mixer`, because pygame may be absent and terminal mode does not need a pygame window.

Use `ffplay -nodisp -autoexit -loglevel error <input>` as a subprocess when terminal playback starts, and terminate that process in player cleanup. Keep the existing `ffmpeg` plus `pygame.mixer` path for windowed playback so seek, pause, and volume controls continue to use the pygame UI.
