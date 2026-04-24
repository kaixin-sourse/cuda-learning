# Stream Experiment Notes Template

## Environment

- GPU:
- Total elements:
- Chunk size:
- Stream count:

## Results

| Version | Time(ms) | Check | Notes |
|---|---------:|-------|---|
| Pageable sequential |  72.0532 | PASS  | |
| Pinned sequential |  27.6066 | PASS  | |
| Pinned multi-stream |  13.2901 | PASS  | |

## Timeline Observations

- H2D copies:
- Kernel execution:
- D2H copies:
- Overlap observed:

## Conclusion

- Main bottleneck:
- Best version:
- Next experiment:
