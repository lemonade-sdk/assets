# Lemonade Brand Kit

This folder contains transparent PNG exports of the Lemonade logo lockup used in
the top-left of `https://lemonade-server.ai`: the Lemonade icon plus the
`Lemonade` name.

## PNG Exports

Light mode uses the website navbar wordmark color `#18181B`.
Dark mode uses the docs dark-mode primary text color `#F2EFE5`.

| File | Dimensions |
| --- | ---: |
| `lemonade-logo-light-128h.png` | 527 x 128 |
| `lemonade-logo-light-256h.png` | 1050 x 256 |
| `lemonade-logo-light-512h.png` | 2099 x 512 |
| `lemonade-logo-light-1024h.png` | 4209 x 1024 |
| `lemonade-logo-dark-128h.png` | 527 x 128 |
| `lemonade-logo-dark-256h.png` | 1050 x 256 |
| `lemonade-logo-dark-512h.png` | 2099 x 512 |
| `lemonade-logo-dark-1024h.png` | 4209 x 1024 |

## Regenerate

Run:

```sh
python3 make_brand_kit.py
```

The generator uses the bundled `source/lemonade-homepage-favicon.ico` and
`vendor/PlusJakartaSans-ExtraBold.ttf`. The layout follows the website header
ratios: icon height, a 0.25em gap, Plus Jakarta Sans ExtraBold text, and
proportional padding on a transparent canvas.
