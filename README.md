# chessfp — Chess Fingerprint

Identify a chess.com player from their moves alone, and tell anyone *which pro they play like*.

This project trains a behavioral-stylometry model on games from a curated set of professional players and streamers on chess.com (Magnus, Hikaru, Naroditsky, GothamChess, Firouzja, the Botez sisters, etc.) and exposes two things:

1. **Player ID** — given a small set of unlabeled games, predict which pro played them.
2. **"Who do you play like?"** — given *your* games, return the pro whose style is closest in embedding space.

## Why this exists

Behavioral stylometry in chess has been shown to work very well — McIlroy-Young et al. (NeurIPS 2021) hit 98% accuracy on 2,500 Lichess players using transformer architectures. What hasn't been built is a polished, public-facing tool focused on the **chess.com pro/streamer ecosystem** — that's the gap this project fills.

## Status

Early. Currently building the data pipeline.

- [x] Project scaffold
- [ ] Chess.com fetcher
- [ ] PGN parsing & feature extraction
- [ ] Model training
- [ ] "Play like a pro" inference

## Layout

```
chessengine/
├── src/chessfp/        # library code
├── scripts/            # CLI entry points (fetch, train, infer)
├── data/raw/           # raw PGN archives (gitignored)
├── notebooks/          # exploration
├── players.json        # curated player handles
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fetch data

```bash
python scripts/fetch_games.py
```

The fetcher is rate-limited (1 req/sec), resumable (skips months already on disk), and identifies itself with a contact email per chess.com's published API guidelines.

## Data source

[chess.com Published-Data API](https://www.chess.com/news/view/published-data-api) — public, no auth required, free.

## Prior art

- McIlroy-Young, Wang, Sen, Kleinberg, Anderson — *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess* (NeurIPS 2021)
- [Maia Chess](https://www.maiachess.com/)
