# ChessLab AI

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

![ChessLab AI gameplay against a trained model](docs/assets/chesslab-gameplay.gif)

[English](#english) · [Português (Brasil)](#português-brasil)

## English

ChessLab AI is a local, inspectable chess-learning laboratory built with Python 3.10+ and a responsive HTML interface. It imports PGN collections, learns through imitation and self-play, supports guided corrections and checkpoint fine-tuning, and exposes real network weights and activations.

### Quick start

Windows PowerShell:

```powershell
.\start.ps1
```

Linux or macOS:

```bash
./start.sh
```

The launchers create `.venv`, install pinned dependencies, open the browser, and run ChessLab at `http://127.0.0.1:5000` by default.

For development tools:

```bash
python -m pip install -r requirements-dev.txt
```

### Configuration

The tracked [`.env.example`](.env.example) file is included in the repository. Copy it to `.env` and change values without editing application code:

```env
CHESSLAB_HOST=127.0.0.1
CHESSLAB_PORT=5000
CHESSLAB_DEBUG=false
CHESSLAB_MAX_CONTENT_MB=64
CHESSLAB_MAX_SESSIONS=100
CHESSLAB_SESSION_TTL_HOURS=12
CHESSLAB_SESSION_DB=data/chesslab.sqlite3
```

### Features

- PGN validation, multi-file import, and automatic organization by dominant player.
- Imitation learning, guided learning, bounded REINFORCE self-play, and hybrid training.
- Fine-tuning from saved or imported `.npz` checkpoints.
- Configurable NumPy MLP with inspectable neurons, activations, and weights.
- Deterministic game-aware 70/15/15 train, validation, and test splits.
- Round-robin championships with color swaps, self-play clones, Elo, and standings.
- Pure-policy, tactical, and two-ply minimax play modes.
- Persistent PGN replays for self-play and championship games.
- Persistent SQLite game sessions with TTL and LRU limits.
- Reproducible run manifests, telemetry, seeds, and versioned checkpoints.
- Cross-platform launchers, pinned dependencies, and MIT licensing.

### Learning architecture

The input contains 773 values: 12 piece planes, side to move, and four castling rights. The output contains 4,096 origin/destination logits. Illegal moves are masked before selection.

```text
PGN games ──┐
Mentor moves ├──► Board encoder ──► 12 × 8 × 8 + 5 metadata
Self-play ───┘                              │
                                           ▼
                                     773 inputs
                                           │
                                           ▼
                              1–3 configurable dense layers
                                 (8–256 neurons + ReLU)
                                           │
                                           ▼
                              4,096 origin→destination logits
                                           │
                                           ▼
                                  Legal-move mask
                                           │
                                           ▼
                                  Policy / Search ──► Move
```

Imitation training minimizes cross-entropy against PGN moves. Self-play uses a bounded REINFORCE update with terminal results and a small material signal. Hybrid training performs imitation before controlled self-play. Guided examples let a human assign preferred legal moves and priorities.

ChessLab is an educational and experimental system, not a replacement for Stockfish or a full AlphaZero implementation. Promotion moves currently share their origin/destination logit and prefer queen promotion during inference. Championship Elo is local to one controlled league and is not a FIDE rating.

### Recommended model workflow

1. Import diverse PGNs and select datasets explicitly.
2. Train imitation baselines with multiple seeds and preserve held-out games.
3. Add tactical corrections through the Mentor without overweighting a few positions.
4. Fine-tune the strongest checkpoint with a lower learning rate.
5. Run equal-budget championships against frozen historical versions.
6. Inspect replays for collapse, repetition, and regressions before promotion.

### Tests

```bash
python -m pytest -q
```

The suite covers model encoding, datasets, PGN validation, session persistence, concurrency coordination, replays, tournament scoring, and critical Flask endpoints.

---

## Português (Brasil)

O ChessLab AI é um laboratório local e inspecionável de aprendizado de xadrez, construído em Python 3.10+ com uma interface HTML responsiva. Ele importa coleções PGN, aprende por imitação e autojogo, aceita correções guiadas e fine-tuning de checkpoints, além de expor pesos e ativações reais da rede.

### Início rápido

Windows PowerShell:

```powershell
.\start.ps1
```

Linux ou macOS:

```bash
./start.sh
```

Os launchers criam a `.venv`, instalam dependências fixadas, abrem o navegador e iniciam o ChessLab em `http://127.0.0.1:5000` por padrão.

Para instalar as ferramentas de desenvolvimento:

```bash
python -m pip install -r requirements-dev.txt
```

### Configuração

O arquivo [`.env.example`](.env.example) está versionado e incluído no repositório. Copie-o para `.env` e altere as configurações sem editar o código:

```env
CHESSLAB_HOST=127.0.0.1
CHESSLAB_PORT=5000
CHESSLAB_DEBUG=false
CHESSLAB_MAX_CONTENT_MB=64
CHESSLAB_MAX_SESSIONS=100
CHESSLAB_SESSION_TTL_HOURS=12
CHESSLAB_SESSION_DB=data/chesslab.sqlite3
```

### Funcionalidades

- Validação de PGN, importação de múltiplos arquivos e organização automática por jogador dominante.
- Aprendizado por imitação, aprendizado guiado, autojogo REINFORCE limitado e treino híbrido.
- Fine-tuning de checkpoints `.npz` salvos ou importados.
- MLP NumPy configurável com neurônios, ativações e pesos inspecionáveis.
- Separação determinística por partida: 70% treino, 15% validação e 15% teste.
- Campeonatos round-robin com troca de cores, clones, Elo e classificação.
- Modos de política pura, avaliação tática e minimax de dois plies.
- Replays PGN persistentes de autojogos e campeonatos.
- Sessões de partida persistidas em SQLite, com TTL e limite LRU.
- Manifestos reproduzíveis, telemetria, seeds e checkpoints versionados.
- Launchers multiplataforma, dependências fixadas e licença MIT.

### Arquitetura de aprendizado

A entrada possui 773 valores: 12 planos de peças, lado a jogar e quatro direitos de roque. A saída possui 4.096 logits de origem/destino. Movimentos ilegais são mascarados antes da escolha.

```text
Partidas PGN ──┐
Lances Mentor ─┼──► Codificador ──► 12 × 8 × 8 + 5 metadados
Autojogo ──────┘                           │
                                          ▼
                                     773 entradas
                                          │
                                          ▼
                              1–3 camadas densas configuráveis
                                  (8–256 neurônios + ReLU)
                                          │
                                          ▼
                              4.096 logits de origem→destino
                                          │
                                          ▼
                                 Máscara de lances legais
                                          │
                                          ▼
                                 Política / Busca ──► Lance
```

O treino por imitação minimiza entropia cruzada contra os lances dos PGNs. O autojogo usa uma atualização REINFORCE limitada com resultado terminal e um pequeno sinal material. O modo híbrido realiza imitação antes do autojogo controlado. Exemplos guiados permitem que uma pessoa indique movimentos legais preferidos e suas prioridades.

O ChessLab é educacional e experimental; não substitui o Stockfish nem implementa um AlphaZero completo. Promoções compartilham o logit de origem/destino e preferem dama durante a inferência. O Elo do Campeonato é local àquela liga controlada e não corresponde a rating FIDE.

### Fluxo recomendado

1. Importe PGNs variados e selecione os datasets explicitamente.
2. Treine baselines de imitação com várias seeds e preserve partidas de teste.
3. Adicione correções táticas pelo Mentor sem dar peso excessivo a poucas posições.
4. Faça fine-tuning do melhor checkpoint com uma taxa de aprendizado menor.
5. Rode campeonatos com orçamento igual contra versões históricas congeladas.
6. Inspecione replays procurando colapso, repetição e regressões antes da promoção.

### Testes

```bash
python -m pytest -q
```

A suíte cobre codificação do modelo, datasets, validação PGN, persistência de sessões, coordenação concorrente, replays, pontuação de torneios e endpoints Flask críticos.

## License / Licença

MIT — see [LICENSE](LICENSE).
