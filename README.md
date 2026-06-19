# ChessLab AI

Um laboratório local de xadrez em Python com interface HTML. Ele importa PGNs, aprende os lances por imitação, explora por autojogo, salva checkpoints e mostra uma amostra real dos pesos e ativações da rede.

## Começar

No PowerShell, dentro desta pasta:

```powershell
.\start.ps1
```

Ou manualmente:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Abra `http://127.0.0.1:5000`. O PGN de Kasparov fornecido já está disponível na Biblioteca.

## O que está implementado

- Importação de um ou vários arquivos `.pgn`, incluindo comentários de relógio.
- Regimes de imitação, autojogo por reforço e treino híbrido.
- MLP NumPy personalizável com 1–3 camadas ocultas de 8–256 neurônios.
- Configuração de épocas, lote, taxa de aprendizado, seed, temperatura e orçamento.
- Separação determinística por partida: 70% treino, 15% validação e 15% teste.
- Checkpoints `.npz` exportáveis e recarregáveis.
- Manifesto e telemetria por execução em `runs/`.
- Arena com validação integral de regras por `python-chess`.
- Visualização dos neurônios top-k, ativações e pesos entre camadas.

## Como o aprendizado funciona

A entrada possui 773 valores: 12 planos de peças, lado a jogar e quatro direitos de roque. A saída possui 4.096 logits, um para cada par casa-de-origem/casa-de-destino. Na Arena, movimentos ilegais são mascarados antes da escolha.

Imitação minimiza entropia cruzada contra os lances dos PGNs. Autojogo usa uma atualização REINFORCE limitada, com resultado terminal e um pequeno sinal material. O modo híbrido começa com imitação e só depois explora em autojogo.

Este é um laboratório educacional e inspecionável, não um substituto para mecanismos de busca como Stockfish ou AlphaZero completo. Promoções compartilham o mesmo logit de origem/destino e preferem dama na inferência.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cada treino concluído salva o checkpoint em `models/` e os detalhes reproduzíveis em `runs/<id>/`.
