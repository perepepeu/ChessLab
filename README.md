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
- Aprendizado guiado: ensine lances diretamente no tabuleiro, com comentário e prioridade.
- Fine-tuning de qualquer checkpoint salvo ou modelo `.npz` importado.
- MLP NumPy personalizável com 1–3 camadas ocultas de 8–256 neurônios.
- Configuração de épocas, lote, taxa de aprendizado, seed, temperatura e orçamento.
- Separação determinística por partida: 70% treino, 15% validação e 15% teste.
- Checkpoints `.npz` exportáveis e recarregáveis.
- Manifesto e telemetria por execução em `runs/`.
- Arena com validação integral de regras por `python-chess`.
- Arena com política pura, avaliação tática ou busca minimax de dois plies.
- Campeonato round-robin com troca de cores, clones para autojogo, Elo e classificação.
- Replays persistentes em PGN de todo autojogo de treino e partida de campeonato.
- Visualização dos neurônios top-k, ativações e pesos entre camadas.

## Como o aprendizado funciona

A entrada possui 773 valores: 12 planos de peças, lado a jogar e quatro direitos de roque. A saída possui 4.096 logits, um para cada par casa-de-origem/casa-de-destino. Na Arena, movimentos ilegais são mascarados antes da escolha.

Imitação minimiza entropia cruzada contra os lances dos PGNs. Autojogo usa uma atualização REINFORCE limitada, com resultado terminal e um pequeno sinal material. O modo híbrido começa com imitação e só depois explora em autojogo.

Este é um laboratório educacional e inspecionável, não um substituto para mecanismos de busca como Stockfish ou AlphaZero completo. Promoções compartilham o mesmo logit de origem/destino e preferem dama na inferência.

## Fluxo recomendado para evoluir um modelo

1. Importe PGNs variados e selecione explicitamente os conjuntos no Treinamento.
2. Faça imitação com mais de uma seed e preserve um conjunto de teste.
3. Adicione correções táticas no Mentor sem concentrar peso excessivo em uma única posição.
4. Selecione o melhor checkpoint como modelo base e faça fine-tuning com taxa menor.
5. Rode o Campeonato contra versões históricas sob o mesmo nível de busca.
6. Assista aos replays, procure colapso de diversidade ou repetição e só então promova o vencedor.

O Elo do Campeonato é local à execução e serve para comparação controlada dentro daquela liga; não corresponde a rating FIDE.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cada treino concluído salva o checkpoint em `models/` e os detalhes reproduzíveis em `runs/<id>/`.
